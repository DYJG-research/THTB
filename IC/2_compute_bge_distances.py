import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import time
import argparse

# BGE model configuration - can be overridden by environment variables or command line parameters
DEFAULT_MODEL_PATH = "./models/bge-large-en-v1.5"  # Default local BGE model path

OUTPUT_DIR = "bge_outputs"
DISTANCE_MATRIX_FILE = os.path.join(OUTPUT_DIR, "subject_distance_matrix.pkl")
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "subject_embeddings.pkl")
HEATMAP_FILE = os.path.join(OUTPUT_DIR, "distance_heatmap.png")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Compute semantic distances between subjects using BGE model')
    parser.add_argument('--gpu', type=int, default=None, 
                       help='Specify GPU device ID (e.g.: --gpu 0 to use GPU 0)')
    parser.add_argument('--cpu', action='store_true', 
                       help='Force use CPU (ignore GPU)')
    parser.add_argument('--input', type=str, default="subject_descriptions.xlsx",
                       help='Input file path (default: subject_descriptions.xlsx)')
    parser.add_argument('--output-dir', type=str, default="bge_outputs",
                       help='Output directory (default: bge_outputs)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='BGE model path (default uses environment variable BGE_MODEL_PATH or local path)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    return parser.parse_args()

def get_model_path(args):
    """Get model path"""
    # Priority: command line argument > environment variable > default path
    if args.model_path:
        return args.model_path
    elif os.getenv('BGE_MODEL_PATH'):
        return os.getenv('BGE_MODEL_PATH')
    else:
        return DEFAULT_MODEL_PATH

def get_device(gpu_id=None, force_cpu=False):
    """Get computing device"""
    if force_cpu:
        device = torch.device("cpu")
        print(f"🖥️ Force using CPU")
        return device
    
    if gpu_id is not None:
        if torch.cuda.is_available():
            if gpu_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{gpu_id}")
                print(f"🎮 Using specified GPU: cuda:{gpu_id}")
                print(f"   GPU name: {torch.cuda.get_device_name(gpu_id)}")
                print(f"   GPU memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
                return device
            else:
                print(f"⚠️ Warning: GPU {gpu_id} does not exist, available GPU count: {torch.cuda.device_count()}")
                print("🔄 Automatically selecting available GPU...")
        else:
            print("⚠️ Warning: CUDA not available, will use CPU")
            return torch.device("cpu")
    
    # Auto select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🎮 Auto-selected GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("🖥️ Using CPU")
    
    return device

def ensure_output_dir(output_dir):
    """Ensure output directory exists"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")

def load_bge_model(device, model_path):
    """Load BGE model"""
    print(f"⏳ Loading local BGE model: {model_path}...")
    try:
        # Check if local model path exists
        if not os.path.exists(model_path):
            print(f"❌ Error: Local model path does not exist: {model_path}")
            print("Please ensure model files are downloaded and placed in the correct directory")
            return None
        
        # Use sentence_transformers to load model
        model = SentenceTransformer(model_path, device=device)
        print(f"✅ BGE model loaded successfully")
        print(f"   Model max sequence length: {model.max_seq_length}")
        print(f"   Model device: {model.device}")
        
        return model
    except Exception as e:
        print(f"❌ Failed to load BGE model: {e}")
        return None

def get_embedding(text, model):
    """Get text embedding using BGE model"""
    return model.encode(text, convert_to_tensor=True)

def compute_distance_matrix(subject_descriptions, model, device, batch_size=64):
    """Compute distance matrix between all subjects using BGE model's built-in similarity method"""
    subjects = list(subject_descriptions.keys())
    descriptions = list(subject_descriptions.values())
    
    print(f"🧮 Computing embeddings for {len(subjects)} subjects...")
    
    # Batch encode all descriptions
    subject_embeddings = {}
    all_embeddings = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(descriptions), batch_size), desc="Computing embeddings", unit="batch"):
        batch_descriptions = descriptions[i:i+batch_size]
        batch_subjects = subjects[i:i+batch_size]
        
        # Get embeddings for this batch
        batch_embeddings = model.encode(batch_descriptions, convert_to_tensor=True, device=device)
        
        # Store individual embeddings
        for j, subject in enumerate(batch_subjects):
            embedding = batch_embeddings[j]
            subject_embeddings[subject] = embedding.cpu().numpy()
            all_embeddings.append(embedding)
    
    # Stack all embeddings
    all_embeddings = torch.stack(all_embeddings)
    
    print(f"🧮 Computing pairwise distances...")
    
    # Compute cosine similarity matrix
    similarity_matrix = model.similarity(all_embeddings, all_embeddings)
    
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix.cpu().numpy()
    
    # Ensure diagonal is 0 (distance from subject to itself)
    np.fill_diagonal(distance_matrix, 0)
    
    print(f"✅ Distance matrix computation completed")
    print(f"   Matrix shape: {distance_matrix.shape}")
    print(f"   Min distance: {distance_matrix.min():.4f}")
    print(f"   Max distance: {distance_matrix.max():.4f}")
    print(f"   Mean distance: {distance_matrix.mean():.4f}")
    
    return distance_matrix, subjects, subject_embeddings

def visualize_distance_matrix(distance_matrix, subjects, output_dir):
    """Visualize distance matrix as heatmap"""
    print("📊 Creating distance matrix heatmap...")
    
    plt.figure(figsize=(15, 12))
    
    # Create heatmap
    sns.heatmap(distance_matrix, 
                xticklabels=subjects, 
                yticklabels=subjects,
                cmap='viridis',
                annot=False,
                fmt='.3f',
                cbar_kws={'label': 'Semantic Distance'})
    
    plt.title('Subject Semantic Distance Matrix', fontsize=16, pad=20)
    plt.xlabel('Subjects', fontsize=12)
    plt.ylabel('Subjects', fontsize=12)
    
    # Adjust label size and rotation
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8)
    
    # Save image
    heatmap_file = os.path.join(output_dir, "distance_heatmap.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"✅ Distance matrix heatmap saved to: {heatmap_file}")
    plt.close()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set output directory
    OUTPUT_DIR = args.output_dir
    DISTANCE_MATRIX_FILE = os.path.join(OUTPUT_DIR, "subject_distance_matrix.pkl")
    EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "subject_embeddings.pkl")
    
    start_time = time.time()
    print("🚀 Starting BGE model subject distance calculation program...")
    
    # Get model path
    model_path = get_model_path(args)
    print(f"🔄 Using model path: {model_path}")

    # Get computing device
    device = get_device(args.gpu, args.cpu)
    
    # Ensure output directory exists
    ensure_output_dir(OUTPUT_DIR)
    
    # Load subject description data
    input_file = args.input
    print(f"📂 Loading subject description data: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"✅ Successfully loaded data, {len(df)} rows")
    except Exception as e:
        print(f"❌ Cannot load file {input_file}: {e}")
        return
    
    # Validate required columns
    if 'subject' not in df.columns or 'description' not in df.columns:
        available_columns = ", ".join(df.columns)
        print(f"❌ Missing required columns: subject, description. Available columns: {available_columns}")
        return
    
    # Load BGE model
    model = load_bge_model(device, model_path)
    if model is None:
        print("❌ Model loading failed, program terminated")
        return
    
    # Create subject description dictionary
    print("🔄 Processing subject description data...")
    subject_descriptions = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing subject descriptions", unit="item"):
        subject = row['subject']
        description = row['description']
        
        if pd.isna(description) or description == "":
            print(f"⚠️ Warning: Subject '{subject}' has no description")
            continue
            
        subject_descriptions[subject] = description
    
    print(f"✅ Successfully loaded descriptions for {len(subject_descriptions)} subjects")
    
    # Compute distance matrix
    distance_matrix, subjects, subject_embeddings = compute_distance_matrix(subject_descriptions, model, device, args.batch_size)
    
    # Save embeddings
    print("💾 Saving embeddings...")
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(subject_embeddings, f)
    print(f"✅ Embeddings saved to: {EMBEDDINGS_FILE}")
    
    # Save distance matrix and corresponding subject list
    print("💾 Saving distance matrix...")
    with open(DISTANCE_MATRIX_FILE, 'wb') as f:
        pickle.dump({'matrix': distance_matrix, 'subjects': subjects}, f)
    print(f"✅ Distance matrix saved to: {DISTANCE_MATRIX_FILE}")
    
    # Visualize distance matrix
    visualize_distance_matrix(distance_matrix, subjects, OUTPUT_DIR)
    
    # Save as CSV format for easy viewing
    print("💾 Saving distance matrix in CSV format...")
    distance_df = pd.DataFrame(distance_matrix, index=subjects, columns=subjects)
    distance_csv = os.path.join(OUTPUT_DIR, "subject_distance_matrix.csv")
    distance_df.to_csv(distance_csv)
    print(f"✅ Distance matrix CSV saved to: {distance_csv}")
    
    # Display statistics
    print("\n📊 Distance matrix statistics:")
    print(f"   - Matrix size: {distance_matrix.shape}")
    print(f"   - Min distance: {distance_matrix.min():.4f}")
    print(f"   - Max distance: {distance_matrix.max():.4f}")
    print(f"   - Mean distance: {distance_matrix.mean():.4f}")
    print(f"   - Standard deviation: {distance_matrix.std():.4f}")
    
    # Display device usage
    if device.type == 'cuda':
        print(f"\n🎮 GPU usage:")
        print(f"   - GPU memory used: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"   - GPU memory cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    # Display processing time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n⏱️ Total processing time: {total_time:.2f} seconds")
    print("🎉 Program execution completed!")

if __name__ == "__main__":
    main()
