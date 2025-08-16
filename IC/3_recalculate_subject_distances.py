import os
import pandas as pd
import numpy as np
import pickle
import json
import ast
from tqdm import tqdm
import argparse

# Configuration
DEFAULT_INPUT_FILE = "subject_normalized.xlsx"
DEFAULT_OUTPUT_FILE = "subject_normalized_bge.xlsx"
DEFAULT_DISTANCE_MATRIX_FILE = "bge_outputs/subject_distance_matrix.pkl"

# Checkpoint configuration
CHECKPOINT_FILE = "recalculate_distances_checkpoint.json"
BATCH_SIZE = 500  # Save checkpoint after processing this many rows


def load_checkpoint():
    """Load processing progress checkpoint"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'processed_rows': 0}

def save_checkpoint(processed_rows):
    """Save current processing progress to checkpoint file"""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump({'processed_rows': processed_rows}, f)

def load_distance_matrix(matrix_file):
    """Load pre-computed subject distance matrix"""
    try:
        with open(matrix_file, 'rb') as f:
            data = pickle.load(f)
            distance_matrix = data['matrix']
            subjects = data['subjects']
            
            # Create subject to index mapping
            subject_to_idx = {subject: i for i, subject in enumerate(subjects)}
            
            print(f"✅ Successfully loaded distance matrix with {len(subjects)} subjects")
            return distance_matrix, subject_to_idx
    except Exception as e:
        print(f"❌ Error loading distance matrix file {matrix_file}: {e}")
        return None, None

def parse_subjects(subjects_str):
    """Parse subject list from string"""
    if isinstance(subjects_str, str):
        try:
            # Try to parse as list string
            return ast.literal_eval(subjects_str)
        except:
            # If failed, might be comma-separated string
            return [s.strip() for s in subjects_str.split(',') if s.strip()]
    elif isinstance(subjects_str, list):
        return subjects_str
    else:
        return []

def calculate_average_distance(subjects, distance_matrix, subject_to_idx):
    """Calculate average distance between a group of subjects"""
    n = len(subjects)
    if n <= 1:
        return 0.0
    
    # Collect all subjects that can be found in the distance matrix
    valid_subjects = []
    valid_indices = []
    for subject in subjects:
        subject_norm = subject.lower().replace(' ', '_')
        if subject_norm in subject_to_idx:
            valid_subjects.append(subject_norm)
            valid_indices.append(subject_to_idx[subject_norm])
    
    n_valid = len(valid_subjects)
    if n_valid <= 1:
        return 0.0
    
    # Calculate average distance
    total_distance = 0.0
    pair_count = 0
    
    for i in range(n_valid):
        for j in range(i+1, n_valid):
            idx1 = valid_indices[i]
            idx2 = valid_indices[j]
            distance = distance_matrix[idx1, idx2]
            total_distance += distance
            pair_count += 1
    
    if pair_count == 0:
        return 0.0
    
    return total_distance / pair_count


def main():
    # Declare global variables first
    global CHECKPOINT_FILE, BATCH_SIZE
    
    parser = argparse.ArgumentParser(description="Recalculate subject semantic distances and ICS scores based on pre-computed distance matrix")
    parser.add_argument('-i', '--input', default=DEFAULT_INPUT_FILE, help='Input Excel file path, default: %(default)s')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_FILE, help='Output Excel file path, default: %(default)s')
    parser.add_argument('-m', '--distance-matrix', dest='distance_matrix', default=DEFAULT_DISTANCE_MATRIX_FILE, help='Distance matrix pkl file path, default: %(default)s')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for checkpoint saving, default: %(default)s')
    parser.add_argument('--checkpoint', default=CHECKPOINT_FILE, help='Checkpoint file path, default: %(default)s')
    
    args = parser.parse_args()
    
    # Update global variables
    CHECKPOINT_FILE = args.checkpoint
    BATCH_SIZE = args.batch_size
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file does not exist: {args.input}")
        return 1
    
    # Load distance matrix
    print(f"📊 Loading distance matrix: {args.distance_matrix}")
    distance_matrix, subject_to_idx = load_distance_matrix(args.distance_matrix)
    if distance_matrix is None or subject_to_idx is None:
        print("❌ Cannot load distance matrix, program terminated")
        return 1
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    start_row = checkpoint['processed_rows']
    print(f"🔄 Starting from row {start_row}")
    
    # Load Excel file
    print(f"📂 Loading file: {args.input}")
    try:
        df = pd.read_excel(args.input)
        total_rows = len(df)
        print(f"✅ Successfully loaded {total_rows} rows of data")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return 1
    
    # Validate required columns
    required_columns = ['subjects']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        available_columns = ", ".join(df.columns)
        print(f"❌ Missing required columns: {missing_columns}. Available columns: {available_columns}")
        return 1
    
    # Add new columns
    if 'bge_semantic_distance' not in df.columns:
        df['bge_semantic_distance'] = 0.0
    if 'bge_ICS' not in df.columns:
        df['bge_ICS'] = 0.0
    
    # Process each row
    print(f"🧮 Processing semantic distance calculations...")
    for i in tqdm(range(start_row, total_rows), desc="Calculating subject semantic distances"):
        row = df.iloc[i]
        
        # Parse subject list
        subjects = parse_subjects(row['subjects'])
        
        if subjects and len(subjects) > 1:
            # Calculate BGE-based average semantic distance
            avg_distance = calculate_average_distance(subjects, distance_matrix, subject_to_idx)
            df.at[i, 'bge_semantic_distance'] = avg_distance
            
            # Calculate new ICS score
            # Assume ICS = normalized_subject_count + lambda * semantic_distance

            normalized_subject_count = row['normalized_subject_count']
            bge_ics = normalized_subject_count + avg_distance
            df.at[i, 'bge_ICS'] = round(bge_ics, 2)

        else:
            # If only 0 or 1 subject, distance is 0, ICS equals normalized subject count
            df.at[i, 'bge_semantic_distance'] = 0.0
            df.at[i, 'bge_ICS'] = row['normalized_subject_count']
        
        # Periodically save checkpoint
        if (i + 1) % BATCH_SIZE == 0 or i == total_rows - 1:
            save_checkpoint(i + 1)
            print(f"✅ Processed {i+1}/{total_rows} rows, progress {(i+1)/total_rows*100:.1f}%")
            
            # Also save current progress results
            interim_output = f"{os.path.splitext(args.output)[0]}_interim.xlsx"
            df.to_excel(interim_output, index=False)
            print(f"💾 Interim results saved to: {interim_output}")
    
    # Save final results
    print(f"💾 Saving results to: {args.output}")
    df.to_excel(args.output, index=False)
    
    # Print summary statistics
    print(f"\n📊 Processing Summary:")
    print(f"   - Total rows processed: {total_rows}")
    if 'bge_semantic_distance' in df.columns:
        distances = df['bge_semantic_distance'].dropna()
        if len(distances) > 0:
            print(f"   - Average semantic distance: {distances.mean():.4f}")
            print(f"   - Distance standard deviation: {distances.std():.4f}")
            print(f"   - Min distance: {distances.min():.4f}")
            print(f"   - Max distance: {distances.max():.4f}")
    
    if 'bge_ICS' in df.columns:
        ics_scores = df['bge_ICS'].dropna()
        if len(ics_scores) > 0:
            print(f"   - Average BGE ICS score: {ics_scores.mean():.4f}")
            print(f"   - ICS score standard deviation: {ics_scores.std():.4f}")
    
    print("🎉 Processing completed!")
    
    # Clean up checkpoint file
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            print("🧹 Checkpoint file cleaned up")
        except Exception:
            pass
    
    return 0


if __name__ == "__main__":
    exit(main())
