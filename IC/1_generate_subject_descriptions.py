import os
import pandas as pd
import json
import time
from openai import OpenAI
import re
from tqdm import tqdm
import argparse
from pathlib import Path

def load_api_config():
    """Load API configuration from environment variables or config file"""
    # Try to get API configuration from environment variables first
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base_url = os.environ.get("OPENAI_API_BASE")
    model_name = os.environ.get("OPENAI_MODEL")
    
    # If environment variables not found, try to load from config file
    if not api_key or not api_base_url or not model_name:
        try:
            # Look for config file in parent directory
            script_dir = Path(__file__).parent
            config_path = script_dir.parent / "config_template.json"
            
            if config_path.exists():
                print(f"Loading API configuration from: {config_path}")
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                llm_config = config.get('llm_api_config', {})
                api_key = llm_config.get('api_key', 'EMPTY')
                api_base_url = llm_config.get('api_base')
                model_name = llm_config.get('model')
                
                print(f"Loaded API config - Base: {api_base_url}, Model: {model_name}")
            else:
                print(f"Config file not found at: {config_path}")
        except Exception as e:
            print(f"Failed to load config file: {e}")
    
    if not api_key or not api_base_url or not model_name:
        raise ValueError("API configuration not complete! Please set OPENAI_API_KEY, OPENAI_API_BASE and OPENAI_MODEL environment variables, or ensure config_template.json exists with llm_api_config section.")
    
    return api_key, api_base_url, model_name

# Load API configuration
API_KEY, API_BASE_URL, MODEL_NAME = load_api_config()

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)

# File-related default paths
INPUT_FILE = "subject_distribution.xlsx"
OUTPUT_FILE = "subject_descriptions.xlsx"
CHECKPOINT_FILE = "subject_descriptions_checkpoint.json"

def load_checkpoint():
    """Load checkpoint to restore processed progress"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'processed_subjects': [], 'descriptions': {}}

def save_checkpoint(checkpoint_data):
    """Save checkpoint data"""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

def generate_subject_description(subject, retry=3):
    """Generate description for a subject"""
    prompt = f"""\\no_think As an academic expert, please provide a comprehensive description of the discipline of {subject}. 
Include the scope, key concepts, research methods, and the significance of this discipline.
Your description should be thorough, academic in nature, and approximately 3-5 sentences in length.
Focus only on providing factual information about this discipline.
"""
    
    for attempt in range(retry):
        try:
            print(f"\nGenerating description for '{subject}'...")
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Low temperature for more consistent results
                max_tokens=500,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    print(chunk_content, end="", flush=True)
                    full_response += chunk_content
            
            # Clean response
            cleaned_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
            cleaned_response = cleaned_response.replace("```json", '')
            cleaned_response = cleaned_response.replace("```", '')
            cleaned_response = cleaned_response.strip()
            
            print(f"\n\nDescription generation for '{subject}' completed!")
            time.sleep(1)  # Avoid API requests being too frequent
            return cleaned_response
            
        except Exception as e:
            print(f"Attempt {attempt+1}/{retry} failed: {e}")
            if attempt < retry - 1:
                wait_time = 5 * (attempt + 1)
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Unable to generate description for '{subject}'")
                return f"The discipline of {subject} is a field of academic study."

def main():
    # Use global paths
    input_file = INPUT_FILE
    output_file = OUTPUT_FILE
    
    print(f"Loading subject data: {input_file}")
    
    try:
        df = pd.read_excel(input_file)
        print(f"✅ Successfully loaded {len(df)} rows")
    except Exception as e:
        raise RuntimeError(f"Cannot read input file {input_file}: {e}")
    
    # Validate required columns
    if 'subject' not in df.columns:
        available_columns = ", ".join(df.columns)
        raise ValueError(f"Missing required column: subject. Available columns: {available_columns}")
    
    # Get unique subject list
    subjects = df['subject'].unique().tolist()
    print(f"Found {len(subjects)} unique subjects")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_subjects = checkpoint['processed_subjects']
    descriptions = checkpoint['descriptions']
    
    print(f"Already have descriptions for {len(processed_subjects)} subjects")
    
    # Filter out unprocessed subjects
    subjects_to_process = [s for s in subjects if s not in processed_subjects]
    
    if not subjects_to_process:
        print("All subjects already have descriptions, no processing needed")
    else:
        print(f"Need to process {len(subjects_to_process)} subjects")
        
        # Process each subject
        for subject in tqdm(subjects_to_process, desc="Generating subject descriptions"):
            description = generate_subject_description(subject)
            descriptions[subject] = description
            processed_subjects.append(subject)
            
            # Save checkpoint every 5 processed subjects
            if len(processed_subjects) % 5 == 0:
                checkpoint['processed_subjects'] = processed_subjects
                checkpoint['descriptions'] = descriptions
                save_checkpoint(checkpoint)
                print(f"Checkpoint saved, processed {len(processed_subjects)} subjects")
    
    # Create output DataFrame
    output_data = []
    for subject in subjects:
        description = descriptions.get(subject, "")
        output_data.append({
            'subject': subject, 
            'description': description
        })
    
    output_df = pd.DataFrame(output_data)
    
    # Save to Excel
    output_df.to_excel(output_file, index=False)
    print(f"All subject descriptions saved to {output_file}")
    
    # Final checkpoint save
    checkpoint['processed_subjects'] = list(set(processed_subjects))
    checkpoint['descriptions'] = descriptions
    save_checkpoint(checkpoint)
    
    # Print summary
    print(f"\n📊 Generation Summary:")
    print(f"   - Total unique subjects: {len(subjects)}")
    print(f"   - Descriptions generated: {len(descriptions)}")
    print(f"   - Average description length: {sum(len(desc) for desc in descriptions.values())/len(descriptions):.1f} characters")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate brief descriptions for subject list")
    parser.add_argument('-i', '--input', default=INPUT_FILE, help='Input subject distribution Excel path, default: %(default)s')
    parser.add_argument('-o', '--output', default=OUTPUT_FILE, help='Output subject description Excel path, default: %(default)s')
    parser.add_argument('-c', '--checkpoint', default=CHECKPOINT_FILE, help='Checkpoint file path, default: %(default)s')
    args = parser.parse_args()

    # Override global file paths
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    CHECKPOINT_FILE = args.checkpoint

    main()
