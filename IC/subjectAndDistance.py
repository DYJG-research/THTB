import json
import logging
import re
import os
import argparse
import pandas as pd
from openai import OpenAI
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Subject identification only

def get_required_subjects(task_description):
    """Identify the academic disciplines required for the task (returns English, standard discipline names in lowercase+underscore format)"""
    prompt = f"""\\no_think You are a multidisciplinary knowledge expert. Please identify the main academic disciplines required to complete the following task. Return a comma-separated list in English only, without any explanation.

Use standard discipline names that match common English academic terminology, such as: computer_science, mathematics, physics, psychology, linguistics, etc.

Task description: {task_description}

Return only the list of discipline names in English, for example: "computer_science, mathematics, linguistics"""  # noqa: E501
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            stream=True,
        )

        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:  # Real-time accumulation
                chunk_content = chunk.choices[0].delta.content
                print(chunk_content, end="", flush=True)
                full_response += chunk_content

        cleaned_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
        cleaned_response = cleaned_response.replace("```json", '')
        cleaned_response = cleaned_response.replace("```", '')
        cleaned_response = cleaned_response.strip()

        logging.info("Model response (English):")
        logging.info(cleaned_response)
        print("\nModel response:")
        print(cleaned_response)

        subjects = [s.strip().lower().replace(' ', '_') for s in cleaned_response.split(",") if s.strip()]
        # Remove duplicates while maintaining order (deduplicate by appearance order)
        seen = set()
        unique_subjects = []
        for subj in subjects:
            if subj not in seen:
                seen.add(subj)
                unique_subjects.append(subj)
        return unique_subjects
    except Exception as e:
        print("❌ API call failed:", e)
        return []


def analyze_task(task_description, visualize=False):  # visualize parameter kept for external compatibility
    """Subject identification only, no normalization calculation, no similarity/distance matrix and ICS calculation."""
    print(f"🔍 Analyzing task: {task_description}")

    subjects = get_required_subjects(task_description)
    print("📚 Identified subjects:", subjects)

    raw_subject_count = len(subjects)

    result = {
        "task": task_description,
        "subjects": subjects,
        "raw_subject_count": raw_subject_count,
        # Compatibility with external code: explicitly not calculated, set to None
        "normalized_subject_count": None,
        "ICS": None,
    }
    return result


def process_excel_file(input_file, output_file=None, subject_dist_file=None):
    """Process Excel file to analyze subjects for each query-response pair"""
    print(f"📊 Reading Excel file: {input_file}")
    
    try:
        df = pd.read_excel(input_file)
        print(f"✅ Successfully loaded {len(df)} rows")
    except Exception as e:
        raise RuntimeError(f"Cannot read input file {input_file}: {e}")
    
    # Validate required columns
    required_columns = ['query', 'response']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        available_columns = ", ".join(df.columns)
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {available_columns}")
    
    # Process each row
    results = []
    subject_counts = {}
    
    print("🔍 Analyzing subjects for each query-response pair...")
    
    for idx, row in df.iterrows():
        query = str(row['query'])
        response = str(row['response'])
        
        # Combine query and response for analysis
        combined_text = f"Query: {query}\nResponse: {response}"
        
        print(f"\n--- Processing row {idx + 1}/{len(df)} ---")
        result = analyze_task(combined_text)
        
        # Add to results
        row_result = row.to_dict()
        row_result.update({
            'identified_subjects': ', '.join(result['subjects']),
            'subject_count': result['raw_subject_count']
        })
        results.append(row_result)
        
        # Count subject occurrences
        for subject in result['subjects']:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    if output_file:
        results_df.to_excel(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
    
    # Create subject distribution DataFrame
    if subject_dist_file:
        subject_dist_df = pd.DataFrame([
            {'subject': subject, 'count': count, 'percentage': count/len(df)*100}
            for subject, count in sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        subject_dist_df.to_excel(subject_dist_file, index=False)
        print(f"✅ Subject distribution saved to: {subject_dist_file}")
    
    # Print summary
    print(f"\n📊 Analysis Summary:")
    print(f"   - Total samples processed: {len(df)}")
    print(f"   - Unique subjects identified: {len(subject_counts)}")
    print(f"   - Average subjects per sample: {sum(subject_counts.values())/len(df):.2f}")
    
    return results_df, subject_dist_df if subject_dist_file else None


def main():
    parser = argparse.ArgumentParser(description="Subject and Distance Analysis Tool")
    parser.add_argument("-i", "--input", required=True, help="Input Excel file path")
    parser.add_argument("-o", "--output", help="Output Excel file path for results")
    parser.add_argument("-s", "--subject-dist", help="Output Excel file path for subject distribution")
    parser.add_argument("--single-task", help="Analyze a single task description")
    
    args = parser.parse_args()
    
    if args.single_task:
        # Single task analysis
        result = analyze_task(args.single_task)
        print(f"\n📊 Analysis Result:")
        print(f"   - Task: {result['task']}")
        print(f"   - Subjects: {result['subjects']}")
        print(f"   - Subject count: {result['raw_subject_count']}")
        return 0
    
    if not args.input:
        print("❌ Error: Input file is required")
        return 1
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file does not exist: {args.input}")
        return 1
    
    try:
        # Process Excel file
        results_df, subject_dist_df = process_excel_file(
            input_file=args.input,
            output_file=args.output,
            subject_dist_file=args.subject_dist
        )
        
        print("\n🎉 Subject analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
