import pandas as pd
import json
import os
import sys
import traceback
import logging
import time
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add current directory to path to ensure subjectAndDistance module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # Try to import analyze_task function
    from subjectAndDistance import analyze_task, API_BASE_URL
    
    # Print API URL to confirm using the correct address
    print(f"Using API address: {API_BASE_URL}")
except ImportError:
    print("Error: Cannot import subjectAndDistance module. Please ensure subjectAndDistance.py is in the same directory.")
    sys.exit(1)

# Try to import tqdm for progress display, create simple replacement if not available
try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError:
    print("Note: tqdm library not installed, will use simple progress display. Install tqdm for better progress bar experience.")
    tqdm_available = False
    def tqdm(iterable, **kwargs):
        total = kwargs.get('total', None)
        desc = kwargs.get('desc', '')
        for i, item in enumerate(iterable):
            if i % 10 == 0:
                print(f"{desc}: {i}/{total if total else '?'}", end='\r')
            yield item

# Input and output file paths
DEFAULT_INPUT_FILE = "train-alpaca.xlsx"
DEFAULT_OUTPUT_FILE = "train-alpaca-subjects.xlsx"
DEFAULT_CHECKPOINT_FILE = "checkpoint.json"  # Checkpoint resume file
DEFAULT_SUBJECT_DIST_FILE = "subject_distribution.xlsx"  # Subject distribution statistics file

# Processing configuration
SAVE_INTERVAL = 2000  # Save results after processing this many rows
TEST_MODE = False  # Test mode, set to False to process all data
MAX_ERRORS = 5  # Maximum allowed consecutive errors

def load_checkpoint(checkpoint_file):
    """Load last processing checkpoint information"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            print(f"✅ Loaded checkpoint: last processed row {checkpoint.get('last_index', 0)}")
            return checkpoint
        except Exception as e:
            print(f"⚠️ Failed to read checkpoint file: {e}")
    return {"last_index": 0, "processed_count": 0, "success_count": 0, "error_count": 0}

def save_checkpoint(checkpoint_file, idx, processed_count, success_count, error_count):
    """Save current processing checkpoint information"""
    checkpoint = {
        "last_index": idx,
        "processed_count": processed_count,
        "success_count": success_count,
        "error_count": error_count,
        "timestamp": datetime.now().isoformat()
    }
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved checkpoint: currently processed to row {idx}")

def main():
    parser = argparse.ArgumentParser(description="Analyze interdisciplinary complexity of each query row in Excel")
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT_FILE, help="Input Excel file path, default: %(default)s")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FILE, help="Output Excel file path, default: %(default)s")
    parser.add_argument("-c", "--checkpoint", default=DEFAULT_CHECKPOINT_FILE, help="Checkpoint resume file path, default: %(default)s")
    parser.add_argument("-s", "--subjects-output", default=DEFAULT_SUBJECT_DIST_FILE, help="Subject distribution statistics Excel path, default: %(default)s")
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL, help="Save interval (rows), default: %(default)s")
    parser.add_argument("--max-errors", type=int, default=MAX_ERRORS, help="Maximum consecutive errors allowed, default: %(default)s")
    parser.add_argument("--test-mode", action="store_true", help="Enable test mode (process limited rows)")
    
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    checkpoint_file = args.checkpoint
    subject_dist_file = args.subjects_output
    save_interval = args.save_interval
    max_errors = args.max_errors
    test_mode = args.test_mode
    
    print(f"📊 Starting analysis of Excel file: {input_file}")
    
    # Validate input file
    if not os.path.exists(input_file):
        print(f"❌ Error: File {input_file} does not exist")
        return 1
    
    # Load checkpoint information
    checkpoint = load_checkpoint(checkpoint_file)
    last_processed_idx = checkpoint.get("last_index", 0)
    processed_count = checkpoint.get("processed_count", 0)
    success_count = checkpoint.get("success_count", 0)
    error_count = checkpoint.get("error_count", 0)
    
    # Read Excel file
    try:
        df = pd.read_excel(input_file)
        print(f"✅ Successfully read file, {len(df)} rows of data")
        
        # If there's a checkpoint, show recovery status
        if last_processed_idx > 0:
            print(f"🔄 Will continue from row {last_processed_idx}, already successfully processed {success_count} rows")
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        traceback.print_exc()
        return 1
    
    # Ensure query column exists
    if 'query' not in df.columns:
        available_columns = ", ".join(df.columns)
        print(f"❌ Error: Excel file does not have 'query' column. Available columns: {available_columns}")
        return 1
    
    # Initialize new columns if they don't exist
    if 'subjects' not in df.columns:
        df['subjects'] = None
    if 'raw_subject_count' not in df.columns:
        df['raw_subject_count'] = None
    if 'normalized_subject_count' not in df.columns:
        df['normalized_subject_count'] = None
    if 'ICS' not in df.columns:
        df['ICS'] = None
    
    start_time = time.time()
    total_rows = len(df)
    consecutive_errors = 0
    
    # Test mode: only process first 100 rows
    if test_mode:
        total_rows = min(100, total_rows)
        print(f"⚠️ Test mode enabled, will only process first {total_rows} rows")
    
    print(f"🔄 Starting processing from row {last_processed_idx}...")
    
    # Process each row
    try:
        for idx in tqdm(range(last_processed_idx, total_rows), desc="Analyzing queries", unit="row"):
            try:
                query = df.at[idx, 'query']
                if pd.isna(query) or str(query).strip() == "":
                    print(f"⚠️ Row {idx}: Empty query, skipping")
                    continue
                
                # Analyze task
                result = analyze_task(str(query))
                
                # Store results
                df.at[idx, 'subjects'] = json.dumps(result['subjects'], ensure_ascii=False)
                df.at[idx, 'raw_subject_count'] = result['raw_subject_count']
                df.at[idx, 'normalized_subject_count'] = result.get('normalized_subject_count', None)
                df.at[idx, 'ICS'] = result.get('ICS', None)
                
                processed_count += 1
                success_count += 1
                consecutive_errors = 0  # Reset consecutive error counter
                
                # Periodic save
                if (idx + 1) % save_interval == 0:
                    try:
                        df.to_excel(output_file, index=False)
                        save_checkpoint(checkpoint_file, idx + 1, processed_count, success_count, error_count)
                        print(f"💾 Interim save completed, processed {idx + 1}/{total_rows} rows")
                    except Exception as save_error:
                        print(f"⚠️ Interim save failed: {save_error}")
                
            except Exception as e:
                error_count += 1
                consecutive_errors += 1
                print(f"❌ Row {idx} processing error: {e}")
                
                if consecutive_errors >= max_errors:
                    print(f"❌ Too many consecutive errors ({consecutive_errors}), terminating processing")
                    break
                
                # Log detailed error information
                logging.error(f"Row {idx} error details: {traceback.format_exc()}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error during processing: {e}")
        traceback.print_exc()
    
    # Normalize subject counts (Min-Max normalization)
    print("🔄 Performing Min-Max normalization...")
    try:
        valid_counts = df['raw_subject_count'].dropna()
        if len(valid_counts) > 0:
            min_raw = valid_counts.min()
            max_raw = valid_counts.max()
            
            if max_raw > min_raw:
                df['normalized_subject_count'] = df['raw_subject_count'].apply(
                    lambda x: (x - min_raw) / (max_raw - min_raw) if pd.notna(x) else None
                )
            else:
                # If all values are the same, set normalized value to 0.5
                df['normalized_subject_count'] = df['raw_subject_count'].apply(
                    lambda x: 0.5 if pd.notna(x) else None
                )
            print(f"✅ Completed global Min-Max normalization: min={min_raw}, max={max_raw}")
    except Exception as e:
        print(f"❌ Normalization calculation failed: {e}")
    
    # Save final results
    try:
        df.to_excel(output_file, index=False)
        print(f"✅ Analysis completed, results saved to {output_file}")
        # Update checkpoint one last time
        save_checkpoint(checkpoint_file, total_rows, processed_count, success_count, error_count)
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        # Try to save as CSV as backup
        try:
            csv_output = output_file.replace('.xlsx', '.csv')
            df.to_csv(csv_output, index=False)
            print(f"💾 Results saved as CSV: {csv_output}")
        except:
            print("❌ CSV save also failed, please check file permissions or disk space")
    
    # Calculate total time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    # Only perform statistical analysis if there are successfully processed rows
    if success_count > 0:
        print("\n📊 Statistical Analysis:")
        print(f"   - Total processed: {processed_count} queries, successful: {success_count}, failed: {error_count}")
        print(f"   - Total time: {hours}h {minutes}m {seconds}s")
        
        # Filter out null values before calculating statistics
        valid_raw_counts = df['raw_subject_count'].dropna()
        valid_norm_counts = df['normalized_subject_count'].dropna()
        
        if len(valid_raw_counts) > 0:
            print(f"   - Average raw subject count: {valid_raw_counts.mean():.2f}")
            print(f"   - Max raw subject count: {valid_raw_counts.max():.2f}")
            print(f"   - Min raw subject count: {valid_raw_counts.min():.2f}")
            
        if len(valid_norm_counts) > 0:
            print(f"   - Average normalized subject count: {valid_norm_counts.mean():.2f}")
        
        # Subject distribution
        all_subjects = []
        for subjects_json in df['subjects'].dropna():
            try:
                subjects = json.loads(subjects_json)
                all_subjects.extend(subjects)
            except:
                pass
        
        if all_subjects:
            subjects_count = pd.Series(all_subjects).value_counts()
            print("\n📈 Subject Distribution (Top 20):")
            for subject, count in subjects_count.head(20).items():
                print(f"   - {subject}: {count} times")
            
            # Save subject distribution statistics
            subjects_df = pd.DataFrame({
                "subject": subjects_count.index,
                "count": subjects_count.values,
                "percentage": (subjects_count.values / subjects_count.sum() * 100).round(2)
            })
            subjects_df.to_excel(subject_dist_file, index=False)
            print(f"✅ Subject distribution statistics saved to {subject_dist_file}")
    
    # Clean up checkpoint file if processing completed successfully
    if success_count > 0 and os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print("🧹 Checkpoint file cleaned up")
        except Exception:
            pass
    
    print("🎉 Analysis completed!")
    return 0

if __name__ == "__main__":
    try:
        # Start main process
        exit(main())
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
        exit(1)
    except Exception as e:
        print(f"❌ Unhandled error occurred: {e}")
        traceback.print_exc()
        exit(1)
    finally:
        print("\n📋 Analysis ended")
