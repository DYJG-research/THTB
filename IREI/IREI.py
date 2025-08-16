import json
import pandas as pd
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bars
import argparse

# Configuration parameters (default values, can be overridden by command line)
DEFAULT_INPUT_XLSX = ''  # Input Excel file path (must contain query, response columns)
DEFAULT_OUTPUT_XLSX = ''  # Output Excel file path


# 1. Read and process Excel data (with average score)
def process_with_average_score(input_path: str, output_path: str) -> pd.DataFrame:
    print("📊 Reading Excel file...")
    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        raise RuntimeError(f"Cannot read input file {input_path}: {e}")

    # Validate required columns
    for col in ['query', 'response']:
        if col not in df.columns:
            raise ValueError(f"Input Excel missing required column: {col}")

    # ✅ Calculate basic metrics
    print("📈 Calculating length metrics...")
    df['query_length'] = df['query'].astype(str).str.len()
    df['response_length'] = df['response'].astype(str).str.len()
    df['total_length'] = df['query_length'] + df['response_length']

    # ✅ Ratio calculation (pandas vectorization)
    print("📊 Calculating response/instruction ratio...")
    df['response_instruction_ratio'] = (df['response_length'] / df['query_length']).replace([np.inf, -np.inf], np.nan).fillna(0)

    # 🔥 Pre-calculate min/max
    total_len_min = df['total_length'].min()
    total_len_max = df['total_length'].max()
    ratio_min = df['response_instruction_ratio'].min()
    ratio_max = df['response_instruction_ratio'].max()

    total_len_range = total_len_max - total_len_min
    ratio_range = ratio_max - ratio_min

    # ✅ Normalized scores (0-1)
    if total_len_range == 0:
        df['length_score'] = 0.5
    else:
        df['length_score'] = (df['total_length'] - total_len_min) / total_len_range  # 0~1

    if ratio_range == 0:
        df['ratio_score'] = 0.5
    else:
        df['ratio_score'] = (df['response_instruction_ratio'] - ratio_min) / ratio_range  # 0~1

    # ✅ Context score (weighted average)
    print("🎯 Calculating context scores...")
    df['IREI'] = (df['length_score'] + df['ratio_score']) / 2

    print(f"💾 Saving results to {output_path}...")
    df.to_excel(output_path, index=False,
                columns=['query', 'response', 'total_length',
                         'response_instruction_ratio', 'length_score', 'ratio_score', 'IREI'])

    return df

def main():
    parser = argparse.ArgumentParser(description="IREI Context Score Calculator")
    parser.add_argument('-i', '--input', default=DEFAULT_INPUT_XLSX, 
                       help=f'Input Excel file path (default: {DEFAULT_INPUT_XLSX})')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_XLSX, 
                       help=f'Output Excel file path (default: {DEFAULT_OUTPUT_XLSX})')
    
    args = parser.parse_args()
    
    try:
        # Process data with average score
        result_df = process_with_average_score(args.input, args.output)
        
        print("\n🎉 IREI calculation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
