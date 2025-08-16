import pandas as pd
import numpy as np
import argparse
import os

def calculate_ihs(input_file, output_file=None, bloom_column="normalized_bloom_score", bge_column="bge_ICS"):
    """
    Calculate Intrinsic Hardness Score (IHS) by averaging normalized Bloom score and BGE ICS
    
    Args:
        input_file: Input Excel file path
        output_file: Output Excel file path (default: IHS.xlsx)
        bloom_column: Column name for normalized Bloom scores (default: normalized_bloom_score)
        bge_column: Column name for BGE ICS scores (default: bge_ICS)
    
    Returns:
        pd.DataFrame: DataFrame with IHS scores added
    """
    
    # 1️⃣ Read Excel file
    print(f"📊 Reading Excel file: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"✅ Successfully loaded {len(df)} rows")
    except Exception as e:
        raise RuntimeError(f"Cannot read input file {input_file}: {e}")
    
    # Validate required columns
    required_columns = [bloom_column, bge_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        available_columns = ", ".join(df.columns)
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {available_columns}")
    
    # 2️⃣ Calculate IHS (average of normalized Bloom score and BGE ICS)
    print("🧮 Calculating IHS scores...")
    df["IHS"] = (df[bloom_column] + df[bge_column]) / 2
    
    # 3️⃣ Save results to new file
    if output_file is None:
        output_file = "IHS.xlsx"
    
    print(f"💾 Saving results to: {output_file}")
    df.to_excel(output_file, index=False)
    
    # Print statistics
    print(f"\n📊 IHS Statistics:")
    print(f"   - Mean IHS: {df['IHS'].mean():.4f}")
    print(f"   - Std IHS: {df['IHS'].std():.4f}")
    print(f"   - Min IHS: {df['IHS'].min():.4f}")
    print(f"   - Max IHS: {df['IHS'].max():.4f}")
    
    print(f"✅ Processing completed, results saved to: {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="IHS (Intrinsic Hardness Score) Calculator")
    parser.add_argument("input_file", help="Input Excel file path")
    parser.add_argument("-o", "--output", help="Output Excel file path (default: IHS.xlsx)")
    parser.add_argument("--bloom-column", default="normalized_bloom_score", 
                       help="Column name for normalized Bloom scores (default: normalized_bloom_score)")
    parser.add_argument("--bge-column", default="bge_ICS", 
                       help="Column name for BGE ICS scores (default: bge_ICS)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file does not exist: {args.input_file}")
        return 1
    
    try:
        # Calculate IHS
        result_df = calculate_ihs(
            input_file=args.input_file,
            output_file=args.output,
            bloom_column=args.bloom_column,
            bge_column=args.bge_column
        )
        
        print("\n🎉 IHS calculation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
