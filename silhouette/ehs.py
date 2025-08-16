import pandas as pd
import numpy as np
import argparse
import os

def calculate_ehs(input_file, output_file=None, irei_column="IREI", silhouette_column="normalized_silhouette_coefficient"):
    """
    Calculate Extraneous Hardness Score (EHS) by averaging IREI and normalized silhouette coefficient
    
    Args:
        input_file: Input Excel file path
        output_file: Output Excel file path (default: EHS.xlsx)
        irei_column: Column name for IREI scores (default: IREI)
        silhouette_column: Column name for normalized silhouette coefficient (default: normalized_silhouette_coefficient)
    
    Returns:
        pd.DataFrame: DataFrame with EHS scores added
    """
    
    # 1️⃣ Read Excel file
    print(f"📊 Reading Excel file: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"✅ Successfully loaded {len(df)} rows")
    except Exception as e:
        raise RuntimeError(f"Cannot read input file {input_file}: {e}")
    
    # Validate required columns
    required_columns = [irei_column, silhouette_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        available_columns = ", ".join(df.columns)
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {available_columns}")
    
    # 2️⃣ Calculate EHS (average of IREI and normalized silhouette coefficient)
    print("🧮 Calculating EHS scores...")
    df["EHS"] = (df[irei_column] + df[silhouette_column]) / 2
    
    # 3️⃣ Save results to new file
    if output_file is None:
        output_file = "EHS.xlsx"
    
    print(f"💾 Saving results to: {output_file}")
    df.to_excel(output_file, index=False)
    
    # Print statistics
    print(f"\n📊 EHS Statistics:")
    print(f"   - Mean EHS: {df['EHS'].mean():.4f}")
    print(f"   - Std EHS: {df['EHS'].std():.4f}")
    print(f"   - Min EHS: {df['EHS'].min():.4f}")
    print(f"   - Max EHS: {df['EHS'].max():.4f}")
    
    print(f"✅ Processing completed, results saved to: {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="EHS (Extraneous Hardness Score) Calculator")
    parser.add_argument("input_file", help="Input Excel file path")
    parser.add_argument("-o", "--output", help="Output Excel file path (default: EHS.xlsx)")
    parser.add_argument("--irei-column", default="IREI", 
                       help="Column name for IREI scores (default: IREI)")
    parser.add_argument("--silhouette-column", default="normalized_silhouette_coefficient", 
                       help="Column name for normalized silhouette coefficient (default: normalized_silhouette_coefficient)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file does not exist: {args.input_file}")
        return 1
    
    try:
        # Calculate EHS
        result_df = calculate_ehs(
            input_file=args.input_file,
            output_file=args.output,
            irei_column=args.irei_column,
            silhouette_column=args.silhouette_column
        )
        
        print("\n🎉 EHS calculation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
