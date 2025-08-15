#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Selection Tool
Function: Sort and filter operations on Excel files
Supports custom file names, column names and filtering ranges
"""

import pandas as pd
import os
import argparse
import sys

def sort_and_filter_data(input_file, column_name, filter_percentage=5.0, ascending=False, output_suffix="_sorted", filtered_output=None):
    """
    Sort and filter Excel files

    Args:
        input_file: Input Excel file path
        column_name: Column name to sort by
        filter_percentage: Filter percentage (default 5%, i.e., top 5%)
        ascending: Whether to sort in ascending order (default False, i.e., descending)
        output_suffix: Output file suffix
        filtered_output: Custom filtered result output filename (required, can be filename or full path; automatically uses input file extension if no extension)

    Returns:
        tuple: (sorted file path, filtered file path)
    """

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    # Force requirement to provide filtered output filename
    if not filtered_output:
        raise ValueError("Must specify filtered result output filename (--filtered-name or interactive input).")

    print(f"Reading file: {input_file}")

    # Read Excel file
    try:
        df = pd.read_excel(input_file)
        print(f"Successfully read file, {len(df)} rows of data")
    except Exception as e:
        raise Exception(f"Failed to read Excel file: {e}")

    # Check if column exists
    if column_name not in df.columns:
        available_columns = ", ".join(df.columns)
        raise ValueError(f"Column '{column_name}' not found. Available columns: {available_columns}")

    print(f"Sorting by column: {column_name}")
    print(f"Sort order: {'Ascending' if ascending else 'Descending'}")

    # Sort data
    df_sorted = df.sort_values(by=column_name, ascending=ascending)

    # Generate sorted file path
    input_dir = os.path.dirname(input_file)
    input_name = os.path.splitext(os.path.basename(input_file))[0]
    input_ext = os.path.splitext(input_file)[1]
    sorted_file = os.path.join(input_dir, f"{input_name}{output_suffix}{input_ext}")

    # Save sorted file
    df_sorted.to_excel(sorted_file, index=False)
    print(f"Sorted file saved: {sorted_file}")

    # Calculate number of rows to filter
    total_rows = len(df_sorted)
    filter_count = max(1, int(total_rows * filter_percentage / 100))
    
    print(f"Total rows: {total_rows}")
    print(f"Filter percentage: {filter_percentage}%")
    print(f"Rows to select: {filter_count}")

    # Filter data (take top N rows)
    df_filtered = df_sorted.head(filter_count)

    # Handle filtered output file path
    if not os.path.splitext(filtered_output)[1]:  # No extension
        filtered_output += input_ext
    
    # If not absolute path, save in input file directory
    if not os.path.isabs(filtered_output):
        filtered_output = os.path.join(input_dir, filtered_output)

    # Save filtered file
    df_filtered.to_excel(filtered_output, index=False)
    print(f"Filtered file saved: {filtered_output}")
    print(f"Selected {len(df_filtered)} rows (top {filter_percentage}%)")

    return sorted_file, filtered_output

def interactive_mode():
    """Interactive mode for user input"""
    print("=== Data Selection Tool - Interactive Mode ===")
    
    # Get input file
    while True:
        input_file = input("Please enter input Excel file path: ").strip()
        if os.path.exists(input_file):
            break
        print(f"File does not exist: {input_file}, please re-enter")
    
    # Read file and display available columns
    try:
        df = pd.read_excel(input_file)
        print(f"\nFile read successfully, {len(df)} rows of data")
        print("Available columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Get column name
    while True:
        column_input = input("\nPlease enter column name (or number): ").strip()
        if column_input.isdigit():
            col_index = int(column_input) - 1
            if 0 <= col_index < len(df.columns):
                column_name = df.columns[col_index]
                break
            else:
                print("Invalid column number")
        elif column_input in df.columns:
            column_name = column_input
            break
        else:
            print("Column not found, please re-enter")
    
    # Get filter percentage
    while True:
        try:
            filter_percentage = float(input("Please enter filter percentage (e.g., 5 for 5%): ").strip())
            if 0 < filter_percentage <= 100:
                break
            else:
                print("Percentage must be between 0 and 100")
        except ValueError:
            print("Please enter a valid number")
    
    # Get sort order
    while True:
        order_input = input("Sort order (1: Descending, 2: Ascending): ").strip()
        if order_input in ['1', '2']:
            ascending = order_input == '2'
            break
        else:
            print("Please enter 1 or 2")
    
    # Get filtered output filename
    while True:
        filtered_output = input("Please enter filtered result filename: ").strip()
        if filtered_output:
            break
        else:
            print("Filename cannot be empty")
    
    # Execute sorting and filtering
    try:
        sorted_file, filtered_file = sort_and_filter_data(
            input_file, column_name, filter_percentage, ascending, 
            filtered_output=filtered_output
        )
        print(f"\n=== Operation Completed ===")
        print(f"Sorted file: {sorted_file}")
        print(f"Filtered file: {filtered_file}")
    except Exception as e:
        print(f"Operation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Data Selection Tool - Sort and filter Excel files")
    parser.add_argument("input_file", nargs='?', help="Input Excel file path")
    parser.add_argument("column_name", nargs='?', help="Column name to sort by")
    parser.add_argument("-p", "--percentage", type=float, default=5.0, 
                       help="Filter percentage (default: 5.0)")
    parser.add_argument("-a", "--ascending", action="store_true", 
                       help="Sort in ascending order (default: descending)")
    parser.add_argument("-s", "--suffix", default="_sorted", 
                       help="Output file suffix (default: _sorted)")
    parser.add_argument("-f", "--filtered-name", 
                       help="Filtered result output filename (required)")
    parser.add_argument("-i", "--interactive", action="store_true", 
                       help="Run in interactive mode")

    args = parser.parse_args()

    # Interactive mode
    if args.interactive or (not args.input_file and not args.column_name):
        interactive_mode()
        return

    # Command line mode
    if not args.input_file or not args.column_name:
        print("Error: input_file and column_name are required in command line mode")
        print("Use -i or --interactive for interactive mode")
        parser.print_help()
        sys.exit(1)

    if not args.filtered_name:
        print("Error: --filtered-name is required")
        parser.print_help()
        sys.exit(1)

    try:
        sorted_file, filtered_file = sort_and_filter_data(
            args.input_file, 
            args.column_name, 
            args.percentage, 
            args.ascending, 
            args.suffix,
            args.filtered_name
        )
        print(f"\n=== Operation Completed ===")
        print(f"Sorted file: {sorted_file}")
        print(f"Filtered file: {filtered_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
