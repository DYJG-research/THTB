#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
THTB Simple Framework - Simplified Data Selection Framework
A more direct and reliable THTB data selection implementation

Usage:
    python thtb_simple.py --input dataset.xlsx --output final_dataset.xlsx

Author: THTB Team
Version: 1.0.0
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np


class THBTSimpleFramework:
    """
    THTB Simplified Framework - Implements complete pipeline by calling existing scripts
    """
    
    def __init__(self, 
                 input_file: str,
                 output_file: str,
                 config: Optional[Dict] = None,
                 verbose: bool = True):
        """
        Initialize THTB Simplified Framework
        
        Args:
            input_file: Input dataset file path
            output_file: Final output file path  
            config: Configuration parameters dictionary
            verbose: Whether to display detailed information
        """
        self.input_file = Path(input_file).resolve()
        self.output_file = Path(output_file).resolve()
        self.verbose = verbose
        
        # Validate input file
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        
        # Create output directory
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create working directory
        self.work_dir = Path(tempfile.mkdtemp(prefix="thtb_work_"))
        self.project_root = Path(__file__).parent.resolve()
        
        # Default configuration
        self.default_config = {
            'quality_filter_percentage': 20.0,
            'ihs_filter_percentage': 50.0,
            'ehs_filter_percentage': 50.0,
            'gpu_id': 2,
            'silhouette_clusters': 3,
            'query_column': 'query',
            'response_column': 'response',
            'keep_intermediate_files': False,
            'reward_model_path': None,
            'bge_model_path': None
        }
        
        self.config = {**self.default_config, **(config or {})}
        
        self.log(f"THTB Simplified Framework initialized")
        self.log(f"Input file: {self.input_file}")
        self.log(f"Output file: {self.output_file}")
        self.log(f"Working directory: {self.work_dir}")
    
    def log(self, message: str, level: str = "INFO"):
        """Log information"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def run_script(self, script_path: str, args: List[str], cwd: Optional[str] = None) -> bool:
        """
        Run Python script
        
        Args:
            script_path: Script path
            args: Command line arguments list
            cwd: Working directory
            
        Returns:
            bool: Returns True if execution succeeds
        """
        try:
            cmd = [sys.executable, script_path] + args
            self.log(f"Executing command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                if self.verbose and result.stdout:
                    print(result.stdout)
                return True
            else:
                self.log(f"Script execution failed: {result.stderr}", "ERROR")
                if result.stdout:
                    self.log(f"Output: {result.stdout}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error executing script: {str(e)}", "ERROR")
            return False
    
    def run(self) -> bool:
        """
        Execute complete THTB data selection pipeline
        
        Returns:
            bool: Returns True if execution succeeds, False if it fails
        """
        try:
            self.log("Starting THTB data selection pipeline", "INFO")
            
            # Copy input file to working directory
            current_file = self.work_dir / "current_data.xlsx"
            shutil.copy2(self.input_file, current_file)
            
            # Stage 1: Quality Filtering
            self.log("=" * 60)
            self.log("Stage 1: Quality Filtering")
            self.log("=" * 60)
            if not self._stage1_quality_filtering(current_file):
                return False
            current_file = self.work_dir / "reward_top20.xlsx"
            
            # Stage 2: Intrinsic Hardness Score
            self.log("=" * 60) 
            self.log("Stage 2: Intrinsic Hardness Score (IHS)")
            self.log("=" * 60)
            if not self._stage2_intrinsic_hardness(current_file):
                return False
            current_file = self.work_dir / "ihs_top50.xlsx"
            
            # Stage 3: Extraneous Hardness Score
            self.log("=" * 60)
            self.log("Stage 3: Extraneous Hardness Score (EHS)")  
            self.log("=" * 60)
            if not self._stage3_extraneous_hardness(current_file):
                return False
            current_file = self.work_dir / "ehs_top50.xlsx"
            
            # Copy final result
            shutil.copy2(current_file, self.output_file)
            
            self.log("=" * 60)
            self.log("THTB data selection pipeline completed!", "SUCCESS")
            self.log(f"Final results saved to: {self.output_file}")
            
            # Process final output to JSONL format
            if not self._process_final_output_to_jsonl(self.output_file):
                self.log("Warning: Failed to convert final output to JSONL format", "WARNING")
            
            # Clean up temporary files
            if not self.config['keep_intermediate_files']:
                self._cleanup_temp_files()
                
            return True
            
        except Exception as e:
            self.log(f"Execution failed: {str(e)}", "ERROR")
            self.log(traceback.format_exc(), "ERROR")
            return False
    
    def _stage1_quality_filtering(self, input_file: Path) -> bool:
        """Stage 1: Quality Filtering"""
        
        # Step 1.1: Reward model scoring
        self.log("Step 1.1: Using reward model scoring...")
        reward_script = self.project_root / "reward_score" / "reward_score.py"
        reward_output = self.work_dir / "reward_scored.xlsx"
        
        # Build reward script arguments
        reward_args = ["-i", str(input_file), 
                       "-c", self.config['query_column'], self.config['response_column'],
                       "-o", str(reward_output),
                       "-g", str(self.config['gpu_id'])]
        
        # Add reward model path if specified
        if self.config.get('reward_model_path'):
            reward_args.extend(["-p", self.config['reward_model_path']])
        
        if not self.run_script(
            str(reward_script),
            reward_args,
            cwd=str(self.project_root / "reward_score")
        ):
            return False
        
        # Step 1.2: Quality-based filtering (top 20%)
        self.log("Step 1.2: Quality-based filtering (top 20%)...")
        selection_script = self.project_root / "dataSelection.py"
        
        return self.run_script(
            str(selection_script),
            [str(reward_output),
             "reward_model_score_normalized",
             "-p", str(self.config['quality_filter_percentage']),
             "-f", "reward_top20.xlsx"],
            cwd=str(self.work_dir)
        )
    
    def _stage2_intrinsic_hardness(self, input_file: Path) -> bool:
        """Stage 2: Intrinsic Hardness Score"""
        
        # Step 2.1: Bloom taxonomy scoring
        self.log("Step 2.1: Bloom taxonomy scoring...")
        bloom_script = self.project_root / "bloom_score" / "bloom.py"
        bloom_output = self.work_dir / "bloom_scored.xlsx"
        
        if not self.run_script(
            str(bloom_script),
            ["-i", str(input_file), "-o", str(bloom_output)],
            cwd=str(self.project_root / "bloom_score")
        ):
            return False
        
        # Step 2.2: Subject analysis
        self.log("Step 2.2: Subject analysis...")
        analyze_script = self.project_root / "IC" / "analyze_excel.py"
        analyzed_output = self.work_dir / "analyzed.xlsx"
        
        if not self.run_script(
            str(analyze_script),
            ["-i", str(bloom_output),
             "-o", str(analyzed_output)],
            cwd=str(self.project_root / "IC")
        ):
            return False
        
        # Step 2.3: Generate subject descriptions
        self.log("Step 2.3: Generating subject descriptions...")
        desc_script = self.project_root / "IC" / "1_generate_subject_descriptions.py"
        desc_output = self.work_dir / "subject_descriptions.xlsx"
        
        if not self.run_script(
            str(desc_script),
            ["-i", "subject_distribution.xlsx",
             "-o", str(desc_output)],
            cwd=str(self.project_root / "IC")
        ):
            return False
        
        # Step 2.4: Compute BGE distances
        self.log("Step 2.4: Computing BGE semantic distances...")
        bge_script = self.project_root / "IC" / "2_compute_bge_distances.py"
        
        # Build BGE script arguments
        bge_args = ["--gpu", str(self.config['gpu_id']),
                    "--input", str(desc_output),
                    "--output-dir", str(self.work_dir / "bge_outputs")]
        
        # Add BGE model path if specified
        if self.config.get('bge_model_path'):
            bge_args.extend(["--model-path", self.config['bge_model_path']])
        
        if not self.run_script(
            str(bge_script),
            bge_args,
            cwd=str(self.project_root / "IC")
        ):
            return False
        
        # Step 2.5: Recalculate subject distances
        self.log("Step 2.5: Recalculating subject distances...")
        recalc_script = self.project_root / "IC" / "3_recalculate_subject_distances.py"
        ics_output = self.work_dir / "new_ics.xlsx"
        
        if not self.run_script(
            str(recalc_script),
            ["-i", str(analyzed_output),
             "-o", str(ics_output),
             "-m", str(self.work_dir / "bge_outputs" / "subject_distance_matrix.pkl")],
            cwd=str(self.project_root / "IC")
        ):
            return False
        
        # Step 2.6: Calculate IHS
        self.log("Step 2.6: Calculating IHS scores...")
        if not self._calculate_ihs(ics_output):
            return False
        
        # Step 2.7: IHS-based filtering (top 50%)
        self.log("Step 2.7: IHS-based filtering (top 50%)...")
        ihs_file = self.work_dir / "IHS.xlsx"
        selection_script = self.project_root / "dataSelection.py"
        
        return self.run_script(
            str(selection_script),
            [str(ihs_file),
             "IHS",
             "-p", str(self.config['ihs_filter_percentage']),
             "-f", "ihs_top50.xlsx"],
            cwd=str(self.work_dir)
        )
    
    def _stage3_extraneous_hardness(self, input_file: Path) -> bool:
        """Stage 3: Extraneous Hardness Score"""
        
        # Step 3.1: Calculate IREI
        self.log("Step 3.1: Calculating IREI index...")
        irei_script = self.project_root / "IREI" / "IREI.py"
        irei_output = self.work_dir / "IREI.xlsx"
        
        if not self.run_script(
            str(irei_script),
            ["-i", str(input_file), "-o", str(irei_output)],
            cwd=str(self.project_root / "IREI")
        ):
            return False
        
        # Step 3.2: Calculate silhouette coefficient
        self.log("Step 3.2: Calculating silhouette coefficient...")
        silhouette_script = self.project_root / "silhouette" / "silhouette_Coefficient.py"
        silhouette_output = self.work_dir / "lk.xlsx"
        
        if not self.run_script(
            str(silhouette_script),
            [str(irei_output),
             "--column", self.config['query_column'],
             "--clusters", str(self.config['silhouette_clusters']),
             "--output", str(silhouette_output)],
            cwd=str(self.project_root / "silhouette")
        ):
            return False
        
        # Step 3.3: Calculate EHS and final filtering
        self.log("Step 3.3: Calculating EHS scores and final filtering...")
        if not self._calculate_ehs(silhouette_output):
            return False
        
        # EHS-based final filtering
        ehs_file = self.work_dir / "EHS.xlsx"
        selection_script = self.project_root / "dataSelection.py"
        
        return self.run_script(
            str(selection_script),
            [str(ehs_file), "EHS",
             "-p", str(self.config['ehs_filter_percentage']),
             "-f", "ehs_top50.xlsx"],
            cwd=str(self.work_dir)
        )
    
    def _calculate_ihs(self, ics_file: Path) -> bool:
        """Calculate IHS scores"""
        try:
            # Read data
            df = pd.read_excel(ics_file)
            
            # Calculate IHS = (normalized_bloom_score + bge_ICS) / 2
            required_cols = ['normalized_bloom_score', 'bge_ICS']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.log(f"Missing required columns: {missing_cols}", "ERROR")
                return False
            
            df['IHS'] = (df['normalized_bloom_score'] + df['bge_ICS']) / 2
            
            # Save results
            ihs_file = self.work_dir / "IHS.xlsx"
            df.to_excel(ihs_file, index=False)
            
            self.log(f"IHS calculation completed, {len(df)} data points")
            return True
            
        except Exception as e:
            self.log(f"IHS calculation failed: {str(e)}", "ERROR")
            return False
    
    def _calculate_ehs(self, silhouette_file: Path) -> bool:
        """Calculate EHS scores"""
        try:
            # Read data
            df = pd.read_excel(silhouette_file)
            
            # Calculate EHS = (IREI + normalized_silhouette_coefficient) / 2
            required_cols = ['IREI', 'normalized_silhouette_coefficient']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.log(f"Missing required columns: {missing_cols}", "ERROR")
                return False
            
            df['EHS'] = (df['IREI'] + df['normalized_silhouette_coefficient']) / 2
            
            # Save results
            ehs_file = self.work_dir / "EHS.xlsx"
            df.to_excel(ehs_file, index=False)
            
            self.log(f"EHS calculation completed, {len(df)} data points")
            return True
            
        except Exception as e:
            self.log(f"EHS calculation failed: {str(e)}", "ERROR")
            return False
    
    def _process_final_output_to_jsonl(self, output_file: Path) -> bool:
        """Process final Excel output to JSONL format with only query and response columns"""
        try:
            self.log("Processing final output to JSONL format...")
            
            # Read the final Excel file
            df = pd.read_excel(output_file)
            self.log(f"Loaded final dataset with {len(df)} rows")
            
            # Check if required columns exist
            required_cols = ['query', 'response']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.log(f"Missing required columns: {missing_cols}", "ERROR")
                return False
            
            # Keep only query and response columns
            df_filtered = df[['query', 'response']].copy()
            
            # Save filtered Excel file
            filtered_excel = output_file.parent / f"{output_file.stem}_filtered.xlsx"
            df_filtered.to_excel(filtered_excel, index=False)
            self.log(f"Filtered Excel saved to: {filtered_excel}")
            
            # Convert to JSONL format
            jsonl_file = output_file.parent / f"{output_file.stem}.jsonl"
            
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for _, row in df_filtered.iterrows():
                    query = str(row["query"]) if pd.notna(row["query"]) else ""
                    response = str(row["response"]) if pd.notna(row["response"]) else ""
                    
                    # Create JSON object
                    data = {
                        "query": query,
                        "response": response
                    }
                    
                    # Write to JSONL file
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            self.log(f"JSONL file saved to: {jsonl_file}")
            self.log(f"Successfully processed {len(df_filtered)} records to JSONL format")
            
            return True
            
        except Exception as e:
            self.log(f"Failed to process final output to JSONL: {str(e)}", "ERROR")
            return False

    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if self.work_dir.exists():
                shutil.rmtree(self.work_dir)
                self.log("Temporary files cleaned up")
        except Exception as e:
            self.log(f"Warning when cleaning up temporary files: {str(e)}", "WARNING")


def main():
    """Main function - Command line interface"""
    parser = argparse.ArgumentParser(
        description="THTB Simplified Data Selection Framework - One-click execution of complete three-stage data filtering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic usage
  python thtb_simple.py --input dataset.xlsx --output final_dataset.xlsx
  
  # Custom parameters
  python thtb_simple.py --input dataset.xlsx --output final_dataset.xlsx \\
    --quality-filter 20 --ihs-filter 50 --ehs-filter 50 --gpu 0
  
  # With custom model paths
  python thtb_simple.py --input dataset.xlsx --output final_dataset.xlsx \\
    --reward-model-path /path/to/reward/model --bge-model-path /path/to/bge/model
        """
    )
    
    # Required parameters
    parser.add_argument('--input', '-i', required=True,
                       help='Input dataset file path (Excel format)')
    parser.add_argument('--output', '-o', required=True,
                       help='Final output file path')
    
    # Optional parameters
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Display detailed progress information')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep intermediate temporary files')
    
    # Filtering parameters
    parser.add_argument('--quality-filter', type=float, default=20.0,
                       help='Quality filtering percentage (default: 20.0)')
    parser.add_argument('--ihs-filter', type=float, default=50.0,
                       help='IHS filtering percentage (default: 50.0)')
    parser.add_argument('--ehs-filter', type=float, default=50.0,
                       help='EHS filtering percentage (default: 50.0)')
    
    # Model parameters
    parser.add_argument('--gpu', type=int, default=2,
                       help='GPU device ID (default: 2)')
    parser.add_argument('--reward-model-path', type=str,
                       help='Path to reward model directory (optional)')
    parser.add_argument('--bge-model-path', type=str,
                       help='Path to BGE model directory (optional)')
    
    # Algorithm parameters
    parser.add_argument('--silhouette-clusters', type=int, default=3,
                       help='Number of clusters for silhouette analysis (default: 3)')
    
    # Data column parameters
    parser.add_argument('--query-column', default='query',
                       help='Query column name (default: query)')
    parser.add_argument('--response-column', default='response',
                       help='Response column name (default: response)')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'quality_filter_percentage': args.quality_filter,
        'ihs_filter_percentage': args.ihs_filter,
        'ehs_filter_percentage': args.ehs_filter,
        'gpu_id': args.gpu,
        'silhouette_clusters': args.silhouette_clusters,
        'query_column': args.query_column,
        'response_column': args.response_column,
        'keep_intermediate_files': args.keep_temp,
        'reward_model_path': args.reward_model_path,
        'bge_model_path': args.bge_model_path
    }
    
    try:
        # Create framework instance
        framework = THBTSimpleFramework(
            input_file=args.input,
            output_file=args.output,
            config=config,
            verbose=args.verbose
        )
        
        # Execute framework
        success = framework.run()
        
        if success:
            print("\n" + "="*60)
            print("🎉 THTB data selection pipeline executed successfully!")
            print(f"📁 Final results: {args.output}")
            print("="*60)
            sys.exit(0)
        else:
            print("\n" + "="*60)
            print("❌ THTB data selection pipeline execution failed!")
            print("Please check the error information above for troubleshooting")
            print("="*60)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Framework initialization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
