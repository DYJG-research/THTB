import os
import sys
import re
import traceback
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# Constants and utility functions
DIMENSION_NAME = "reward_model_score"
DEFAULT_MODEL_REPO = "Shanghai_AI_Laboratory/internlm2-7b-reward"
REQUIRED_MODEL_KEYWORD = "internlm2-7b-reward"


def _normalize_score(raw: float) -> float:
    """Map raw scores from [-5, 5] range to [0, 1] and clip."""
    return max(0.0, min(1.0, (float(raw) + 5) / 10))


def _install_if_missing(package: str, spec: Optional[str] = None) -> None:
    try:
        __import__(package)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", spec or package])


def load_local_reward_model(use_local_model: bool, model_path: str, gpu_id: int):
    """Load local reward model (only allows InternLM2-7B-Reward). Returns (local_model, local_tokenizer) on success, otherwise raises exception."""
    if not use_local_model:
        raise RuntimeError("Must use InternLM2-7B-Reward model for scoring.")

    try:
        import torch

        # Environment
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # GPU settings
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if gpu_id >= num_gpus:
                gpu_id = 0
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Dependencies
        try:
            import modelscope  
        except ImportError:
            _install_if_missing("modelscope")
            import modelscope  

        try:
            import accelerate 
        except ImportError:
            _install_if_missing("accelerate", "accelerate>=0.26.0")
            import accelerate 

        # Try to ensure transformers exists
        try:
            import transformers 
        except ImportError:
            print("transformers not detected, trying to install transformers>=4.40...")
            _install_if_missing("transformers", "transformers>=4.40.0")
            import transformers 

        from modelscope import AutoModel

        # Model identifier: prioritize local path
        model_identifier = model_path if os.path.exists(model_path) else DEFAULT_MODEL_REPO
        print(f"Model identifier: {model_identifier}")

        # Force validate model identifier
        ident_norm = str(model_identifier).replace("\\", "/").lower()
        if REQUIRED_MODEL_KEYWORD not in ident_norm:
            raise RuntimeError("Not using InternLM2-7B-Reward model, program terminated.")

        # Set tokenizer directory: only allow using local directory (prioritize passed directory, otherwise use fixed local path)
        tokenizer_dir = model_path if os.path.isdir(model_path) else None
        print(f"tokenizer_dir: {tokenizer_dir if tokenizer_dir else 'Local tokenizer directory not found'}")

        # Insert tokenizer directory and model directory at the beginning of sys.path
        if tokenizer_dir and (tokenizer_dir not in sys.path):
            sys.path.insert(0, tokenizer_dir)
        model_dir = os.path.dirname(model_identifier) if not os.path.isdir(model_identifier) else model_identifier
        if model_dir and (model_dir not in sys.path):
            sys.path.insert(0, model_dir)

        # Load tokenizer: only load from local directory (don't use modelscope, don't use Llama fallback)
        local_tokenizer = None
        if tokenizer_dir:
            # Fallback plan A: try to directly import local tokenizer source code and instantiate
            try:
                import importlib.util
                tok_py_paths = [
                    os.path.join(tokenizer_dir, "tokenization_internlm2.py"),
                    os.path.join(tokenizer_dir, "tokenization_internlm2_fast.py"),
                ]
                loaded = False
                for py_path in tok_py_paths:
                    if os.path.isfile(py_path):
                        try:
                            spec = importlib.util.spec_from_file_location("tokenization_internlm2", py_path)
                            if spec and spec.loader:
                                tok_module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(tok_module)
                                if hasattr(tok_module, "InternLM2Tokenizer"):
                                    local_tokenizer = tok_module.InternLM2Tokenizer.from_pretrained(tokenizer_dir)
                                    print(f"Successfully loaded tokenizer from local source: {py_path}")
                                    loaded = True
                                    break
                        except Exception as e:
                            print(f"Failed to load tokenizer from {py_path}: {e}")
                            continue
                if not loaded:
                    raise RuntimeError("Failed to load local tokenizer from source files")
            except Exception as e:
                print(f"Failed to load local tokenizer: {e}")
                raise RuntimeError("Must use local InternLM2-7B-Reward tokenizer")

        # Load model: prioritize local path
        try:
            # Determine target device
            target_device = f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu"
            print(f"Target device: {target_device}")
            
            if os.path.isdir(model_identifier):
                print(f"Loading model from local directory: {model_identifier}")
                local_model = AutoModel.from_pretrained(
                    model_identifier,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
            else:
                print(f"Loading model from ModelScope: {model_identifier}")
                local_model = AutoModel.from_pretrained(
                    model_identifier,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
            
            # Explicitly move model to target device
            local_model = local_model.to(target_device)
            print(f"✅ Model moved to device: {target_device}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise RuntimeError("Failed to load InternLM2-7B-Reward model")

        print("✅ Model and tokenizer loaded successfully")
        return local_model, local_tokenizer

    except Exception as e:
        print(f"❌ Error loading local reward model: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Failed to load local reward model: {e}")


def _local_model_inference(query_text: str, response_text: str, local_model, local_tokenizer, gpu_id: int):
    """
    Use local InternLM2-7B-Reward model for inference.
    Returns: (score_text, normalized_score)
    Any exception will be raised to terminate the program.
    """
    try:
        import torch

        # Construct conversation
        conversation = [
            {"role": "user", "content": query_text},
            {"role": "assistant", "content": response_text}
        ]

        # Apply chat template
        input_text = local_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        
        # Tokenize
        inputs = local_tokenizer(input_text, return_tensors="pt")
        
        # Move to appropriate device
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = local_model(**inputs)
            # Get reward score (usually the last logit)
            reward_score = outputs.logits[0, -1].item()
        
        # Normalize score
        normalized_score = _normalize_score(reward_score)
        
        return str(reward_score), normalized_score

    except Exception as e:
        print(f"❌ Local model inference error: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Local model inference failed: {e}")


def evaluate_tuple(
    data_tuple: List[Any],
    columns: List[str],
    use_local_model: bool,
    local_model,
    local_tokenizer,
    gpu_id: int,
) -> float:
    """Evaluate (Query, Response) tuple with reward model scoring. Raises exception if specified model not used or error occurs."""
    if not use_local_model:
        raise RuntimeError("Must use local InternLM2-7B-Reward model for evaluation.")

    if len(data_tuple) < 2:
        raise ValueError("Data tuple must contain at least 2 elements (Query, Response)")

    query_text = str(data_tuple[0]) if data_tuple[0] is not None else ""
    response_text = str(data_tuple[1]) if data_tuple[1] is not None else ""

    if not query_text.strip() or not response_text.strip():
        raise ValueError("Query and Response cannot be empty")

    try:
        score_text, normalized_score = _local_model_inference(
            query_text, response_text, local_model, local_tokenizer, gpu_id
        )
        return normalized_score
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


def get_tuple_reward_details(
    data_tuple: List[Any],
    columns: List[str],
    use_local_model: bool,
    local_model,
    local_tokenizer,
    gpu_id: int,
) -> Dict[str, Any]:
    try:
        score_text, normalized_score = _local_model_inference(
            str(data_tuple[0]) if data_tuple[0] is not None else "",
            str(data_tuple[1]) if data_tuple[1] is not None else "",
            local_model,
            local_tokenizer,
            gpu_id,
        )
        return {
            "reward_model_score": score_text,
            "reward_model_normalized_score": normalized_score,
        }
    except Exception as e:
        print(f"❌ Failed to get reward details: {e}")
        raise


def evaluate_tuples_from_excel(
    file_path: str,
    columns: List[str],
    output_file: Optional[str] = None,
    sheet_name: Any = 0,
    use_local_model: bool = True,
    model_path: str = DEFAULT_MODEL_REPO,
    gpu_id: int = 0,
) -> pd.DataFrame:
    """Read data from Excel file and evaluate reward model scores by (Query, Response) tuples. Any exception will be raised to terminate."""
    
    print(f"📊 Reading Excel file: {file_path}")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"✅ Successfully loaded {len(df)} rows")
    except Exception as e:
        raise RuntimeError(f"Cannot read input file {file_path}: {e}")

    # Validate columns
    missing = [c for c in columns if c not in df.columns]
    if missing:
        columns = [c for c in columns if c in df.columns]
        if not columns:
            available_columns = ", ".join(df.columns)
            raise ValueError(f"Specified columns do not exist in input file. Available columns: {available_columns}")

    print(f"🔄 Loading reward model...")
    local_model, local_tokenizer = load_local_reward_model(use_local_model, model_path, gpu_id)

    results: Dict[str, Any] = {c: df[c].copy() for c in columns}
    name = DIMENSION_NAME
    results[f"{name}"] = pd.Series([None] * len(df), index=df.index)
    results[f"{name}_normalized"] = pd.Series([None] * len(df), index=df.index)

    import time
    temp_output_file = None
    if output_file:
        file_name, file_ext = os.path.splitext(output_file)
        temp_output_file = f"{file_name}_temp{file_ext}"
    last_save = time.time()
    save_interval = 60

    print(f"🧮 Evaluating {name} for {len(df)} samples...")
    for idx in tqdm(df.index, desc=f"Evaluating {name}"):
        row = df.loc[idx]
        data_tuple = [row[c] if c in row.index else None for c in columns]

        details = get_tuple_reward_details(
            data_tuple=data_tuple,
            columns=columns,
            use_local_model=use_local_model,
            local_model=local_model,
            local_tokenizer=local_tokenizer,
            gpu_id=gpu_id,
        )
        reward_model_score = details.get("reward_model_score", None)
        results[f"{name}"][idx] = reward_model_score
        results[f"{name}_normalized"][idx] = details.get("reward_model_normalized_score", None)

        now = time.time()
        if temp_output_file and now - last_save >= save_interval:
            pd.DataFrame(results).to_excel(temp_output_file, index=False)
            last_save = now

    result_df = pd.DataFrame(results)

    if output_file:
        result_df.to_excel(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        if temp_output_file and os.path.exists(temp_output_file):
            try:
                os.remove(temp_output_file)
            except Exception:
                pass
    
    # Print summary
    print(f"\n📊 Evaluation Summary:")
    print(f"   - Total samples evaluated: {len(df)}")
    if f"{name}_normalized" in result_df.columns:
        scores = result_df[f"{name}_normalized"].dropna()
        if len(scores) > 0:
            print(f"   - Average normalized score: {scores.mean():.4f}")
            print(f"   - Score standard deviation: {scores.std():.4f}")
            print(f"   - Min score: {scores.min():.4f}")
            print(f"   - Max score: {scores.max():.4f}")
    
    return result_df


def main() -> None:
    import argparse

    # Use relative paths, relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_model_path = os.environ.get("REWARD_MODEL_PATH", os.path.join(project_root, "models", "internlm2-7b-reward"))

    parser = argparse.ArgumentParser(description="Reward Model Score Evaluation Tool (Functional Simplified Version)")
    parser.add_argument("--input", "-i", help="Input Excel file path")
    parser.add_argument("--output", "-o", help="Output Excel file path")

    parser.add_argument("--columns", "-c", nargs="+", help="List of column names to evaluate (e.g.: query response)")
    parser.add_argument("--model-path", "-p", default=default_model_path, help="Local reward model path")
    parser.add_argument("--gpu-id", "-g", type=int, default=0, help="Specify GPU ID to use")
    parser.add_argument("--no-local-model", action="store_true", help="Don't use local model (will trigger error)")

    args = parser.parse_args()

    if not args.input or not args.columns:
        print("Error: Must provide --input and --columns")
        print("Example: python reward_score.py -i data.xlsx -c query response -o result.xlsx")
        sys.exit(1)

    try:
        use_local_model = not args.no_local_model
        print(f"Available dimension: {DIMENSION_NAME}")

        result = evaluate_tuples_from_excel(
            file_path=args.input,
            columns=args.columns,
            output_file=args.output,
            sheet_name=0,
            use_local_model=use_local_model,
            model_path=args.model_path,
            gpu_id=args.gpu_id,
        )
        if result is not None and not result.empty:
            print("🎉 Evaluation completed successfully!")
        else:
            print("❌ Evaluation failed or no results")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Program terminated: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
