import argparse
import os
import sys
import json
import concurrent.futures
import threading
import time
from tqdm import tqdm
from datetime import datetime
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_test.models.vllm_base import VLLMBase
from run_test.utils.tools import SearchTool
from run_test.inference import run_single_sample
from run_test.utils.analysis import generate_summary

EVAL_DATA_DIR = os.environ.get("EVAL_DATA_DIR", "data/evaluation")

# Configuration
DATASETS = {
    "qampari": os.path.join(EVAL_DATA_DIR, "qampari_test.jsonl"),
    "quest": os.path.join(EVAL_DATA_DIR, "quest_test.jsonl"),
    "mintaka": os.path.join(EVAL_DATA_DIR, "mintaka_test.jsonl"),
    "webqsp": os.path.join(EVAL_DATA_DIR, "webqsp_test.jsonl"),
}

MODELS = {
    "qwen3-8b-grpo": VLLMBase,
    "qwen3-8b-spader": VLLMBase,
    "llama3-8b-grpo": VLLMBase,
    "llama3-8b-spader": VLLMBase,
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.identity").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def evaluate_dataset(args, dataset_key, dataset_path):
    start_time = time.time()
    # Setup Paths
    dataset_output_dir = os.path.join(args.output_dir, dataset_key)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_prefix = f"{args.exp_name}_" if args.exp_name else ""
    base_filename = f"{exp_prefix}{args.model}_{dataset_key}_{timestamp}"
    
    result_file = os.path.join(dataset_output_dir, f"{base_filename}.jsonl")
    summary_file = os.path.join(dataset_output_dir, "results.md")
    
    # Load Data
    logger.info(f"Loading data for {dataset_key} from {dataset_path}")
    data = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to load data from {dataset_path}: {e}")
        return

    if args.limit:
        data = data[:args.limit]
    logger.info(f"Loaded {len(data)} samples")
    
    # Init Lock for file writing
    file_lock = threading.Lock()
    
    # Thread Work function
    def worker(idx, item):
        # Instantiate the correct model class from registry
        model_cls = MODELS[args.model]
        model_instance = model_cls(model_name=args.model, api_base=args.api_base, api_key="EMPTY")
        tool_instance = SearchTool(url=args.tool_url, top_k=args.top_k, initial_k=args.initial_k, max_chars=args.truncate_length)
        
        try:
            res, metrics = run_single_sample(idx, item, model_instance, tool_instance)
            save_result(res, result_file, file_lock)
            return metrics
        except Exception as e:
            logger.error(f"Failed sample {idx}: {e}")
            return {}

    metrics_collection = []
    
    logger.info(f"Starting evaluation for {dataset_key} with {args.concurrency} threads...")
    logger.info(f"Results will be saved to: {result_file}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(worker, i, item): i for i, item in enumerate(data)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data), desc=f"Progress ({dataset_key})"):
            idx = futures[future]
            try:
                m = future.result()
                metrics_collection.append(m)
            except Exception as e:
                logger.error(f"Exception in future {idx}: {e}")

    # Sort the results file by index
    logger.info(f"Sorting results in {result_file}...")
    try:
        sorted_results = []
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        sorted_results.append(json.loads(line))
            
            # Sort by index
            sorted_results.sort(key=lambda x: x.get('index', 0))
            
            # Rewrite sorted results
            with open(result_file, 'w', encoding='utf-8') as f:
                for item in sorted_results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Sorted results saved to {result_file}")
    except Exception as e:
        logger.error(f"Failed to sort results: {e}")

    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}h{int(minutes)}m{int(seconds)}s"

    logger.info(f"Evaluation for {dataset_key} complete. Duration: {time_str}")
    if args.analyze:
        generate_summary(summary_file, dataset_key, metrics_collection, args.exp_name, args.model, len(data))

def save_result(result, file_path, lock):
    with lock:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

def _resolve_datasets(dataset_arg):
    dataset_arg = dataset_arg.strip()
    if dataset_arg.lower() == "all":
        return DATASETS

    dataset_tokens = [token.strip() for token in dataset_arg.split(",") if token.strip()]
    if not dataset_tokens:
        raise ValueError("Dataset argument is empty after parsing.")

    datasets_to_run = {}
    for token in dataset_tokens:
        path = DATASETS.get(token)
        name = token
        if not path:
            if os.path.exists(token):
                path = token
                name = token.split("/")[-1].replace(".jsonl", "").replace(".json", "")
            else:
                raise ValueError(f"Dataset {token} not found in registry or paths.")
        datasets_to_run[name] = path

    return datasets_to_run


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name/path, comma-separated list, or 'all'. Named datasets are resolved under EVAL_DATA_DIR.",
    )
    parser.add_argument("--model", type=str, default="qwen3-8b-grpo", help="Model name (must be in MODELS registry)")
    parser.add_argument("--api_base", type=str, default="http://localhost:8001/v1", help="API Base URL")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of threads")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--tool_url", type=str, default="http://127.0.0.1:8000/retrieve", help="Tool URL")
    parser.add_argument("--top_k", type=int, default=5, help="Top K retrieval results")
    parser.add_argument("--initial_k", type=int, default=50, help="Initial retrieval K before reranking")
    parser.add_argument("--truncate_length", type=int, default=10000, help="Max length of tool output")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--analyze", action="store_true", help="Enable detailed analysis summary")
    parser.add_argument("--output_dir", type=str, default="run_test/results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Validation
    if args.model not in MODELS:
        available = list(MODELS.keys())
        raise ValueError(f"Model '{args.model}' not found in registry. Available models: {available}")
    
    datasets_to_run = _resolve_datasets(args.dataset)

    for key, path in datasets_to_run.items():
        logger.info(f"Processing dataset: {key}")
        evaluate_dataset(args, key, path)

if __name__ == "__main__":
    main()
