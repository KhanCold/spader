
import logging
from datetime import datetime
import collections
import math
import fcntl

logger = logging.getLogger(__name__)

def generate_summary(summary_file, dataset_name, metrics_list, exp_name, model_name, total_count=0):
    """
    Updates the results.md table with the current run's key metrics.
    Adds a new row for the current run and highlights SOTA (max) values for each metric.
    """
    if not metrics_list:
        logger.warning("No metrics to summarize.")
        return
    
    # 1. Compute stats (Means only) for current run
    metric_values = collections.defaultdict(list)
    for m in metrics_list:
        for k, v in m.items():
            if isinstance(v, (int, float)):
                metric_values[k].append(v)
    
    current_stats = {}
    for k, values in metric_values.items():
        if values:
            current_stats[k] = sum(values) / len(values)

    # 2. Define Key Metrics Order — keep only recall / f1 / precision
    priority_metrics = ["recall", "f1_score", "precision"]
    
    # 3. Read existing file to preserve history (under cross-process file lock)
    import os
    existing_headers = []
    existing_rows = []
    
    # Standard columns
    fixed_columns = ["Experiment", "Model", "Date"]

    # Use fcntl file lock to prevent race conditions when multiple
    # eval processes write to the same results.md concurrently.
    os.makedirs(os.path.dirname(summary_file) or ".", exist_ok=True)
    lock_path = summary_file + ".lock"
    lock_fd = open(lock_path, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        logger.info(f"Acquired file lock: {lock_path}")

        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    lines = f.readlines()
                
                # Simple parser for Markdown table
                # Find header line
                header_line_idx = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith("|") and "Experiment" in line:
                        header_line_idx = i
                        break
                
                if header_line_idx != -1:
                    # Parse headers
                    headers = [h.strip() for h in lines[header_line_idx].strip().strip('|').split('|')]
                    existing_headers = headers
                    
                    # Parse rows (skip separator)
                    for line in lines[header_line_idx+2:]:
                        if not line.strip().startswith("|"): continue
                        row_vals = [v.strip() for v in line.strip().strip('|').split('|')]
                        
                        if len(row_vals) == len(headers):
                            clean_row = {}
                            for h, v in zip(headers, row_vals):
                                # Remove existing bold markers to get raw value
                                clean_v = v.replace("**", "")
                                clean_row[h] = clean_v
                            existing_rows.append(clean_row)
            except Exception as e:
                logger.error(f"Failed to read existing summary file: {e}")

        # 4. Headers — always rebuild so legacy metric columns are pruned
        current_keys = [k for k in priority_metrics if k in current_stats]
        existing_headers = fixed_columns + current_keys

        # 5. Add current row
        new_row = {
            "Experiment": exp_name if exp_name else "-",
            "Model": model_name,
            "Date": datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        for k in existing_headers[3:]: # Metric columns
            if k in current_stats:
                val = current_stats[k]
                # Format
                if k == "call_count" or k == "repetition_count":
                     new_row[k] = f"{val:.2f}"
                else:
                     new_row[k] = f"{val:.4f}"
            else:
                new_row[k] = "-"
        
        existing_rows.append(new_row)
        
        # 6. Determine SOTAs (Best values)
        # Define which metrics are "lower is better"
        lower_is_better = {"repetition_count", "format_error", "call_count", "entity_per_search", "character_length", "num_turns", "assistant_turns"}
        
        metric_cols = existing_headers[3:]
        sota_values = {}
        
        for col in metric_cols:
            vals = []
            for r in existing_rows:
                v_str = r.get(col, "-")
                if v_str == "-":
                    continue
                try:
                    # Remove ** for parsing if they exist
                    raw_v = v_str.replace("**", "")
                    vals.append(float(raw_v))
                except:
                    pass
            
            if vals:
                 if col in lower_is_better:
                     sota_values[col] = min(vals)
                 else:
                     sota_values[col] = max(vals)
        
        # 7. Generate Table Content
        summary_md = f"# Evaluation Results for {dataset_name}\n\n"
        
        # Header
        header_str = "| " + " | ".join(existing_headers) + " |"
        summary_md += header_str + "\n"
        
        # Separator
        sep_str = "| " + " | ".join([":---:" if i >= 3 else ":---" for i in range(len(existing_headers))]) + " |"
        summary_md += sep_str + "\n"
        
        # Rows
        for r in existing_rows:
            row_str = "|"
            for col in existing_headers:
                val = r.get(col, "-")
                # Check SOTA
                if col in sota_values and val != "-":
                    try:
                        fval = float(val.replace("**", ""))
                        # Use math.isclose for float comparison or simple equals
                        if math.isclose(fval, sota_values[col], rel_tol=1e-9):
                            if not val.startswith("**"):
                                 val = f"**{val}**"
                    except:
                        pass
                row_str += f" {val} |"
            summary_md += row_str + "\n"
        
        # Write to file
        try:
            with open(summary_file, 'w') as f:
                f.write(summary_md)
            logger.info(f"Summary written to {summary_file}")
        except Exception as e:
            logger.error(f"Failed to write summary to {summary_file}: {e}")

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
