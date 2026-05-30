# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This reward function is adapted for QAMPARI-style multi-answer QA.

from __future__ import annotations
import re
from typing import Any

from run_test.utils.metrics import (
    normalize_answer,
    to_normalized_set,
    get_answers,
    _parse_target_list,
    compute_tool_count,
    compute_repetition_count,
)


def check_format_error(solution_str: str) -> str | None:
    """Return a short reason string if *solution_str* has a format error, else ``None``.

    Checks: nested <think>, extra </think>, unclosed <think>,
    multiple think blocks, missing <answer>.
    Only checks content after the first "Question:" marker to avoid
    counting tags from the prompt template itself.
    """
    marker_match = re.search(r"Question:", solution_str, re.IGNORECASE)
    check_text = solution_str[marker_match.end() :] if marker_match else solution_str

    depth = 0
    for m in re.finditer(r"</?think>", check_text, re.IGNORECASE):
        if m.group().lower() == "<think>":
            depth += 1
            if depth > 1:
                return "nested_think"
        else:
            depth -= 1
            if depth < 0:
                return "extra_close_think"
    if depth > 0:
        return "unclosed_think"
    if not re.search(r"<answer>", check_text, re.IGNORECASE):
        return "no_answer_tag"
    # <answer> present but </answer> missing → truncated mid-answer
    if not re.search(r"</answer>", check_text, re.IGNORECASE):
        return "truncated_answer"
    return None


def compute_score_qampari(
    solution_str: str,
    ground_truth: Any,
) -> dict[str, Any]:
    """Reward function for QAMPARI-style multi-answer QA.

    Returns a dict so trainers can log richer metrics.
    """
    target = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
    target_list = _parse_target_list(target)

    gt_norm_set = to_normalized_set(target_list)
    num_gt = len(gt_norm_set)

    call_count = compute_tool_count(solution_str)
    predicted_answers = get_answers(solution_str)
    repetition_count = compute_repetition_count(predicted_answers)

    # --- Format-error early return (real structural errors only) ---
    fmt_error = check_format_error(solution_str)
    if fmt_error is not None:
        return {
            "score": -1.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "exact_match": 0.0,
            "call_count": call_count,
            "repetition_count": repetition_count,
            "format_error": 1.0,
            # "format_error_reason": fmt_error,
            "num_predicted": 0,
            "num_target": int(num_gt),
            "num_matches_in_traj": 0,
        }

    # empty_answer (<answer>[""]</answer> / <answer>[]</answer>) is NOT a format
    # error — the format is correct.  Normal computation yields f1=0, score=0.
    pred_norm_set = to_normalized_set(predicted_answers)
    num_pred = len(pred_norm_set)

    # Compute stats using sets
    matched_gt = pred_norm_set.intersection(gt_norm_set)
    matched_count = len(matched_gt)

    if num_pred > 0 and num_gt > 0:
        precision = matched_count / num_pred
        recall = matched_count / num_gt
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        precision, recall, f1_score = 0.0, 0.0, 0.0

    recall_at_5 = min(matched_count / min(5.0, num_gt), 1.0) if num_gt > 0 else 0.0
    recall_at_10 = min(matched_count / min(10.0, num_gt), 1.0) if num_gt > 0 else 0.0

    is_perfect = 1.0 if f1_score == 1.0 else 0.0

    # Count how many ground truth answers appear in the raw solution text
    normalized_solution = normalize_answer(solution_str)
    matches_in_traj = [gt for gt in gt_norm_set if gt in normalized_solution]
    num_matches_in_traj = len(matches_in_traj)

    total_score = f1_score 

    return {
        "score": float(total_score),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "recall@5": float(recall_at_5),
        "recall@10": float(recall_at_10),
        "exact_match": float(is_perfect),
        "call_count": int(call_count),
        "repetition_count": int(repetition_count),
        "format_error": 0.0,
        # "format_error_reason": "",
        "num_predicted": int(num_pred),
        "num_target": int(num_gt),
        "num_matches_in_traj": int(num_matches_in_traj),
    }
