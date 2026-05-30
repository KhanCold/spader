from __future__ import annotations

import ast
import json
import re
import string
from typing import Any


_ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: Any) -> str:
    """Normalization that handles None and non-string inputs safely."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = text.replace("''", '"').replace("``", '"')
    text = text.replace("’", "'").replace("‘", "'").replace("”", '"').replace("“", '"')

    exclude = set(string.punctuation + "—" + "–" + "‐")
    text = "".join(ch for ch in text if ch not in exclude)

    text = _ARTICLES_REGEX.sub(" ", text)
    return " ".join(text.split())


def to_normalized_set(targets: list[Any] | None) -> set[str]:
    """Normalize elements, remove empty strings, and return a unique set."""
    if not targets:
        return set()
    return {normalize_answer(t) for t in targets} - {""}


def _parse_target_list(target: Any) -> list[str]:
    """Back-compat helper: coerce various ground-truth shapes to a flat list[str].

    Kept for RL training code that still imports it from this module.
    """
    if target is None:
        return []
    if isinstance(target, list):
        return [str(x) for x in target]
    if isinstance(target, tuple):
        return [str(x) for x in target]

    if isinstance(target, str):
        stripped = target.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(stripped)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
        return [target]

    return [str(target)]


def get_answers(solution_str: str) -> list[str] | None:
    """Extract predicted answers from the last <answer>...</answer> block."""
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if not matches:
        return None

    content = matches[-1].strip()
    if not content:
        return None

    if content.startswith("[") and content.endswith("]"):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(content)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if str(x).strip()]
            except Exception:
                pass

    return [content]


def compute_tool_count(solution_str: str) -> int:
    """Count tool queries inside <tool_call>...</tool_call> blocks."""
    tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", solution_str, re.DOTALL)
    call_count = 0
    for tool_call in tool_calls:
        try:
            parsed = json.loads(tool_call)
            if isinstance(parsed, list):
                call_count += len(parsed)
            elif isinstance(parsed, dict) and "arguments" in parsed:
                args = parsed.get("arguments", {})
                call_count += len(args.get("query_list", []))
            else:
                call_count += 1
        except Exception:
            call_count += 1
    return call_count


def compute_repetition_count(predicted_answers: list[str]) -> int:
    if not predicted_answers:
        return 0
    normalized = [str(x).strip().lower() for x in predicted_answers]
    return len(normalized) - len(set(normalized))


def compute_entity_per_search(correct_predicted_count: int, call_count: int) -> float:
    """Average tool calls needed to obtain one matched entity."""
    if call_count <= 0 or correct_predicted_count <= 0:
        return 0.0
    return call_count / correct_predicted_count


def _normalize_entities(ground_truth: Any) -> tuple[list[set[str]], list[str]]:
    """Reshape ground_truth (list[list[str]]) into (alias_sets, canonical_names).

    - Deduplicates by normalized alias set: two entities whose aliases normalize
      to the exact same set are merged into one. This preserves the old flat-list
      scoring semantics (where duplicates were silently dropped by ``set()``).
    - canonical_names[i]: first non-empty raw alias of the kept entity.
    - Entities whose aliases all normalize to "" are dropped.
    """
    if not isinstance(ground_truth, list):
        return [], []
    seen: dict[frozenset[str], str] = {}
    for aliases in ground_truth:
        if not isinstance(aliases, list):
            continue
        normed = {normalize_answer(a) for a in aliases}
        normed.discard("")
        if not normed:
            continue
        key = frozenset(normed)
        if key not in seen:
            first = next((str(a) for a in aliases if str(a).strip()), "")
            seen[key] = first
    alias_sets = [set(k) for k in seen.keys()]
    canonicals = list(seen.values())
    return alias_sets, canonicals


def _normalize_predictions(predicted: list[str] | None) -> set[str]:
    if not predicted:
        return set()
    return {n for n in (normalize_answer(p) for p in predicted) if n}


def _format_error_metrics(
    *,
    call_count: int,
    num_gt: int,
    canonicals: list[str],
    matches_in_traj: list[str],
    character_length: int,
    num_turns: int,
    assistant_turns: int,
) -> dict[str, Any]:
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "recall@20": 0.0,
        "recall@50": 0.0,
        "exact_match": 0.0,
        "call_count": call_count,
        "entity_per_search": 0.0,
        "repetition_count": 0,
        "format_error": 1.0,
        "matches_in_traj": matches_in_traj,
        "num_matches_in_traj": len(matches_in_traj),
        "num_target": int(num_gt),
        "num_predicted": 0,
        "character_length": character_length,
        "num_turns": num_turns,
        "assistant_turns": assistant_turns,
        "predicted": [],
        "correct_predicted": [],
        "missing": list(canonicals),
        "over_predicted": [],
    }


def compute_score_qampari(
    solution_data: str | list,
    ground_truth: Any,
) -> dict[str, Any]:
    """Score multi-answer QA with alias-aware matching.

    ground_truth shape: list[list[str]] — outer = gold entities,
    inner = aliases for that entity. An entity counts as recalled iff
    any of its aliases appears (after normalization) in the predictions.
    """

    if isinstance(solution_data, list):
        solution_str = "\n".join(str(m.get("content", "")) for m in solution_data)
        assistant_str = "\n".join(
            str(m.get("content", "")) for m in solution_data if m.get("role") == "assistant"
        )
        num_turns = len(solution_data)
        assistant_turns = sum(1 for m in solution_data if m.get("role") == "assistant")
        user_messages = [str(m.get("content", "")) for m in solution_data if m.get("role") == "user"]
        # Skip the first user message (the question itself); the rest are tool responses.
        tool_results_str = "\n".join(user_messages[1:]) if len(user_messages) > 1 else ""
    else:
        print("Warning: solution_data is not a list of messages. num_turns and assistant_turns set to 1.")
        solution_str = str(solution_data)
        assistant_str = solution_str
        num_turns = 1
        assistant_turns = 1
        tool_results_str = ""

    character_length = len(solution_str)

    gt_alias_sets, gt_canonicals = _normalize_entities(ground_truth)
    num_gt = len(gt_alias_sets)
    all_aliases: set[str] = set().union(*gt_alias_sets) if gt_alias_sets else set()

    normalized_tool_results = normalize_answer(tool_results_str)
    matches_in_traj = [
        canon
        for aliases, canon in zip(gt_alias_sets, gt_canonicals)
        if any(alias in normalized_tool_results for alias in aliases)
    ]

    call_count = compute_tool_count(solution_str)
    predicted_answers = get_answers(assistant_str)

    if predicted_answers is None:
        return _format_error_metrics(
            call_count=call_count,
            num_gt=num_gt,
            canonicals=gt_canonicals,
            matches_in_traj=matches_in_traj,
            character_length=character_length,
            num_turns=num_turns,
            assistant_turns=assistant_turns,
        )

    pred_set = _normalize_predictions(predicted_answers)
    num_pred = len(pred_set)

    # Entity-level recall: a gold entity counts iff any of its aliases is predicted.
    matched_canonicals = [
        canon
        for aliases, canon in zip(gt_alias_sets, gt_canonicals)
        if pred_set & aliases
    ]
    matched_count = len(matched_canonicals)

    # Prediction-level precision: predictions that match any alias of any gold entity.
    correct_pred = pred_set & all_aliases
    over_pred = pred_set - all_aliases

    precision = len(correct_pred) / num_pred if num_pred else 0.0
    recall = matched_count / num_gt if num_gt else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    def recall_at(k: int) -> float:
        return min(matched_count / min(k, num_gt), 1.0) if num_gt else 0.0

    missing = [
        canon
        for aliases, canon in zip(gt_alias_sets, gt_canonicals)
        if not (pred_set & aliases)
    ]

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "recall@5": float(recall_at(5)),
        "recall@10": float(recall_at(10)),
        "recall@20": float(recall_at(20)),
        "recall@50": float(recall_at(50)),
        "exact_match": 1.0 if f1_score == 1.0 else 0.0,
        "num_predicted": int(num_pred),
        "num_target": int(num_gt),
        "call_count": int(call_count),
        "entity_per_search": float(compute_entity_per_search(matched_count, call_count)),
        "repetition_count": int(compute_repetition_count(predicted_answers)),
        "format_error": 0.0,
        "matches_in_traj": matches_in_traj,
        "num_matches_in_traj": len(matches_in_traj),
        "character_length": character_length,
        "num_turns": num_turns,
        "assistant_turns": assistant_turns,
        "predicted": list(pred_set),
        "correct_predicted": matched_canonicals,
        "missing": missing,
        "over_predicted": list(over_pred),
    }
