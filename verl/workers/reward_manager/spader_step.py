# Copyright 2026
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

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score.qa_em_qampari import compute_score_qampari
from verl.utils.reward_score.qampari_spader_step import (
    compute_group_entity_frequency,
    compute_spader_step_rewards,
    compute_step_discovery,
)
from run_test.utils.metrics import _parse_target_list, compute_tool_count
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _safe_get_gt_list(ground_truth: Any) -> list[str]:
    target = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
    target_list = _parse_target_list(target)
    return [t for t in (str(x) for x in target_list) if t.strip()]


def _extract_tool_step_token_spans(response_mask: torch.Tensor, valid_len: int) -> tuple[list[int], list[tuple[int, int]]]:
    """Find tool steps using response_mask transitions.

    Returns:
        decision_token_positions: list[int]
            Token indices (within response) where the tool-call decision happened.
            We assign the step reward to these indices (mask=1 tokens).
        tool_spans: list[(start, end)]
            Spans [start, end) where response_mask==0 (tool responses / inserted messages).
            Each span is one tool-processing round.
    """
    decision_token_positions: list[int] = []
    tool_spans: list[tuple[int, int]] = []

    if valid_len <= 0:
        return decision_token_positions, tool_spans

    # clamp
    rm = response_mask[:valid_len].to(dtype=torch.int64)
    in_tool = False
    tool_start = 0

    for i in range(valid_len):
        is_tool = int(rm[i].item()) == 0
        if is_tool and not in_tool:
            in_tool = True
            tool_start = i
            # decision token is previous token if exists (and should be assistant token)
            if i - 1 >= 0:
                decision_token_positions.append(i - 1)
        elif (not is_tool) and in_tool:
            in_tool = False
            tool_spans.append((tool_start, i))

    if in_tool:
        tool_spans.append((tool_start, valid_len))

    return decision_token_positions, tool_spans


@register("spader_step")
class SpaderStepRewardManager(AbstractRewardManager):
    """SPADER step-wise discovery reward for multi-turn retrieval.

    This reward manager produces *token-level* rewards:
    - For each tool-processing round, assign discovery reward to the tool-call decision token.
    - Optionally add a final answer-quality reward to the last assistant token.

    It computes group frequency N(e, T) within each prompt group (grouped by uid).
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        alpha: float = 1.0,
        beta: float = 4.0,
        lambda_val: float = 1.0,
        tool_round_cost: float = 0.1,
        final_answer_weight: float = 1.0,
        enable_final_answer_reward: bool = True,
        cost_free_steps: int = 0,
        cost_rampup_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambda_val = float(lambda_val)
        self.tool_round_cost = float(tool_round_cost)
        self.final_answer_weight = float(final_answer_weight)
        self.enable_final_answer_reward = bool(enable_final_answer_reward)
        self.cost_free_steps = int(cost_free_steps)
        self.cost_rampup_steps = int(cost_rampup_steps)

    def _get_effective_cost(self, global_step: int) -> float:
        """Compute effective tool_round_cost with optional warmup schedule.

        Schedule:
            step < cost_free_steps                              → 0
            cost_free_steps ≤ step < cost_free_steps + rampup   → linear 0 → tool_round_cost
            step ≥ cost_free_steps + rampup                     → tool_round_cost
        """
        if self.tool_round_cost == 0.0:
            return 0.0
        if global_step < self.cost_free_steps:
            return 0.0
        elapsed = global_step - self.cost_free_steps
        if self.cost_rampup_steps > 0 and elapsed < self.cost_rampup_steps:
            return self.tool_round_cost * (elapsed / self.cost_rampup_steps)
        return self.tool_round_cost

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        # We produce token-level scores aligned with response ids.
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        batch_size = len(data)

        # Dynamic cost schedule based on global training step.
        global_step = data.meta_info.get("global_steps", 0) if data.meta_info else 0
        effective_cost = self._get_effective_cost(global_step)
        # IMPORTANT: reward_extra_info values must align with the original batch order.
        # Downstream logging/dumping assumes metric_vals[i] corresponds to sample i.
        reward_extra_info: dict[str, list] = {}

        prompt_len = data.batch["prompts"].shape[-1]
        attention_mask = data.batch["attention_mask"]
        response_mask = data.batch["response_mask"]

        # Group by uid (prompt group in VERL)
        assert "uid" in data.non_tensor_batch, "SpaderStepRewardManager requires non_tensor_batch['uid']"
        uids = data.non_tensor_batch["uid"]
        uid2indices: dict[Any, list[int]] = defaultdict(list)
        for i in range(len(data)):
            uid2indices[uids[i]].append(i)

        # Pre-decode responses and tool step texts per sample for later group computation.
        # We only decode valid lengths (excluding padding).
        per_sample = []
        for i in range(len(data)):
            # calculate valid length, without padding
            valid_len = int(attention_mask[i, prompt_len:].sum().item())
            valid_len = max(valid_len, 0)
            response_ids = data.batch["responses"][i][:valid_len]
            rm = response_mask[i][:valid_len]

            # extract tool step spans and decision token positions
            decision_positions, tool_spans = _extract_tool_step_token_spans(rm, valid_len)
            tool_texts = []
            call_counts = []
            last_end = 0
            for (s, e) in tool_spans:
                if s > last_end:
                    seg = self.tokenizer.decode(response_ids[last_end:s], skip_special_tokens=True)
                    call_counts.append(compute_tool_count(seg))
                else:
                    call_counts.append(0)

                if e <= s:
                    tool_texts.append("")
                else:
                    # tool spans are mask==0; still decode for entity discovery
                    tool_texts.append(self.tokenizer.decode(response_ids[s:e], skip_special_tokens=True))
                last_end = e

            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            ground_truth = data[i].non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
            gt_list = _safe_get_gt_list(ground_truth)

            per_sample.append(
                {
                    "valid_len": valid_len,
                    "response_ids": response_ids,
                    "response_text": response_text,
                    "decision_positions": decision_positions,
                    "tool_texts": tool_texts,
                    "call_counts": call_counts,
                    "gt_list": gt_list,
                }
            )

        # Compute rewards per group (needs N(e,T) across group).
        for uid, idxs in uid2indices.items():
            #uid: group id, idxs:[1,2,3,...] group member indices in the batch
            # Step 1: per-trajectory discovery
            discoveries = []
            found_union = []
            for i in idxs:
                gt_list = per_sample[i]["gt_list"]
                disc = compute_step_discovery(per_sample[i]["tool_texts"], gt_list)
                discoveries.append(disc)
                union = set().union(*disc.found_by_step) if disc.found_by_step else set()
                found_union.append(union)

            # Step 2: group frequency for entities
            group_freq = compute_group_entity_frequency(found_union)

            # Step 3: compute per-sample step rewards and write token-level scores
            for local_j, i in enumerate(idxs):
                gt_list = per_sample[i]["gt_list"]
                gt_size = len(gt_list)
                disc = discoveries[local_j]

                step_rewards = compute_spader_step_rewards(
                    new_by_step=disc.new_by_step,
                    gt_size=gt_size,
                    group_freq=group_freq,
                    alpha=self.alpha,
                    beta=self.beta,
                    lambda_val=self.lambda_val,
                )

                # Tool cost (per tool-processing round) with dynamic schedule.
                if effective_cost != 0.0 and step_rewards:
                    counts = per_sample[i]["call_counts"]
                    # Formula: r_t = (1/|GT|) * [ Discovery - c * Count ]
                    # step_rewards currently holds (1/|GT|) * Discovery
                    # So we subtract (c * Count) / |GT|
                    if gt_size > 0:
                        step_rewards = [r - (effective_cost * c) / gt_size for r, c in zip(step_rewards, counts)]

                # Assign each step reward to the corresponding decision token.
                decision_positions = per_sample[i]["decision_positions"]
                # If the parser cannot find enough decision positions, align by min length.
                k = min(len(step_rewards), len(decision_positions))
                for t in range(k):
                    pos = decision_positions[t]
                    if 0 <= pos < per_sample[i]["valid_len"]:
                        reward_tensor[i, pos] += float(step_rewards[t])

                # Optionally add final answer reward at last assistant token.
                final_metrics = compute_score_qampari(per_sample[i]["response_text"], gt_list)
                
                raw_final_score = final_metrics["f1_score"]
                final_reward = self.final_answer_weight * raw_final_score if self.enable_final_answer_reward else 0.0

                # last assistant token index (mask==1)
                rm = response_mask[i][: per_sample[i]["valid_len"]]
                assistant_positions = (rm == 1).nonzero(as_tuple=True)[0]
                if assistant_positions.numel() > 0:
                    last_pos = int(assistant_positions[-1].item())
                else:
                    last_pos = max(per_sample[i]["valid_len"] - 1, 0)

                if 0 <= last_pos < per_sample[i]["valid_len"]:
                    reward_tensor[i, last_pos] += float(final_reward)

                # Validation/logging metrics: keep the same keys as qa_em_qampari
                for mk, mv in final_metrics.items():
                    if mk not in reward_extra_info:
                        reward_extra_info[mk] = [0.0] * batch_size
                    reward_extra_info[mk][i] = float(mv)

                # 1. step new entities
                if "step_new_entities" not in reward_extra_info:
                    reward_extra_info["step_new_entities"] = [None] * batch_size
                reward_extra_info["step_new_entities"][i] = json.dumps([list(step) for step in disc.new_by_step])

                # 2. step rewards
                if "step_rewards" not in reward_extra_info:
                    reward_extra_info["step_rewards"] = [None] * batch_size
                # Convert to Python floats for JSON serialization.
                reward_extra_info["step_rewards"][i] = json.dumps([float(r) for r in step_rewards])

                # 3. group entity frequency
                if "group_entity_freq" not in reward_extra_info:
                    reward_extra_info["group_entity_freq"] = [None] * batch_size
                # Fill missing GT entities with zero frequency.
                all_entities = {str(e) for e in gt_list if str(e).strip()}
                
                group_freq_full = {e: int(group_freq.get(e, 0)) for e in all_entities}
                reward_extra_info["group_entity_freq"][i] = json.dumps(group_freq_full)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
