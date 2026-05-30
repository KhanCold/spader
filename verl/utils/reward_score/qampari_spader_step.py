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

"""QAMPARI SPADER step-wise discovery reward helpers.

This module is intentionally model/trainer agnostic. It contains text parsing
and reward computation utilities used by the step-wise reward manager.

Design goals:
- Only credit *correct* entities (must appear in ground-truth list).
- Encourage novelty across a prompt group via 1/N(e, T).
- Provide step-level reward signals suitable for dense/token-level reward.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from run_test.utils.metrics import normalize_answer


def extract_gt_entities_from_text(text: str, gt_entities: Iterable[str]) -> set[str]:
    """Return the subset of GT entities that appear in text (robust normalized substring match)."""
    if not text:
        return set()

    text_norm = normalize_answer(text)
    if not text_norm:
        return set()

    found: set[str] = set()
    for ent in gt_entities:
        ent_norm = normalize_answer(ent)
        if ent_norm and ent_norm in text_norm:
            found.add(ent)
    return found

@dataclass
class StepDiscoveryResult:
    found_by_step: list[set[str]]
    new_by_step: list[set[str]]


def compute_step_discovery(tool_texts_by_step: list[str], gt_entities: list[str]) -> StepDiscoveryResult:
    """Given tool-response texts for each step, compute found/new GT entities per step."""
    found_by_step: list[set[str]] = []
    new_by_step: list[set[str]] = []

    cumulative: set[str] = set()
    for tool_text in tool_texts_by_step:
        found = extract_gt_entities_from_text(tool_text, gt_entities)
        found_by_step.append(found)
        new = found - cumulative
        new_by_step.append(new)
        cumulative |= found

    return StepDiscoveryResult(found_by_step=found_by_step, new_by_step=new_by_step)


def compute_group_entity_frequency(per_traj_found_union: list[set[str]]) -> Counter:
    """N(e, T): number of trajectories in the group that found entity e at least once."""
    freq: Counter = Counter()
    for found in per_traj_found_union:
        for e in found:
            freq[e] += 1
    return freq


def compute_spader_step_rewards(
    new_by_step: list[set[str]],
    gt_size: int,
    group_freq: Counter,
    alpha: float,
    beta: float,
    lambda_val: float = 1.0,
) -> list[float]:
    """r_t^{(i)} = (1/|GT|) * sum_{e in E_new,t} (alpha + beta * exp(-lambda * (N(e) - 1)))."""
    if gt_size <= 0:
        return [0.0 for _ in new_by_step]

    rewards: list[float] = []
    for new_set in new_by_step:
        step_sum = 0.0
        for e in new_set:
            n = float(group_freq.get(e, 1))
            step_sum += alpha + beta * math.exp(-lambda_val * (n - 1))
        rewards.append(step_sum / float(gt_size))
    return rewards
