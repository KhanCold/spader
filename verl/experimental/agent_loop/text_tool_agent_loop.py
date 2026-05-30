# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Text-based multi-turn tool agent loop.

This agent loop is designed for models that are trained with tool instructions
baked into the user prompt (e.g. ``DEFAULT_USER_CONTENT_PREFIX``), rather than
relying on function-call schemas injected via ``apply_chat_template(tools=...)``.

Key differences from :class:`ToolAgentLoop`:

1. **No tool-schema injection** – ``apply_chat_template`` is called with
   ``tools=None`` so the tokeniser never inserts special function-call tokens.
2. **Plain-text tool response** – Tool responses are appended as a ``user``
   message (matching the inference script) instead of the ``tool`` role, so
   training and inference behave identically.
3. **Compatible parser** – Pair this loop with the ``text`` format ToolParser
   which extracts bare ``<tool_call>["q1","q2"]</tool_call>`` lists.
"""
import asyncio
import logging
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import (
    register,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("text_tool_agent")
class TextToolAgentLoop(ToolAgentLoop):
    """Multi-turn agent loop that uses plain-text ``<tool_call>`` / ``<tool_response>`` tags.

    The model is expected to produce ``<tool_call>[...]</tool_call>`` and the
    tool response is fed back as a ``user`` message whose content is the raw
    text returned by the tool (which already contains ``<tool_response>`` tags
    when using :class:`verl.tools.search_tool.SearchTool`).

    Usage – set these in your YAML config::

        actor_rollout_ref.rollout.agent.default_agent_loop: text_tool_agent
        actor_rollout_ref.rollout.multi_turn.format: text
    """

    # ------------------------------------------------------------------
    # Override: do NOT inject tool schemas into the chat template.
    # ------------------------------------------------------------------
    async def _handle_pending_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        """Prepare prompt without injecting function-call tool schemas."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=None,  # ← key difference: no schema injection
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    # ------------------------------------------------------------------
    # Override: append tool responses as ``user`` messages (plain text).
    # ------------------------------------------------------------------
    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Execute tools and append results as plain-text user messages.

        This matches the inference script behaviour where tool output is added
        as ``{"role": "user", "content": tool_output_text}``.
        """
        tasks = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Combine all tool response texts into a single user message, just
        # like the inference script does for each turn.
        combined_text_parts: list[str] = []
        for tool_response, tool_reward, _ in responses:
            text = tool_response.text or ""
            combined_text_parts.append(text)
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        combined_text = "\n".join(combined_text_parts)

        # Append as a user message (not tool role) to match inference behaviour.
        user_message = {"role": "user", "content": combined_text}
        agent_data.messages.append(user_message)

        # Tokenise the new user message.
        response_ids = await self.apply_chat_template(
            [user_message],
            images=None,
            videos=None,
            remove_system_prompt=True,
        )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        # Mark tool-response tokens as non-trainable (mask=0).
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING
