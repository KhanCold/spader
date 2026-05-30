import json
import logging
import re
from typing import List, Dict, Any, Tuple

from run_test.models.base import BaseModel
from run_test.utils.tools import SearchTool
from run_test.utils.metrics import compute_score_qampari

logger = logging.getLogger(__name__)

DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the question. You must strictly follow this loop for every step:\n"
    "1. Always start with reasoning inside <think>...</think>.\n"
    "2. Then, output either a search tool call OR the final answer.\n"
    "   - To search: <tool_call>[\"query1\", \"query2\"]</tool_call>\n"
    "     (System returns top-5 100-word snippets per query in <tool_response>)\n"
    "   - To answer: <answer>[\"Entity A\", \"Entity B\"]</answer>\n"
    "Both <tool_call> and <answer> content must be a JSON list of strings.\n\n"
    "Question: {question}"
)

def parse_tool_call(content: str):
    """Parses the LAST tool call in the content.
    
    Supports two formats:
    - Simple list: <tool_call>["query1", "query2"]</tool_call>
    - Legacy dict: <tool_call>{"name":"search","arguments":{"query_list":[...]}}</tool_call>
    
    Returns a list of query strings, or None if no tool call tag found.
    Raises on parse errors so the caller can feed the real error back to the model.
    """
    def _normalize_query_list(value: Any) -> List[str]:
        if isinstance(value, str):
            q = value.strip()
            return [q] if q else []
        if not isinstance(value, list):
            return []

        normalized: List[str] = []
        for item in value:
            if isinstance(item, str):
                q = item.strip()
                if q:
                    normalized.append(q)
            elif isinstance(item, list):
                normalized.extend(_normalize_query_list(item))
        return normalized

    matches = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
    if not matches:
        return None

    tool_call_json = json.loads(matches[-1])
    # Simple list format: ["query1", "query2"] (also tolerates nested lists)
    if isinstance(tool_call_json, list):
        return _normalize_query_list(tool_call_json)
    # Legacy dict format: {"name": "search", "arguments": {"query_list": [...]}}
    if isinstance(tool_call_json, dict):
        query_list = tool_call_json.get("arguments", {}).get("query_list", [])
        return _normalize_query_list(query_list)
    raise ValueError(f"Unexpected tool call format: {type(tool_call_json)}")

def run_single_sample(
    index: int,
    data_item: Dict[str, Any],
    model: BaseModel,
    search_tool: SearchTool, 
    max_turns: int = 16,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    question = data_item.get("question_text", "")
    if not question:
        # Fallback for different data formats
        question = data_item.get("question", "")
        
    ground_truth = data_item.get("ground_truth")  # Extract ground truth directly
    data_source = data_item.get("data_source")
    
    # Construct complete prompt including system prompt
    messages = []

    system_prompt_content = model.get_system_prompt()
    if system_prompt_content:
        system_msg = {"role": "system", "content": system_prompt_content}
        messages.append(system_msg)
    
    formatted_question = DEFAULT_USER_CONTENT_PREFIX.format(question=question)
    
    user_msg = {"role": "user", "content": formatted_question}
    
    messages.append(user_msg)

    for turn in range(max_turns):
        try:
            content = model.retry_generate(messages)
            messages.append({"role": "assistant", "content": content})

            try:
                query_list = parse_tool_call(content)
            except Exception as e:
                logger.warning(f"Failed to parse tool call: {e}")
                tool_output = (
                    f"<tool_response>Your tool call could not be parsed. "
                    f"Error: {e}. "
                    f"Please output valid JSON inside <tool_call>...</tool_call> tags and try again.</tool_response>"
                )
                messages.append({"role": "user", "content": tool_output})
                continue

            if query_list is not None:
                if not query_list:
                    error_msg = json.dumps({"result": "Error: 'query_list' is missing, empty, or not a list in parameters."})
                    tool_output = f"<tool_response>{error_msg}</tool_response>"
                else:
                    try:
                        tool_output = search_tool(query_list)
                    except Exception as e:
                        tool_output = f"<tool_response>Error when executing tool: {e}</tool_response>"

                messages.append({"role": "user", "content": tool_output})
            else:
                # No tool call => Model has finished thinking/answering
                break

        except Exception as e:
            logger.error(f"Error in turn {turn}: {e}")
            break
    # Calculate score
    metrics = compute_score_qampari(messages, ground_truth)

    result = {
        "index": index,
        "question": question,
        "ground_truth": ground_truth,
        "data_source": data_source,
        "history": messages # Structured log
    }
    
    # Merge metrics into result
    result.update(metrics)
    
    return result, metrics
