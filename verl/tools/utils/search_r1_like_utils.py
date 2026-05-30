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

import json
import logging
import threading
import time
import traceback
import uuid
from typing import Any, Optional

import requests

DEFAULT_TIMEOUT = 30  # Default search request timeout
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10
DEFAULT_INITIAL_K = 50
DEFAULT_MAX_CHARS = 10000

logger = logging.getLogger(__name__)


def call_search_api(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    initial_k: int = DEFAULT_INITIAL_K,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: int = DEFAULT_TIMEOUT,
    use_reranker: Optional[bool] = None,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Calls the remote search API to perform retrieval with retry logic for various errors,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        initial_k: Initial number of results to fetch before reranking/filtering.
        max_chars: Maximum characters for result content.
        timeout: Request timeout in seconds.
        use_reranker: Whether to use reranker (if supported/configured on server).

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {
        "queries": query_list,
        "topk": topk,
        "initial_k": initial_k,
        "max_chars": max_chars,
        "use_reranker": use_reranker
    }

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}"
            )
            response = requests.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = (
                    f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix}Search API call successful on attempt {attempt + 1}")
            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Search API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def _passages2string(retrieval_result):
    """Convert retrieval results to formatted string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()


def perform_single_search_batch(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    initial_k: int = DEFAULT_INITIAL_K,
    max_chars: int = DEFAULT_MAX_CHARS,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
    use_reranker: Optional[bool] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Performs a single batch search for multiple queries (original search tool behavior).

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        initial_k: Initial number of results to fetch.
        max_chars: Max characters per result.
        concurrent_semaphore: Optional semaphore for concurrency control.
        timeout: Request timeout in seconds.
        use_reranker: Control using reranker.

    Returns:
        A tuple (result_text, metadata).
        result_text: The search result JSON string.
        metadata: Metadata dictionary for the batch search.
    """
    logger.info(f"Starting batch search for {len(query_list)} queries.")

    api_response = None
    error_msg = None

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_search_api(
                    retrieval_service_url=retrieval_service_url,
                    query_list=query_list,
                    topk=topk,
                    initial_k=initial_k,
                    max_chars=max_chars,
                    timeout=timeout,
                    use_reranker=use_reranker,
                )
        else:
            api_response, error_msg = call_search_api(
                retrieval_service_url=retrieval_service_url,
                query_list=query_list,
                topk=topk,
                initial_k=initial_k,
                max_chars=max_chars,
                timeout=timeout,
                use_reranker=use_reranker,
            )
    except Exception as e:
        error_msg = f"API Request Exception during batch search: {e}"
        logger.error(f"Batch search: {error_msg}")
        traceback.print_exc()

    metadata = {
        "query_count": len(query_list),
        "queries": query_list,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "total_results": 0,
        "formatted_result": None,
    }

    if error_msg:
        metadata["status"] = "api_error"
        # Match tools.py error structure
        error_res = [{"error": error_msg}]
        result_text = f"<tool_response>{json.dumps(error_res, indent=2, ensure_ascii=False)}</tool_response>"
        logger.error(f"Batch search: API error occurred: {error_msg}")
    elif api_response:
        logger.debug(f"Batch search: API Response: {api_response}")
        metadata["api_response"] = api_response
        metadata["status"] = "success"
        
        # Match tools.py return structure
        result_text = f"<tool_response>{json.dumps(api_response, indent=2, ensure_ascii=False)}</tool_response>"
    else:
        metadata["status"] = "unknown_api_state"
        result_text = f"<tool_response>{json.dumps([{'error': 'Unknown API state (no response and no error message).'}], indent=2, ensure_ascii=False)}</tool_response>"
        logger.error("Batch search: Unknown API state.")

    return result_text, metadata
