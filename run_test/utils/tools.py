import requests
import json
import logging
import time

logger = logging.getLogger(__name__)

# Constants for default values
DEFAULT_SEARCH_URL = "http://127.0.0.1:8000/retrieve"
DEFAULT_TOP_K = 5
DEFAULT_INITIAL_K = 50
DEFAULT_MAX_CHARS = 10000
DEFAULT_TIMEOUT = 300
DEFAULT_RETRIES = 3

SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches the web using keyword-based retrieval (BM25). Returns top-5 results per query by default. Note: Multiple queries in 'query_list' are subject to a 10,000-character total limit, triggering a dynamic top-k strategy that distributes results evenly across all queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "A list of search queries. Include specific keywords that are likely to appear in the target documents."
                }
            },
            "required": ["query_list"]
        }
    }
}

class SearchTool:
    def __init__(self, 
                 url: str = DEFAULT_SEARCH_URL, 
                 top_k: int = DEFAULT_TOP_K, 
                 initial_k: int = DEFAULT_INITIAL_K, 
                 max_chars: int = DEFAULT_MAX_CHARS,
                 timeout: int = DEFAULT_TIMEOUT,
                 retries: int = DEFAULT_RETRIES):
        self.url = url
        self.top_k = top_k
        self.initial_k = initial_k
        self.max_chars = max_chars
        self.timeout = timeout
        self.retries = retries
        self.schema = SEARCH_TOOL_SCHEMA
    
    def search(self, query_list: list[str]) -> list[dict]:
        """
        Executes the search via the API.
        
        Args:
            query_list: List of queries to search.
            
        Returns:
            A list of search results.
        """
        if not query_list:
            return []

        last_exception = None
        for attempt in range(self.retries + 1):
            try:
                # The retrieval server expects "queries": list[str]
                response = requests.post(
                    self.url,
                    json={
                        "queries": query_list, 
                        "topk": self.top_k, 
                        "max_chars": self.max_chars, 
                        "initial_k": self.initial_k
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                return data
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed for search tool: {e}")
                if attempt < self.retries:
                    time.sleep(1)

        logger.error(f"All {self.retries + 1} attempts failed for search tool: {last_exception}")
        return [{"error": str(last_exception)}]

    def __call__(self, query_list: list[str]) -> str:
        results = self.search(query_list)
        
        if isinstance(results, dict) and "result" in results:
            final_result = json.dumps(results["result"], indent=2, ensure_ascii=False)
            if "truncation_warning" in results and results["truncation_warning"]:
                final_result += f"\n{results['truncation_warning']}"
        else:
            final_result = json.dumps(results, indent=2, ensure_ascii=False)
            
        content = f"<tool_response>{final_result}</tool_response>"
        return content
