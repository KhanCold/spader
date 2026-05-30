from run_test.models.base import BaseModel
from openai import OpenAI
import os
import json

class VLLMBase(BaseModel):
    def __init__(self, model_name: str = "qwen3-8b-base", api_base: str = None, api_key: str = "EMPTY", system_prompt: str = None):
        super().__init__(model_name, api_base, api_key, system_prompt)
        
        self.model_name = model_name

        self.client = OpenAI(
            base_url=api_base if api_base else "http://localhost:80001/v1",
            api_key=api_key
        )

    def generate(self, messages: list[dict], tools: list[dict] = None, stop: list[str] = None, max_tokens: int = 5096, temperature: float = 1.0) -> str:
        
        completion_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if tools:
            completion_params["tools"] = tools
        if stop:
            completion_params["stop"] = stop

        response = self.client.chat.completions.create(**completion_params)
        
        # Convert native function calls to the XML-like format used by the evaluator.
        if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls
            content = response.choices[0].message.content or ""
            
            for tc in tool_calls:
                # Basic handling to append XML-like structure if needed by the evaluator
                # If the evaluator expects specific XML format and the model returns structured tools
                try:
                    func_args = tc.function.arguments
                    args_dict = json.loads(func_args)
                    tool_call_struct = {
                        "name": tc.function.name,
                        "arguments": args_dict
                    }
                    content += f"\n<tool_call>\n{json.dumps(tool_call_struct)}\n</tool_call>"
                except:
                    pass
            return content

        return response.choices[0].message.content
