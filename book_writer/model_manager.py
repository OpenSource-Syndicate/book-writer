"""
Book Writer System - Model Manager
Manages different Ollama models for various tasks
"""

import ollama
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from book_writer.config import model_config
from book_writer.response_handler import validate_json_response, extract_function_calls
from book_writer.tool_registry import tool_registry


class OllamaModelManager:
    """Model manager for handling different Ollama models based on task requirements."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.config = model_config
        self.ollama_client = ollama.Client(host=self.config.get_ollama_config().get("base_url", "http://localhost:11434"))
    
    def generate_response(
        self, 
        prompt: str, 
        task: str, 
        system_prompt: Optional[str] = None,
        format_json: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        functions: Optional[List[Dict]] = None
    ) -> Union[str, Dict[str, Any]]:
        """Generate a response using the appropriate model based on the task.
        
        Args:
            prompt: The input prompt
            task: The task type (determines which model to use)
            system_prompt: Optional system prompt to guide the model
            format_json: Whether to request JSON output
            temperature: Temperature setting (overrides config if provided)
            max_tokens: Max tokens setting (overrides config if provided)
            top_p: Top-p setting (overrides config if provided)
            functions: Optional list of function definitions for function calling
            
        Returns:
            Generated response as string or dict (if JSON requested)
        """
        # Get model configuration for the specific task
        model_cfg = self.config.get_model_config(task)
        
        # Use provided parameters or fall back to config
        model_name = model_cfg["model_name"]
        temp = temperature or model_cfg.get("temperature", 0.7)
        max_tok = max_tokens or model_cfg.get("max_tokens", 1024)
        top_p_val = top_p or model_cfg.get("top_p", 0.9)
        
        # Prepare the messages for the model
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Prepare the request options
        options = {
            "temperature": temp,
            "top_p": top_p_val,
        }
        
        # Prepare request parameters
        request_params = {
            "model": model_name,
            "messages": messages,
            "options": options,
            "stream": False
        }
        
        # Add format parameter if JSON is requested
        if format_json:
            request_params["format"] = "json"
        
        # Add functions if provided (for function calling)
        if functions:
            request_params["tools"] = functions
            
        try:
            # Make the API call
            response = self.ollama_client.chat(**request_params)
            
            # Extract content from response
            content = response["message"]["content"]
            
            # If JSON format was requested, parse the response
            if format_json or functions:
                try:
                    # Clean up the response content in case it includes formatting
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]  # Remove ```json
                    if content.endswith("```"):
                        content = content[:-3]  # Remove ```
                    content = content.strip()
                    
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return as string
                    return content
            else:
                return content
        except Exception as e:
            print(f"Error generating response with model {model_name}: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_response_stream(
        self, 
        prompt: str, 
        task: str, 
        system_prompt: Optional[str] = None,
        format_json: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        functions: Optional[List[Dict]] = None,
        callback: Optional[callable] = None
    ) -> str:
        """Generate a streaming response using the appropriate model based on the task.
        
        Args:
            prompt: The input prompt
            task: The task type (determines which model to use)
            system_prompt: Optional system prompt to guide the model
            format_json: Whether to request JSON output
            temperature: Temperature setting (overrides config if provided)
            max_tokens: Max tokens setting (overrides config if provided)
            top_p: Top-p setting (overrides config if provided)
            functions: Optional list of function definitions for function calling
            callback: Optional callback to handle each chunk of response
            
        Returns:
            Complete generated response as string
        """
        # Get model configuration for the specific task
        model_cfg = self.config.get_model_config(task)
        
        # Use provided parameters or fall back to config
        model_name = model_cfg["model_name"]
        temp = temperature or model_cfg.get("temperature", 0.7)
        max_tok = max_tokens or model_cfg.get("max_tokens", 1024)
        top_p_val = top_p or model_cfg.get("top_p", 0.9)
        
        # Prepare the messages for the model
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Prepare the request options
        options = {
            "temperature": temp,
            "top_p": top_p_val,
        }

        # Prepare request parameters for streaming
        request_params = {
            "model": model_name,
            "messages": messages,
            "options": options,
            "stream": True  # Enable streaming
        }

        # Add format parameter if JSON is requested
        if format_json:
            request_params["format"] = "json"

        # Add functions if provided (for function calling)
        if functions:
            request_params["tools"] = functions
            
        try:
            # Create a complete response string
            full_response = ""
            
            # Make the streaming API call
            stream = self.ollama_client.chat(**request_params)
            
            # Process the streaming response
            for chunk in stream:
                if chunk['done']:
                    break
                message = chunk['message']
                content = message.get('content', '')
                
                if content:
                    full_response += content
                    
                    # Call the callback function with the new content if provided
                    if callback:
                        callback(content)
            
            # If JSON format was requested, parse the response
            if format_json or functions:
                try:
                    # Clean up the response content in case it includes formatting
                    full_response = full_response.strip()
                    if full_response.startswith("```json"):
                        full_response = full_response[7:]  # Remove ```json
                    if full_response.endswith("```"):
                        full_response = full_response[:-3]  # Remove ```
                    full_response = full_response.strip()
                    
                    return json.loads(full_response)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return as string
                    return full_response
            else:
                return full_response
        except Exception as e:
            print(f"Error generating streaming response with model {model_name}: {e}")
            error_msg = f"Error generating response: {str(e)}"
            if callback:
                callback(error_msg)
            return error_msg
    
    def execute_function_call(
        self,
        prompt: str,
        task: str,
        functions: List[Dict],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a function call using the appropriate model.
        
        Args:
            prompt: The input prompt that triggers a function call
            task: The task type (determines which model to use)
            functions: List of function definitions
            system_prompt: Optional system prompt to guide the model
            
        Returns:
            Dictionary containing the function call result
        """
        # Use the generate_response method with functions parameter
        result = self.generate_response(
            prompt=prompt,
            task=task,
            system_prompt=system_prompt,
            functions=functions
        )
        
        # If the result contains function calls, return them
        if isinstance(result, dict):
            return result
        else:
            return {"result": result, "raw_response": result}
    
    def execute_function_sequence(
        self,
        prompt: str,
        task: str,
        functions: List[Dict],
        system_prompt: Optional[str] = None,
        max_iterations: int = 3
    ) -> List[Dict[str, Any]]:
        """Execute a sequence of function calls based on the model's responses.
        
        Args:
            prompt: The initial input prompt
            task: The task type (determines which model to use)
            functions: List of function definitions
            system_prompt: Optional system prompt to guide the model
            max_iterations: Maximum number of function call iterations
            
        Returns:
            List of function call results
        """
        results = []
        current_prompt = prompt
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Generate response with potential function calls
            response = self.generate_response(
                prompt=current_prompt,
                task=task,
                system_prompt=system_prompt,
                functions=functions
            )
            
            # If the response contains function calls, execute them
            if isinstance(response, dict):
                # Add the response to results
                results.append(response)
                
                if "tool_calls" in response:
                    # Process each tool call
                    tool_results = []
                    for tool_call in response["tool_calls"]:
                        function_name = tool_call["function"]["name"]
                        function_args = json.loads(tool_call["function"]["arguments"])
                        
                        # Execute the function (in this system, we'll simulate some common functions)
                        result = self._execute_single_function(function_name, function_args)
                        tool_results.append({
                            "function_name": function_name,
                            "arguments": function_args,
                            "result": result
                        })
                    
                    # Create a new prompt that includes the tool results for the next iteration
                    tool_results_text = "\n".join([
                        f"Function '{tr['function_name']}' with arguments {tr['arguments']} returned: {tr['result']}"
                        for tr in tool_results
                    ])
                    current_prompt = f"Previous function calls and results:\n{tool_results_text}\n\nBased on these results, continue with the task: {prompt}"
                else:
                    # No more function calls, return the results
                    break
            else:
                # If response is not a dict, treat it as a final answer
                results.append({"result": response})
                break
        
        return results
    
    def _execute_single_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a single function based on its name and arguments.
        
        Args:
            function_name: The name of the function to execute
            arguments: The arguments to pass to the function
            
        Returns:
            The result of the function execution
        """
        # Use the global tool registry to execute the function
        return tool_registry.execute_tool(function_name, arguments)
    
    def health_check(self) -> bool:
        """Check if the Ollama service is accessible.
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            # Try to list models as a basic health check
            self.ollama_client.list()
            return True
        except Exception:
            return False
    
    def list_available_models(self) -> List[str]:
        """List all available models in Ollama.
        
        Returns:
            List of available model names
        """
        try:
            models_response = self.ollama_client.list()
            # The response from ollama.list() is a dict with a "models" key containing a list
            if isinstance(models_response, dict) and "models" in models_response:
                models_list = models_response["models"]
                return [model["name"] for model in models_list]
            else:
                # Fallback if response format is different
                return []
        except Exception as e:
            print(f"Error listing models: {e}")
            return []


# Global instance of the model manager
model_manager = OllamaModelManager()