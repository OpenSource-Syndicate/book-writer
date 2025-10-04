"""
Book Writer System - Response Handler
Handles structured responses from AI models
"""

import json
from typing import Dict, Any, Optional, Union, List
from json import JSONDecodeError


def validate_json_response(response: str, required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate and parse JSON response from AI model.
    
    Args:
        response: Raw response string from the model
        required_fields: List of required fields in the response (optional)
        
    Returns:
        Parsed JSON response as dictionary
    """
    # Clean up the response in case it includes formatting
    cleaned_response = response.strip()
    
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]  # Remove ```json
    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]  # Remove ```
        
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]  # Remove ```
    
    cleaned_response = cleaned_response.strip()
    
    try:
        parsed_response = json.loads(cleaned_response)
        
        # Validate required fields if specified
        if required_fields:
            for field in required_fields:
                if field not in parsed_response:
                    raise ValueError(f"Missing required field in response: {field}")
        
        return parsed_response
    except JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")


def extract_function_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract function calls from AI model response.
    
    Args:
        response: Parsed response from the model
        
    Returns:
        List of function call dictionaries
    """
    function_calls = []
    
    if "tool_calls" in response:
        for tool_call in response["tool_calls"]:
            function_calls.append({
                "name": tool_call["function"]["name"],
                "arguments": json.loads(tool_call["function"]["arguments"])
            })
    elif "function_calls" in response:
        # Alternative format
        for call in response["function_calls"]:
            function_calls.append({
                "name": call["name"],
                "arguments": call["arguments"]
            })
    
    return function_calls


def format_outline_response(response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Format outline response from AI model to ensure consistent structure.
    
    Args:
        response: Raw response from the model (string or dict)
        
    Returns:
        Formatted outline structure
    """
    if isinstance(response, str):
        parsed = validate_json_response(response)
    else:
        parsed = response
    
    # Ensure the response has the expected structure
    if "parts" not in parsed:
        raise ValueError("Response missing 'parts' field for outline structure")
    
    # Validate each part
    for part in parsed["parts"]:
        if "title" not in part:
            raise ValueError("Part missing 'title' field")
        if "chapters" not in part:
            raise ValueError("Part missing 'chapters' field")
        
        # Validate each chapter
        for chapter in part["chapters"]:
            if "title" not in chapter:
                raise ValueError("Chapter missing 'title' field")
            if "subtopics" not in chapter:
                raise ValueError("Chapter missing 'subtopics' field")
            
            # Validate each subtopic
            for subtopic in chapter["subtopics"]:
                if "title" not in subtopic:
                    raise ValueError("Subtopic missing 'title' field")
    
    return parsed


def format_classification_response(response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Format classification response from AI model to ensure consistent structure.
    
    Args:
        response: Raw response from the model (string or dict)
        
    Returns:
        Formatted classification structure
    """
    if isinstance(response, str):
        parsed = validate_json_response(response, ["chapter_id", "subtopic_id"])
    else:
        parsed = response
    
    # Ensure required fields exist
    result = {
        "chapter_id": parsed.get("chapter_id"),
        "chapter_title": parsed.get("chapter_title"),
        "subtopic_id": parsed.get("subtopic_id"),
        "subtopic_title": parsed.get("subtopic_title"),
        "confidence": parsed.get("confidence", 0.5)
    }
    
    return result