"""
Book Writer System - Tool Registry
Manages available tools/functions that can be called by AI models
"""

from typing import Dict, Callable, Any, List, Optional
import inspect
import json


class ToolRegistry:
    """Registry for functions that can be called by AI models."""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.functions: Dict[str, Callable] = {}
    
    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], function: Callable):
        """Register a function as an available tool.
        
        Args:
            name: The name of the tool
            description: Description of what the tool does
            parameters: JSON schema for the function parameters
            function: The actual function to call
        """
        self.tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        self.functions[name] = function
    
    def get_tool_spec(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the specification for a registered tool.
        
        Args:
            name: The name of the tool
            
        Returns:
            Tool specification or None if not found
        """
        return self.tools.get(name)
    
    def get_all_tool_specs(self) -> List[Dict[str, Any]]:
        """Get specifications for all registered tools.
        
        Returns:
            List of tool specifications
        """
        return list(self.tools.values())
    
    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered tool with the given arguments.
        
        Args:
            name: The name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            Result of the tool execution
        """
        if name not in self.functions:
            return {"error": f"Tool '{name}' not found", "available_tools": list(self.functions.keys())}
        
        try:
            # Call the function with the provided arguments
            function = self.functions[name]
            sig = inspect.signature(function)
            
            # Filter arguments to only include those that the function accepts
            valid_args = {}
            for param_name in sig.parameters:
                if param_name in arguments:
                    valid_args[param_name] = arguments[param_name]
            
            result = function(**valid_args)
            return result
        except Exception as e:
            return {"error": f"Error executing tool '{name}': {str(e)}"}
    
    def register_default_tools(self, note_processor=None, content_manager=None, outline_manager=None):
        """Register default tools for the book writing system.
        
        Args:
            note_processor: NoteProcessor instance
            content_manager: ContentManager instance
            outline_manager: OutlineManager instance
        """
        # Tool to search notes
        def search_notes(query: str, n_results: int = 3):
            """Search for notes based on a query."""
            if note_processor is None:
                return {"error": "NoteProcessor not available"}
            try:
                results = note_processor.retrieve_similar_notes(query, n_results)
                return {
                    "notes_found": len(results),
                    "notes": [{"id": r['id'], "text": r['text'][:200] + "..." if len(r['text']) > 200 else r['text']} for r in results]
                }
            except Exception as e:
                return {"error": f"Error searching notes: {str(e)}"}
        
        self.register_tool(
            name="search_notes",
            description="Search for notes based on a query",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "n_results": {"type": "integer", "description": "Number of results to return", "default": 3}
                },
                "required": ["query"]
            },
            function=search_notes
        )
        
        # Tool to classify content
        def classify_content(content: str, outline_data: Dict[str, Any]):
            """Classify content into appropriate chapter and subtopic."""
            if content_manager is None:
                return {"error": "ContentManager not available"}
            try:
                # This would typically use a content expander with classification
                # For now, return a placeholder response
                return {
                    "suggested_chapter": "General",
                    "suggested_subtopic": "Uncategorized",
                    "confidence": 0.5
                }
            except Exception as e:
                return {"error": f"Error classifying content: {str(e)}"}
        
        self.register_tool(
            name="classify_content",
            description="Classify content into appropriate chapter and subtopic",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The content to classify"},
                    "outline_data": {"type": "object", "description": "The book outline data structure"}
                },
                "required": ["content", "outline_data"]
            },
            function=classify_content
        )
        
        # Tool to expand a note
        def expand_note(note_id: str, style: str = "academic"):
            """Expand a note into detailed content."""
            if note_processor is None or content_manager is None:
                return {"error": "NoteProcessor or ContentManager not available"}
            try:
                # This would typically use the ContentExpander
                # For now, return a placeholder response
                return {
                    "note_id": note_id,
                    "expanded": True,
                    "style": style,
                    "message": f"Note {note_id} expanded in {style} style"
                }
            except Exception as e:
                return {"error": f"Error expanding note: {str(e)}"}
        
        self.register_tool(
            name="expand_note",
            description="Expand a note into detailed content",
            parameters={
                "type": "object",
                "properties": {
                    "note_id": {"type": "string", "description": "The ID of the note to expand"},
                    "style": {"type": "string", "description": "The writing style to use", "default": "academic"}
                },
                "required": ["note_id"]
            },
            function=expand_note
        )
        
        # Tool to validate an outline
        def validate_outline(outline_data: Dict[str, Any]):
            """Validate if an outline is properly structured."""
            try:
                # Check if outline has required structure
                required_keys = ["title", "parts"]
                missing_keys = [key for key in required_keys if key not in outline_data]
                
                issues = []
                if missing_keys:
                    issues.append(f"Missing keys in outline: {missing_keys}")
                
                # Check parts structure
                if "parts" in outline_data:
                    for i, part in enumerate(outline_data["parts"]):
                        if "title" not in part:
                            issues.append(f"Part {i} missing title")
                        if "chapters" not in part:
                            issues.append(f"Part {i} missing chapters")
                        else:
                            for j, chapter in enumerate(part["chapters"]):
                                if "title" not in chapter:
                                    issues.append(f"Chapter {j} in part {i} missing title")
                                if "subtopics" not in chapter:
                                    issues.append(f"Chapter {j} in part {i} missing subtopics")
                
                return {
                    "is_valid": len(issues) == 0,
                    "issues": issues
                }
            except Exception as e:
                return {"error": f"Error validating outline: {str(e)}"}
        
        self.register_tool(
            name="validate_outline",
            description="Validate if an outline is properly structured",
            parameters={
                "type": "object",
                "properties": {
                    "outline_data": {"type": "object", "description": "The outline data to validate"}
                },
                "required": ["outline_data"]
            },
            function=validate_outline
        )


# Global tool registry instance
tool_registry = ToolRegistry()