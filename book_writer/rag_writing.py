"""
Book Writer System - Retrieval-Augmented Writing Module
Implements advanced RAG loop for context-aware writing using Ollama models
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from book_writer.note_processor import NoteProcessor, ContentManager
from book_writer.model_manager import model_manager
from book_writer.tool_registry import tool_registry


class RAGWriter:
    """Class for retrieval-augmented writing."""
    
    def __init__(self, project_path: Union[str, Path], note_processor: NoteProcessor, content_manager: ContentManager):
        """Initialize a new RAG writer.
        
        Args:
            project_path: The path to the project directory
            note_processor: A NoteProcessor instance for retrieving notes
            content_manager: A ContentManager instance for retrieving content
        """
        self.project_path = Path(project_path)
        self.note_processor = note_processor
        self.content_manager = content_manager
        
        # Use the model manager for Ollama models
        self.model_manager = model_manager
        
        # Check if the model service is accessible
        if not self.model_manager.health_check():
            print("Warning: Ollama service not accessible. RAG functionality will be limited")
        
        # Register tools with the tool registry
        tool_registry.register_default_tools(
            note_processor=self.note_processor,
            content_manager=self.content_manager
        )
    
    def retrieve_context(self, query: str, n_notes: int = 3, n_content: int = 3) -> Dict:
        """Retrieve relevant context for writing.
        
        Args:
            query: The query text
            n_notes: The number of notes to retrieve
            n_content: The number of content items to retrieve
            
        Returns:
            A dictionary with retrieved notes and content
        """
        # Retrieve similar notes
        similar_notes = self.note_processor.retrieve_similar_notes(query, n_results=n_notes)
        
        # Retrieve similar content
        similar_content = self.content_manager.retrieve_similar_content(query, n_results=n_content)
        
        return {
            "notes": similar_notes,
            "content": similar_content
        }
    
    def generate_with_context(self, prompt: str, context: Dict, max_length: int = 1000, stream_callback: Optional[callable] = None) -> str:
        """Generate text with retrieved context.
        
        Args:
            prompt: The writing prompt
            context: The retrieved context
            max_length: The maximum length of the generated text (for compatibility, not used in Ollama)
            stream_callback: Optional callback to handle streaming response
            
        Returns:
            The generated text
        """
        # Format the context
        formatted_context = self._format_context(context)
        
        # Create a prompt with context
        full_prompt = f"""
Context Information:
{formatted_context}

Writing Task:
{prompt}

Generated Text:
"""
        
        # Generate text using the model manager with the content expansion model
        try:
            if stream_callback:
                generated_text = self.model_manager.generate_response_stream(
                    prompt=full_prompt,
                    task="content_expansion",  # Use content expansion model for RAG generation
                    temperature=0.7,
                    max_tokens=max_length,
                    top_p=0.9,
                    callback=stream_callback
                )
            else:
                generated_text = self.model_manager.generate_response(
                    prompt=full_prompt,
                    task="content_expansion",  # Use content expansion model for RAG generation
                    temperature=0.7,
                    max_tokens=max_length,
                    top_p=0.9
                )
            
            # Check if there's an error in the response
            if isinstance(generated_text, str) and generated_text.startswith("Error"):
                print(f"Error during RAG generation: {generated_text}")
                return f"[RAG generation error: {generated_text}]\n\nPrompt: {prompt}"
            
            return generated_text
        except Exception as e:
            print(f"Error generating text: {e}")
            error_msg = f"[Error during RAG generation]\n\nPrompt: {prompt}"
            if stream_callback:
                stream_callback(error_msg)
            return error_msg
    
    def _format_context(self, context: Dict) -> str:
        """Format retrieved context for the model.
        
        Args:
            context: The retrieved context
            
        Returns:
            The formatted context as a string
        """
        formatted_text = ""
        
        # Format notes
        if context.get("notes"):
            formatted_text += "RELEVANT NOTES:\n"
            for i, note in enumerate(context["notes"], 1):
                formatted_text += f"{i}. {note['text'][:200]}...\n\n"
        
        # Format content
        if context.get("content"):
            formatted_text += "RELEVANT CONTENT:\n"
            for i, content in enumerate(context["content"], 1):
                content_text = content['content']
                # Truncate long content
                if len(content_text) > 300:
                    content_text = content_text[:300] + "..."
                formatted_text += f"{i}. {content_text}\n\n"
        
        return formatted_text
    
    def rag_loop(self, initial_prompt: str, iterations: int = 3, feedback: str = None) -> str:
        """Run a RAG loop for iterative writing.
        
        Args:
            initial_prompt: The initial writing prompt
            iterations: The number of iterations to run
            feedback: Optional feedback to incorporate
            
        Returns:
            The final generated text
        """
        current_text = ""
        
        for i in range(iterations):
            print(f"RAG iteration {i+1}/{iterations}")
            
            # Create a prompt for this iteration
            if i == 0:
                prompt = initial_prompt
            else:
                prompt = f"""
Continue developing the following text. Maintain consistency and coherence.

Current Text:
{current_text}

{feedback if feedback else ''}
"""
            
            # Retrieve context based on the current state
            context_query = prompt
            if current_text:
                context_query = current_text + "\n" + prompt
            
            context = self.retrieve_context(context_query)
            
            # Generate new text
            new_text = self.generate_with_context(prompt, context)
            
            # Update the current text
            if i == 0:
                current_text = new_text
            else:
                # Append new text, avoiding repetition
                current_text = self._merge_text(current_text, new_text)
        
        return current_text
    
    def _merge_text(self, current_text: str, new_text: str) -> str:
        """Merge current text with new text, avoiding repetition.
        
        Args:
            current_text: The current text
            new_text: The new text to merge
            
        Returns:
            The merged text
        """
        # Simple approach: if the new text starts with content from the end of the current text,
        # find the overlap and only append the non-overlapping part
        
        # Check for overlap (at least 20 characters)
        min_overlap = 20
        max_overlap = min(100, len(current_text), len(new_text))
        
        best_overlap = 0
        overlap_len = 0
        
        for i in range(min_overlap, max_overlap + 1):
            current_end = current_text[-i:]
            new_start = new_text[:i]
            
            if current_end == new_start:
                overlap_len = i
                break
        
        if overlap_len > 0:
            # Append only the non-overlapping part
            return current_text + new_text[overlap_len:]
        else:
            # No significant overlap found, just append with a separator
            return current_text + "\n\n" + new_text
    
    def store_generated_content(self, text: str, title: str, chapter_id: str, subtopic_id: str) -> str:
        """Store generated content.
        
        Args:
            text: The generated text
            title: The title of the content
            chapter_id: The ID of the chapter
            subtopic_id: The ID of the subtopic
            
        Returns:
            The ID of the stored content
        """
        # Store the content
        content_id = self.content_manager.store_content(
            content=text,
            title=title,
            chapter_id=chapter_id,
            subtopic_id=subtopic_id
        )
        
        return content_id