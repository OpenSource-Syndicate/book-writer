"""
Book Writer System - Content Expansion Module
Handles expansion of raw notes into polished text using Ollama models
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from book_writer.note_processor import NoteProcessor, ContentManager
from book_writer.model_manager import model_manager
from book_writer.nlp_classifier import create_nlp_classifier


class ContentExpander:
    """Class for expanding raw notes into polished text."""
    
    def __init__(self, project_path: Union[str, Path], note_processor: NoteProcessor, content_manager: ContentManager):
        """Initialize a new content expander.
        
        Args:
            project_path: The path to the project directory
            note_processor: A NoteProcessor instance for retrieving notes
            content_manager: A ContentManager instance for storing expanded content
        """
        self.project_path = Path(project_path)
        self.note_processor = note_processor
        self.content_manager = content_manager
        
        # Use the model manager for Ollama models
        self.model_manager = model_manager
        
        # Initialize the NLP classifier for reliable content classification
        self.nlp_classifier = create_nlp_classifier()
        
        # Check if the model service is accessible
        if not self.model_manager.health_check():
            print("Warning: Ollama service not accessible. Content expansion may not work.")
    
    def expand_note(self, note_id: str, style: str = "academic", stream_callback: Optional[callable] = None) -> Tuple[str, Dict]:
        """Expand a raw note into polished text.
        
        Args:
            note_id: The ID of the note to expand
            style: The writing style to use (academic, narrative, technical, etc.)
            stream_callback: Optional callback to handle streaming response
            
        Returns:
            A tuple of (expanded_text, metadata)
        """
        # Get the note from ChromaDB
        results = self.note_processor.notes_collection.get(
            ids=[note_id],
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            raise ValueError(f"Note with ID {note_id} not found")
        
        note_text = results["documents"][0]
        metadata = results["metadatas"][0]
        
        # Expand the note using the model
        expanded_text = self._generate_expanded_text(note_text, style, stream_callback)
        
        return expanded_text, metadata
    
    def _generate_expanded_text(self, note_text: str, style: str, stream_callback: Optional[callable] = None) -> str:
        """Generate expanded text from a raw note.
        
        Args:
            note_text: The raw note text
            style: The writing style to use
            stream_callback: Optional callback to handle streaming response
            
        Returns:
            The expanded text
        """
        # Create a prompt for the model
        prompt = self._create_expansion_prompt(note_text, style)
        
        # Generate expanded text using the model manager with the content expansion model
        try:
            if stream_callback:
                # Use streaming if callback is provided
                expanded_text = self.model_manager.generate_response_stream(
                    prompt=prompt,
                    task="content_expansion",
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.9,
                    callback=stream_callback
                )
            else:
                # Use regular response if no callback is provided
                expanded_text = self.model_manager.generate_response(
                    prompt=prompt,
                    task="content_expansion",
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.9
                )
            
            # If there's an error in the response, return the original text with a disclaimer
            if isinstance(expanded_text, str) and expanded_text.startswith("Error"):
                print(f"Error during content expansion: {expanded_text}")
                return f"{note_text}\n\n[Note: Error during content expansion: {expanded_text}]"
            
            return expanded_text
        except Exception as e:
            print(f"Error generating expanded text: {e}")
            error_msg = f"{note_text}\n\n[Note: Error during content expansion: {str(e)}]"
            if stream_callback:
                stream_callback(error_msg)
            return error_msg
    
    def _create_expansion_prompt(self, note_text: str, style: str) -> str:
        """Create a prompt for the model to expand a note.
        
        Args:
            note_text: The raw note text
            style: The writing style to use
            
        Returns:
            The prompt for the model
        """
        style_descriptions = {
            "academic": "formal, well-structured, with clear arguments and evidence",
            "narrative": "engaging, story-like, with vivid descriptions and character development",
            "technical": "precise, detailed, with technical terminology and step-by-step explanations",
            "conversational": "casual, approachable, with a friendly tone and relatable examples"
        }
        
        style_desc = style_descriptions.get(style.lower(), "clear, well-structured, and engaging")
        
        prompt = f"""
Task: Expand the following raw note into a polished, well-developed piece of text.
Style: {style_desc}

Raw Note:
{note_text}

Expanded Text:
"""
        
        return prompt.strip()
    
    def classify_content(self, content: str, outline_data: Dict) -> Dict:
        """Classify content into appropriate chapters and subtopics using reliable NLP algorithms.
        
        Args:
            content: The content to classify
            outline_data: The book outline data
            
        Returns:
            A dictionary with classification results
        """
        print(f"NLP Classifying content snippet: {content[:100]}...")
        
        try:
            # Use the reliable NLP classifier instead of unreliable AI
            nlp_result = self.nlp_classifier.classify_content(content, outline_data)
            print(f"NLP classification result: {nlp_result}")
            
            # Log cache statistics periodically
            cache_stats = self.nlp_classifier.get_cache_stats()
            total_requests = cache_stats["hits"] + cache_stats["misses"]
            if total_requests > 0 and total_requests % 10 == 0:  # Log every 10 requests
                hit_rate = cache_stats["hits"] / total_requests * 100
                print(f"NLP Classifier Cache Stats: Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}, Hit Rate: {hit_rate:.1f}%")
            
            return nlp_result
        except Exception as e:
            error_msg = f"Error during NLP content classification: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            # Fallback to keyword matching if the NLP approach fails
            fallback_result = self._classify_content_fallback(content, outline_data)
            print(f"Fallback result after error: {fallback_result}")
            return fallback_result
    
    def _classify_content_fallback(self, content: str, outline_data: Dict) -> Dict:
        """Fallback classification method using keyword matching.
        
        Args:
            content: The content to classify
            outline_data: The book outline data
            
        Returns:
            A dictionary with classification results
        """
        # Extract all chapter and subtopic titles
        chapters = []
        subtopics = []
        
        for part in outline_data.get("parts", []):
            for chapter in part.get("chapters", []):
                chapters.append({
                    "id": chapter["id"],
                    "title": chapter["title"],
                    "part_id": part["id"]
                })
                
                for subtopic in chapter.get("subtopics", []):
                    subtopics.append({
                        "id": subtopic["id"],
                        "title": subtopic["title"],
                        "chapter_id": chapter["id"]
                    })
        
        # Find best matching chapter and subtopic based on simple keyword matching
        print(f"Using fallback classification for content: {content[:50]}...")
        print(f"Found {len(chapters)} chapters and {len(subtopics)} subtopics for matching")
        
        best_chapter = None
        best_chapter_score = 0
        
        for chapter in chapters:
            # Count how many words from the chapter title appear in the content
            title_words = set(chapter["title"].lower().split())
            content_words = set(content.lower().split())
            score = len(title_words.intersection(content_words))
            
            if score > best_chapter_score:
                best_chapter_score = score
                best_chapter = chapter
        
        print(f"Best chapter match: {best_chapter} (score: {best_chapter_score})")
        
        best_subtopic = None
        best_subtopic_score = 0
        
        if best_chapter:
            # Only consider subtopics from the best matching chapter
            chapter_subtopics = [s for s in subtopics if s["chapter_id"] == best_chapter["id"]]
            print(f"Found {len(chapter_subtopics)} subtopics for chapter {best_chapter['id']}")
            
            for subtopic in chapter_subtopics:
                # Count how many words from the subtopic title appear in the content
                title_words = set(subtopic["title"].lower().split())
                content_words = set(content.lower().split())
                score = len(title_words.intersection(content_words))
                
                if score > best_subtopic_score:
                    best_subtopic_score = score
                    best_subtopic = subtopic
        
        print(f"Best subtopic match: {best_subtopic} (score: {best_subtopic_score})")
        
        result = {
            "chapter": best_chapter,
            "subtopic": best_subtopic,
            "chapter_score": best_chapter_score,
            "subtopic_score": best_subtopic_score
        }
        
        print(f"Fallback classification result: {result}")
        return result
    
    def process_and_store_note(self, note_id: str, style: str = "academic", outline_data: Dict = None) -> str:
        """Process a note, expand it, classify it, and store the expanded content.
        
        Args:
            note_id: The ID of the note to process
            style: The writing style to use
            outline_data: The book outline data for classification
            
        Returns:
            The ID of the stored content
        """
        # Expand the note
        expanded_text, note_metadata = self.expand_note(note_id, style)
        
        # Classify the content if outline data is provided
        if outline_data:
            classification = self.classify_content(expanded_text, outline_data)
            chapter_id = classification["chapter"]["id"] if classification["chapter"] else None
            subtopic_id = classification["subtopic"]["id"] if classification["subtopic"] else None
        else:
            # Use existing classification from note metadata if available
            chapter_id = note_metadata.get("chapter_id")
            subtopic_id = note_metadata.get("subtopic_id")
        
        # If no classification is available, raise an error
        if not chapter_id or not subtopic_id:
            raise ValueError("Could not classify content and no existing classification found")
        
        # Create a title for the content
        title = f"Expanded note from {note_metadata.get('source', 'unknown')}"
        
        # Store the expanded content
        content_id = self.content_manager.store_content(
            content=expanded_text,
            title=title,
            chapter_id=chapter_id,
            subtopic_id=subtopic_id,
            source_note_ids=[note_id]
        )
        
        return content_id