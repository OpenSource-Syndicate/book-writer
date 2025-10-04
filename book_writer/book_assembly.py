"""
Book Writer System - Book Assembly Module
Handles assembly of book content into a complete manuscript
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import markdown
from tqdm import tqdm

from book_writer.note_processor import ContentManager
from book_writer.outline import BookOutline


class BookAssembler:
    """Class for assembling book content into a complete manuscript."""
    
    def __init__(self, project_path: Union[str, Path], content_manager: ContentManager):
        """Initialize a new book assembler.
        
        Args:
            project_path: The path to the project directory
            content_manager: A ContentManager instance for retrieving content
        """
        self.project_path = Path(project_path)
        self.content_manager = content_manager
        self.output_dir = self.project_path / "output"
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def build_book(self, outline: BookOutline, output_format: str = "markdown") -> Path:
        """Build a complete book manuscript from the outline and content.
        
        Args:
            outline: The book outline
            output_format: The output format (markdown, html)
            
        Returns:
            The path to the built book
        """
        print(f"Building book: {outline.title}")
        
        # Create a timestamp for the output file
        timestamp = int(time.time())
        
        # Create the output file path
        if output_format.lower() == "markdown":
            output_file = self.output_dir / f"{outline.title.replace(' ', '_').lower()}_{timestamp}.md"
        elif output_format.lower() == "html":
            output_file = self.output_dir / f"{outline.title.replace(' ', '_').lower()}_{timestamp}.html"
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Build the book content
        content = self._build_content(outline)
        
        # Write the content to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            if output_format.lower() == "html":
                html_content = markdown.markdown(content)
                f.write(html_content)
            else:
                f.write(content)
        
        print(f"Book built successfully: {output_file}")
        return output_file
    
    def _build_content(self, outline: BookOutline) -> str:
        """Build the book content from the outline.
        
        Args:
            outline: The book outline
            
        Returns:
            The complete book content as a string
        """
        content = []
        
        # Add title page
        content.append(f"# {outline.title}\n")
        if outline.author:
            content.append(f"## By {outline.author}\n")
        if outline.description:
            content.append(f"*{outline.description}*\n")
        
        content.append("\n---\n\n")
        
        # Add table of contents
        content.append("# Table of Contents\n")
        
        for i, part in enumerate(outline.parts, 1):
            content.append(f"{i}. {part['title']}\n")
            
            for j, chapter in enumerate(part['chapters'], 1):
                content.append(f"   {i}.{j}. {chapter['title']}\n")
        
        content.append("\n---\n\n")
        
        # Add parts, chapters, and content
        for part_index, part in enumerate(tqdm(outline.parts, desc="Processing parts"), 1):
            content.append(f"# Part {part_index}: {part['title']}\n")
            
            if part.get('description'):
                content.append(f"*{part['description']}*\n\n")
            
            for chapter_index, chapter in enumerate(part['chapters'], 1):
                content.append(f"## Chapter {part_index}.{chapter_index}: {chapter['title']}\n")
                
                if chapter.get('description'):
                    content.append(f"*{chapter['description']}*\n\n")
                
                for subtopic in chapter['subtopics']:
                    content.append(f"### {subtopic['title']}\n")
                    
                    if subtopic.get('description'):
                        content.append(f"*{subtopic['description']}*\n\n")
                    
                    # Get content for this subtopic
                    subtopic_content = self.content_manager.retrieve_content_by_subtopic(subtopic['id'])
                    
                    if subtopic_content:
                        # Sort content by metadata timestamp if available
                        subtopic_content.sort(
                            key=lambda x: x['metadata'].get('timestamp', 0)
                        )
                        
                        # Add all content for this subtopic
                        for content_item in subtopic_content:
                            content.append(f"{content_item['content']}\n\n")
                    else:
                        content.append("*No content available for this subtopic.*\n\n")
            
            content.append("\n---\n\n")
        
        return "\n".join(content)
    
    def update_content(self, outline: BookOutline, subtopic_id: str, content_id: str, new_content: str) -> None:
        """Update content in the book.
        
        Args:
            outline: The book outline
            subtopic_id: The ID of the subtopic
            content_id: The ID of the content to update
            new_content: The new content
        """
        # Find the subtopic in the outline
        subtopic = None
        for part in outline.parts:
            for chapter in part['chapters']:
                for st in chapter['subtopics']:
                    if st['id'] == subtopic_id:
                        subtopic = st
                        break
                if subtopic:
                    break
            if subtopic:
                break
        
        if not subtopic:
            raise ValueError(f"Subtopic with ID {subtopic_id} not found in outline")
        
        # Get the content from ChromaDB
        results = self.content_manager.note_processor.content_collection.get(
            ids=[content_id],
            include=["metadatas"]
        )
        
        if not results["ids"]:
            raise ValueError(f"Content with ID {content_id} not found")
        
        # Update metadata
        metadata = results["metadatas"][0]
        
        # Filter out any None values from metadata as ChromaDB doesn't accept them
        filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Store the updated content
        self.content_manager.note_processor.content_collection.update(
            ids=[content_id],
            documents=[new_content],
            metadatas=[filtered_metadata]
        )
        
        # Update the file
        file_path = self.content_manager.content_dir / f"{content_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                content_data = json.load(f)
            
            content_data["content"] = new_content
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content_data, f, indent=2, ensure_ascii=False)
        
        print(f"Content updated successfully: {content_id}")
    
    def replace_content(self, outline: BookOutline, subtopic_id: str, old_content_id: str, new_content: str, title: str = None) -> str:
        """Replace content in the book with new content.
        
        Args:
            outline: The book outline
            subtopic_id: The ID of the subtopic
            old_content_id: The ID of the content to replace
            new_content: The new content
            title: The title for the new content
            
        Returns:
            The ID of the new content
        """
        # Find the subtopic in the outline
        subtopic = None
        chapter_id = None
        for part in outline.parts:
            for chapter in part['chapters']:
                for st in chapter['subtopics']:
                    if st['id'] == subtopic_id:
                        subtopic = st
                        chapter_id = chapter['id']
                        break
                if subtopic:
                    break
            if subtopic:
                break
        
        if not subtopic:
            raise ValueError(f"Subtopic with ID {subtopic_id} not found in outline")
        
        # Get the old content from ChromaDB
        results = self.content_manager.note_processor.content_collection.get(
            ids=[old_content_id],
            include=["metadatas"]
        )
        
        if not results["ids"]:
            raise ValueError(f"Content with ID {old_content_id} not found")
        
        # Get metadata from old content
        old_metadata = results["metadatas"][0]
        
        # Use the old title if no new title is provided
        if title is None:
            title = old_metadata.get("title", "Replaced content")
        
        # Store the new content
        new_content_id = self.content_manager.store_content(
            content=new_content,
            title=title,
            chapter_id=chapter_id,
            subtopic_id=subtopic_id
        )
        
        # Remove the old content ID from the subtopic's content_ids if it exists
        if 'content_ids' in subtopic and old_content_id in subtopic['content_ids']:
            subtopic['content_ids'].remove(old_content_id)
        
        # Add the new content ID to the subtopic's content_ids
        if 'content_ids' not in subtopic:
            subtopic['content_ids'] = []
        subtopic['content_ids'].append(new_content_id)
        
        print(f"Content replaced successfully: {old_content_id} -> {new_content_id}")
        return new_content_id