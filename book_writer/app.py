"""
Book Writer System - Main Application Interface
Ties all components together into a cohesive application
"""
import cmd
import json
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from book_writer.outline import BookOutline, OutlineManager, create_sample_outline
from book_writer.note_processor import NoteProcessor, ContentManager
from book_writer.content_expansion import ContentExpander
from book_writer.rag_writing import RAGWriter
from book_writer.book_assembly import BookAssembler
from book_writer.export import BookExporter


class BookWriterApp:
    """Main application class for the Book Writer System."""
    
    def __init__(self, project_path: Union[str, Path]):
        """Initialize a new Book Writer application.
        
        Args:
            project_path: The path to the project directory
        """
        self.project_path = Path(project_path)
        self.config_path = self.project_path / "book_config.yaml"
        
        # Load or create configuration
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "project_name": self.project_path.name,
                "created_at": time.time(),
                "current_outline": None
            }
            self._save_config()
        
        # Initialize components
        self.outline_manager = OutlineManager(self.project_path)
        self.note_processor = NoteProcessor(self.project_path)
        self.content_manager = ContentManager(self.project_path, self.note_processor)
        self.content_expander = ContentExpander(self.project_path, self.note_processor, self.content_manager)
        self.rag_writer = RAGWriter(self.project_path, self.note_processor, self.content_manager)
        self.book_assembler = BookAssembler(self.project_path, self.content_manager)
        self.book_exporter = BookExporter(self.project_path)
        
        # Load current outline if available
        self.current_outline = None
        if self.config.get("current_outline"):
            try:
                outline_path = self.project_path / "outlines" / self.config["current_outline"]
                self.current_outline = BookOutline.load(outline_path)
                print(f"Loaded outline: {self.current_outline.title}")
            except Exception as e:
                print(f"Error loading outline: {e}")
    
    def _save_config(self):
        """Save the configuration to disk."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    @classmethod
    def create_project(cls, project_path: Union[str, Path]) -> "BookWriterApp":
        """Create a new book project.
        
        Args:
            project_path: The path to create the project at
            
        Returns:
            A new BookWriterApp instance
        """
        project_path = Path(project_path)
        
        # Create project directories
        project_path.mkdir(exist_ok=True, parents=True)
        (project_path / "outlines").mkdir(exist_ok=True)
        (project_path / "notes").mkdir(exist_ok=True)
        (project_path / "content").mkdir(exist_ok=True)
        (project_path / "output").mkdir(exist_ok=True)
        (project_path / "exports").mkdir(exist_ok=True)
        (project_path / "db").mkdir(exist_ok=True)
        
        # Create a sample outline
        app = cls(project_path)
        sample_outline = create_sample_outline()
        outline_path = app.outline_manager.save_outline(sample_outline)
        
        # Set as current outline
        app.config["current_outline"] = outline_path.name
        app.current_outline = sample_outline
        app._save_config()
        
        return app
    
    def create_outline(self, title: str, author: str = "", description: str = "") -> BookOutline:
        """Create a new book outline.
        
        Args:
            title: The title of the book
            author: The author of the book
            description: A brief description of the book
            
        Returns:
            A new BookOutline instance
        """
        outline = self.outline_manager.create_outline(title, author, description)
        outline_path = self.outline_manager.save_outline(outline)
        
        # Set as current outline
        self.config["current_outline"] = outline_path.name
        self.current_outline = outline
        self._save_config()
        
        return outline
    
    def load_outline(self, file_name: str) -> BookOutline:
        """Load an outline from a file.
        
        Args:
            file_name: The name of the outline file
            
        Returns:
            A BookOutline instance
        """
        outline = self.outline_manager.load_outline(file_name)
        
        # Set as current outline
        self.config["current_outline"] = file_name
        self.current_outline = outline
        self._save_config()
        
        return outline
    
    def process_note(self, text: str, source: str = "conversation", potential_topics: List[str] = None) -> str:
        """Process a new note.
        
        Args:
            text: The text of the note
            source: The source of the note
            potential_topics: List of potential topics for the note
            
        Returns:
            The ID of the processed note
        """
        return self.note_processor.process_note(text, source, potential_topics)
    
    def expand_note(self, note_id: str, style: str = "academic") -> str:
        """Expand a note into polished text.
        
        Args:
            note_id: The ID of the note to expand
            style: The writing style to use
            
        Returns:
            The ID of the expanded content
        """
        if not self.current_outline:
            raise ValueError("No outline loaded. Please load or create an outline first.")
        
        expanded_text, metadata = self.content_expander.expand_note(note_id, style)
        
        # Classify the content
        classification = self.content_expander.classify_content(expanded_text, self.current_outline.to_dict())
        
        if not classification["chapter"] or not classification["subtopic"]:
            raise ValueError("Could not classify content. Please specify chapter and subtopic manually.")
        
        # Store the expanded content
        content_id = self.content_manager.store_content(
            content=expanded_text,
            title=f"Expanded note from {metadata.get('source', 'unknown')}",
            chapter_id=classification["chapter"]["id"],
            subtopic_id=classification["subtopic"]["id"],
            source_note_ids=[note_id]
        )
        
        return content_id
    
    def generate_content(self, prompt: str, chapter_id: str, subtopic_id: str) -> str:
        """Generate new content using RAG.
        
        Args:
            prompt: The writing prompt
            chapter_id: The ID of the chapter
            subtopic_id: The ID of the subtopic
            
        Returns:
            The ID of the generated content
        """
        # Retrieve context
        context = self.rag_writer.retrieve_context(prompt)
        
        # Generate content
        generated_text = self.rag_writer.generate_with_context(prompt, context)
        
        # Store the generated content
        content_id = self.content_manager.store_content(
            content=generated_text,
            title=prompt[:50] + "..." if len(prompt) > 50 else prompt,
            chapter_id=chapter_id,
            subtopic_id=subtopic_id
        )
        
        return content_id
    
    def build_book(self) -> Path:
        """Build the current book.
        
        Returns:
            The path to the built book
        """
        if not self.current_outline:
            raise ValueError("No outline loaded. Please load or create an outline first.")
        
        return self.book_assembler.build_book(self.current_outline)
    
    def export_book(self, format: str) -> Path:
        """Export the current book to the specified format.
        
        Args:
            format: The format to export to ('pdf' or 'epub')
            
        Returns:
            The path to the exported book
        """
        # Build the book first
        markdown_path = self.build_book()
        
        # Export to the specified format
        if format.lower() == "pdf":
            return self.book_exporter.export_to_pdf(markdown_path)
        elif format.lower() == "epub":
            return self.book_exporter.export_to_epub(markdown_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def start_interactive_mode(self):
        """Start the interactive command-line interface."""
        BookWriterShell(self).cmdloop()


class BookWriterShell(cmd.Cmd):
    """Interactive command-line interface for the Book Writer System."""
    
    intro = "Welcome to the Book Writer System. Type 'help' for a list of commands."
    prompt = "book-writer> "
    
    def __init__(self, app: BookWriterApp):
        """Initialize a new command-line interface.
        
        Args:
            app: The BookWriterApp instance
        """
        super().__init__()
        self.app = app
    
    def do_create_outline(self, arg):
        """Create a new book outline.
        
        Usage: create_outline <title> [author] [description]
        """
        args = arg.split(maxsplit=2)
        if not args:
            print("Error: Title is required")
            return
        
        title = args[0]
        author = args[1] if len(args) > 1 else ""
        description = args[2] if len(args) > 2 else ""
        
        outline = self.app.create_outline(title, author, description)
        print(f"Created outline: {outline.title}")
    
    def do_list_outlines(self, arg):
        """List all available outlines.
        
        Usage: list_outlines
        """
        outlines = self.app.outline_manager.list_outlines()
        if not outlines:
            print("No outlines found")
            return
        
        print("Available outlines:")
        for i, outline in enumerate(outlines, 1):
            print(f"{i}. {outline}")
    
    def do_load_outline(self, arg):
        """Load an outline from a file.
        
        Usage: load_outline <file_name>
        """
        if not arg:
            print("Error: File name is required")
            return
        
        try:
            outline = self.app.load_outline(arg)
            print(f"Loaded outline: {outline.title}")
        except Exception as e:
            print(f"Error loading outline: {e}")
    
    def do_add_part(self, arg):
        """Add a new part to the current outline.
        
        Usage: add_part <title> [description]
        """
        if not self.app.current_outline:
            print("Error: No outline loaded")
            return
        
        args = arg.split(maxsplit=1)
        if not args:
            print("Error: Title is required")
            return
        
        title = args[0]
        description = args[1] if len(args) > 1 else ""
        
        part = self.app.current_outline.add_part(title, description)
        print(f"Added part: {title} (ID: {part['id']})")
        
        # Save the updated outline
        self.app.outline_manager.save_outline(self.app.current_outline)
    
    def do_add_chapter(self, arg):
        """Add a new chapter to a part.
        
        Usage: add_chapter <part_id> <title> [description]
        """
        if not self.app.current_outline:
            print("Error: No outline loaded")
            return
        
        args = arg.split(maxsplit=2)
        if len(args) < 2:
            print("Error: Part ID and title are required")
            return
        
        part_id = args[0]
        title = args[1]
        description = args[2] if len(args) > 2 else ""
        
        try:
            chapter = self.app.current_outline.add_chapter(part_id, title, description)
            print(f"Added chapter: {title} (ID: {chapter['id']})")
            
            # Save the updated outline
            self.app.outline_manager.save_outline(self.app.current_outline)
        except ValueError as e:
            print(f"Error: {e}")
    
    def do_add_subtopic(self, arg):
        """Add a new subtopic to a chapter.
        
        Usage: add_subtopic <chapter_id> <title> [description]
        """
        if not self.app.current_outline:
            print("Error: No outline loaded")
            return
        
        args = arg.split(maxsplit=2)
        if len(args) < 2:
            print("Error: Chapter ID and title are required")
            return
        
        chapter_id = args[0]
        title = args[1]
        description = args[2] if len(args) > 2 else ""
        
        try:
            subtopic = self.app.current_outline.add_subtopic(chapter_id, title, description)
            print(f"Added subtopic: {title} (ID: {subtopic['id']})")
            
            # Save the updated outline
            self.app.outline_manager.save_outline(self.app.current_outline)
        except ValueError as e:
            print(f"Error: {e}")
    
    def do_add_note(self, arg):
        """Add a new note.
        
        Usage: add_note <text> [source] [topic1,topic2,...]
        """
        args = arg.split(maxsplit=2)
        if not args:
            print("Error: Text is required")
            return
        
        text = args[0]
        source = args[1] if len(args) > 1 else "conversation"
        potential_topics = args[2].split(",") if len(args) > 2 else None
        
        note_id = self.app.process_note(text, source, potential_topics)
        print(f"Added note: {note_id}")
    
    def do_expand_note(self, arg):
        """Expand a note into polished text.
        
        Usage: expand_note <note_id> [style]
        """
        args = arg.split()
        if not args:
            print("Error: Note ID is required")
            return
        
        note_id = args[0]
        style = args[1] if len(args) > 1 else "academic"
        
        try:
            content_id = self.app.expand_note(note_id, style)
            print(f"Expanded note: {content_id}")
        except Exception as e:
            print(f"Error expanding note: {e}")
    
    def do_generate_content(self, arg):
        """Generate new content using RAG.
        
        Usage: generate_content <chapter_id> <subtopic_id> <prompt>
        """
        args = arg.split(maxsplit=2)
        if len(args) < 3:
            print("Error: Chapter ID, subtopic ID, and prompt are required")
            return
        
        chapter_id = args[0]
        subtopic_id = args[1]
        prompt = args[2]
        
        try:
            content_id = self.app.generate_content(prompt, chapter_id, subtopic_id)
            print(f"Generated content: {content_id}")
        except Exception as e:
            print(f"Error generating content: {e}")
    
    def do_build_book(self, arg):
        """Build the current book.
        
        Usage: build_book
        """
        try:
            output_path = self.app.build_book()
            print(f"Book built successfully: {output_path}")
        except Exception as e:
            print(f"Error building book: {e}")
    
    def do_export_book(self, arg):
        """Export the current book to the specified format.
        
        Usage: export_book <format>
        """
        if not arg:
            print("Error: Format is required (pdf or epub)")
            return
        
        format = arg.lower()
        if format not in ["pdf", "epub"]:
            print("Error: Unsupported format. Use 'pdf' or 'epub'")
            return
        
        try:
            output_path = self.app.export_book(format)
            print(f"Book exported successfully: {output_path}")
        except Exception as e:
            print(f"Error exporting book: {e}")
    
    def do_exit(self, arg):
        """Exit the application.
        
        Usage: exit
        """
        print("Goodbye!")
        return True
    
    def do_quit(self, arg):
        """Exit the application.
        
        Usage: quit
        """
        return self.do_exit(arg)