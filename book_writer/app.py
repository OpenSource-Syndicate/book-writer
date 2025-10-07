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

# Handle the readline issue on Windows before importing other modules
try:
    import readline
    # On Windows, the readline module may not have a backend attribute
    # which causes issues in Python's cmd module
    if not hasattr(readline, 'backend'):
        # If backend attribute doesn't exist, set a default value to avoid AttributeError
        readline.backend = 'builtin'  # or another appropriate default
except ImportError:
    # readline may not be available on all systems
    pass

from book_writer.outline import BookOutline, OutlineManager, create_sample_outline
from book_writer.note_processor import NoteProcessor, ContentManager
from book_writer.content_expansion import ContentExpander
from book_writer.rag_writing import RAGWriter
from book_writer.book_assembly import BookAssembler
from book_writer.export import BookExporter
from book_writer.organized_content import ContentOrganization
from book_writer.relationship_visualizer import RelationshipVisualizer


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
        self.content_organization = ContentOrganization(self.project_path, self.note_processor, self.content_manager)
        self.relationship_visualizer = RelationshipVisualizer(self.project_path, self.content_manager, self.content_organization.organizer)
        
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
    def create_project(cls, project_path: Union[str, Path], book_title: Optional[str] = None, author: str = "AI Assistant", description: str = None, target_pages: Optional[int] = None) -> "BookWriterApp":
        """Create a new book project.
        
        Args:
            project_path: The path to create the project at
            book_title: Optional title for the book (defaults to 'My Book' if not provided)
            author: Optional author name (defaults to 'AI Assistant')
            description: Optional book description (defaults based on title if not provided)
            target_pages: Optional target number of pages for the project
            
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
        
        # Use provided title, or default
        topic = book_title if book_title else "My Book"
        description = description if description else f"An AI-generated guide to {topic}"
        
        # Create sample outline with the specified title/topic
        sample_outline = create_sample_outline(topic=topic, title=topic)
        sample_outline.author = author
        sample_outline.description = description
        
        outline_path = app.outline_manager.save_outline(sample_outline)
        
        # Set as current outline
        app.config["current_outline"] = outline_path.name
        app.current_outline = sample_outline
        
        # Add target pages to configuration if provided
        if target_pages is not None:
            app.config["target_pages"] = target_pages
            # Distribute pages across outline sections
            app._distribute_pages_to_outline(target_pages)
        
        app._save_config()
        
        return app
    
    def _distribute_pages_to_outline(self, total_pages: int) -> None:
        """Distribute target pages across the outline structure."""
        if not self.current_outline or not self.current_outline.parts:
            return
        
        # Calculate total sections to distribute pages
        total_subtopics = 0
        for part in self.current_outline.parts:
            for chapter in part.get("chapters") or []:
                total_subtopics += len(chapter["subtopics"])
        
        if total_subtopics == 0:
            return
        
        # Distribute pages more or less evenly to subtopics
        base_pages_per_subtopic = total_pages // total_subtopics
        remainder_pages = total_pages % total_subtopics
        
        # Assign pages to each subtopic
        remainder_used = 0
        for part in self.current_outline.parts:
            for chapter in part.get("chapters") or []:
                for subtopic in chapter.get("subtopics") or []:
                    # Add an extra page to the first few subtopics to handle the remainder
                    extra_page = 1 if remainder_used < remainder_pages else 0
                    pages_for_subtopic = base_pages_per_subtopic + extra_page
                    # Add page target to the subtopic's metadata
                    if "target_pages" not in subtopic:
                        subtopic["target_pages"] = pages_for_subtopic
                    else:
                        # If already exists, add to it
                        subtopic["target_pages"] += pages_for_subtopic
                    remainder_used += 1
    
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
    
    def organize_content(self, content_ids: List[str]) -> Dict:
        """Organize the book content by clustering it and providing a summary.

        Args:
            content_ids: A list of content IDs to organize.

        Returns:
            A dictionary containing the organization summary.
        """
        if not self.current_outline:
            raise ValueError("No outline loaded. Please load or create an outline first.")
        return self.content_organization.organize_content(self.current_outline, content_ids)

    def get_organization_suggestions(self) -> Dict:
        """Get suggestions for improving the content organization.

        Returns:
            A dictionary containing the organization suggestions.
        """
        if not self.current_outline:
            raise ValueError("No outline loaded. Please load or create an outline first.")
        return self.content_organization.get_organization_suggestions(self.current_outline)
    
    def generate_organization_report(self) -> str:
        """Generate a comprehensive organization report for the current book.
        
        Returns:
            Organization report as string
        """
        if not self.current_outline:
            raise ValueError("No outline loaded. Please load or create an outline first.")
        return self.relationship_visualizer.generate_summary_report(self.current_outline)
    
    def visualize_outline_structure(self) -> str:
        """Generate a text-based visualization of the outline structure.
        
        Returns:
            String representation of the outline structure
        """
        if not self.current_outline:
            raise ValueError("No outline loaded. Please load or create an outline first.")
        return self.relationship_visualizer.visualize_outline_structure(self.current_outline)
    
    def get_writing_progress(self) -> Dict[str, Any]:
        """Get writing progress information including page targets and notes count.
        
        Returns:
            Dictionary containing progress information
        """
        if not self.current_outline:
            raise ValueError("No outline loaded. Please load or create an outline first.")
        
        total_target_pages = self.config.get("target_pages", 0)
        
        # Calculate how many pages have been written
        written_pages = 0
        content_items = self.content_manager.get_all_content()
        for content_id, content_data in content_items.items():
            # Estimate pages based on content length (assuming ~500 words per page)
            word_count = len(content_data.get("content", "").split())
            written_pages += word_count / 500.0
        
        # Count total notes
        all_notes = self.note_processor.notes_collection.get(include=["documents", "metadatas"])
        total_notes = len(all_notes.get("ids", [])) if all_notes and all_notes.get("ids") else 0
        
        # Calculate progress by subtopic
        progress_by_section = []
        for part in self.current_outline.parts or []:
            part_info = {
                "type": "part",
                "id": part["id"],
                "title": part["title"],
                "target_pages": 0,  # Sum of all subtopics in this part
                "written_pages": 0,  # Sum of all content in this part
                "target_notes": 0,   # Not directly used for now
                "written_notes": 0,   # How many notes are assigned to this part
                "chapters": []
            }
            
            for chapter in part.get("chapters") or []:
                chapter_info = {
                    "type": "chapter",
                    "id": chapter["id"],
                    "title": chapter["title"],
                    "target_pages": 0,  # Sum of all subtopics in this chapter
                    "written_pages": 0,  # Sum of all content in this chapter
                    "target_notes": 0,
                    "written_notes": 0,
                    "subtopics": []
                }
                
                for subtopic in chapter.get("subtopics") or []:
                    subtopic_target = subtopic.get("target_pages", 0)
                    subtopic_info = {
                        "type": "subtopic",
                        "id": subtopic["id"],
                        "title": subtopic["title"],
                        "target_pages": subtopic_target,
                        "written_pages": 0,  # To be calculated based on content assigned to this subtopic
                        "target_notes": 0,  # Not directly used for now
                        "written_notes": 0  # To be calculated based on notes assigned to this subtopic
                    }
                    
                    # Update parent targets
                    chapter_info["target_pages"] += subtopic_target
                    part_info["target_pages"] += subtopic_target
                    
                    # Find content assigned to this subtopic
                    for content_id, content_data in content_items.items():
                        if content_data.get("subtopic_id") == subtopic["id"]:
                            word_count = len(content_data.get("content", "").split())
                            subtopic_info["written_pages"] += word_count / 500.0
                            chapter_info["written_pages"] += word_count / 500.0
                            part_info["written_pages"] += word_count / 500.0
                    
                    # Find notes assigned to this subtopic
                    all_note_ids = all_notes.get("ids", []) if all_notes else []
                    all_note_metadatas = all_notes.get("metadatas", []) if all_notes else []
                    for i, note_id in enumerate(all_note_ids):
                        note_metadata = all_note_metadatas[i] if i < len(all_note_metadatas) else {}
                        if note_metadata.get("subtopic_id") == subtopic["id"]:
                            subtopic_info["written_notes"] += 1
                            chapter_info["written_notes"] += 1
                            part_info["written_notes"] += 1
                    
                    chapter_info["subtopics"].append(subtopic_info)
                
                part_info["chapters"].append(chapter_info)
            
            progress_by_section.append(part_info)
        
        return {
            "total_target_pages": total_target_pages,
            "total_written_pages": written_pages,
            "total_notes": total_notes,
            "progress_percentage": (written_pages / total_target_pages * 100) if total_target_pages > 0 else 0,
            "progress_by_section": progress_by_section
        }
    
    def get_gamification_recommendations(self) -> Dict[str, Any]:
        """Get gamification recommendations to encourage writing.
        
        Returns:
            Dictionary containing recommendations and encouragement
        """
        progress = self.get_writing_progress()
        
        recommendations = []
        
        # Check if user is behind on their writing goal
        if progress["progress_percentage"] < 25:
            recommendations.append("You're just getting started! Try to add at least 5 more notes today to build momentum.")
        elif progress["progress_percentage"] < 50:
            recommendations.append("Great progress! You're on your way. Try to add 3-5 more notes to keep the momentum going.")
        elif progress["progress_percentage"] < 75:
            recommendations.append("You're doing well! Consider adding 2-3 more notes to stay on track.")
        elif progress["progress_percentage"] < 100:
            recommendations.append("Almost there! Just a little more to reach your goal. Add a few more notes to finish strong.")
        else:
            recommendations.append("Congratulations! You've reached your target. Consider adding more notes to expand on your content.")
        
        # Compare notes to pages written
        if progress["total_notes"] == 0:
            recommendations.append("You haven't added any notes yet. Try adding your first note in the Writer's Desk tab!")
        elif progress["total_written_pages"] / max(progress["total_notes"], 1) < 0.5:  # Less than 0.5 pages per note on avg
            recommendations.append("Your notes are creating shorter content. Try expanding your notes with more details for richer content.")
        else:
            recommendations.append("Great job! Your notes are generating substantial content.")
        
        # Identify sections that need more attention
        underwritten_sections = []
        for part in progress["progress_by_section"]:
            for chapter in part.get("chapters") or []:
                for subtopic in chapter.get("subtopics") or []:
                    if subtopic["written_notes"] == 0 and subtopic["target_pages"] > 0:
                        underwritten_sections.append(f"{part['title']} > {chapter['title']} > {subtopic['title']}")
        
        if underwritten_sections:
            recommendations.append(f"Consider writing notes for these sections that currently have no content: {', '.join(underwritten_sections[:3])}")  # Limit to first 3 to avoid overwhelming
        
        # Add milestone celebration
        milestones = []
        if progress["progress_percentage"] >= 25 and progress["progress_percentage"] < 50:
            milestones.append("25% Complete Milestone: Well done! You've completed the first quarter of your book!")
        elif progress["progress_percentage"] >= 50 and progress["progress_percentage"] < 75:
            milestones.append("50% Complete Milestone: Halfway there! You're doing great!")
        elif progress["progress_percentage"] >= 75 and progress["progress_percentage"] < 100:
            milestones.append("75% Complete Milestone: So close! You're almost done!")
        elif progress["progress_percentage"] >= 100:
            milestones.append("100% Complete Milestone: Amazing work! You've reached your target!")
        
        return {
            "recommendations": recommendations,
            "milestones": milestones,
            "current_progress": progress
        }
    
    def create_content_relationship(self, source_id: str, target_id: str, relationship_type: str, strength: float = 1.0) -> bool:
        """Create a relationship between two content items.
        
        Args:
            source_id: ID of the source content
            target_id: ID of the target content
            relationship_type: Type of relationship ('sequential', 'parallel', etc.)
            strength: Strength of the relationship (0.0 to 1.0)
            
        Returns:
            True if relationship was created successfully
        """
        return self.content_organization.organizer.create_content_relationships(source_id, target_id, relationship_type, strength)

    def visualize_content_structure(self, output_path: Union[str, Path]) -> None:
        """Generate a visual representation of the content structure.

        Args:
            output_path: The path to save the visualization.
        """
        if not self.current_outline:
            raise ValueError("No outline loaded. Please load or create an outline first.")
        self.content_organization.visualize_content_structure(self.current_outline, output_path)

    
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

    def do_organize_content(self, arg):
        """Organize the book content by clustering it and providing a summary.

        Usage: organize_content <content_id1> <content_id2> ...
        """
        if not arg:
            print("Error: Content IDs are required")
            return

        content_ids = arg.split()

        try:
            summary = self.app.organize_content(content_ids)
            print(json.dumps(summary, indent=2))
        except Exception as e:
            print(f"Error organizing content: {e}")

    def do_get_suggestions(self, arg):
        """Get suggestions for improving the content organization.

        Usage: get_suggestions
        """
        try:
            suggestions = self.app.get_organization_suggestions()
            print(json.dumps(suggestions, indent=2))
        except Exception as e:
            print(f"Error getting suggestions: {e}")

    def do_visualize_content(self, arg):
        """Generate a visual representation of the content structure.

        Usage: visualize_content <output_path>
        """
        if not arg:
            print("Error: Output path is required")
            return

        output_path = arg

        try:
            self.app.visualize_content_structure(output_path)
            print(f"Content structure visualization saved to {output_path}.png")
        except Exception as e:
            print(f"Error visualizing content structure: {e}")
    
    
    def do_gen_report(self, arg):
        """Generate a comprehensive organization report for the current book.
        
        Usage: gen_report
        """
        try:
            report = self.app.generate_organization_report()
            print(report)
            
            # Also save the report to a file
            report_path = self.app.project_path / "output" / "organization_report.txt"
            os.makedirs(report_path.parent, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report also saved to: {report_path}")
        except Exception as e:
            print(f"Error generating organization report: {e}")

    def do_show_progress(self, arg):
        """Show writing progress and recommendations.
        
        Usage: show_progress
        """
        try:
            progress_data = self.app.get_writing_progress()
            print(f"\n📚 Writing Progress Report")
            print(f"=========================")
            print(f"Target Pages: {progress_data['total_target_pages']}")
            print(f"Written Pages: {progress_data['total_written_pages']:.1f}")
            print(f"Progress: {progress_data['progress_percentage']:.1f}%")
            print(f"Total Notes: {progress_data['total_notes']}")
            
            # Show progress by major sections
            print(f"\n📖 Progress by Part:")
            for part in progress_data['progress_by_section']:
                print(f"  • {part['title']}: {part['written_pages']:.1f}/{part['target_pages']:.1f} pages ({(part['written_pages']/part['target_pages']*100) if part['target_pages'] > 0 else 0:.1f}%)")
            
            # Show gamification recommendations
            recommendations_data = self.app.get_gamification_recommendations()
            print(f"\n🎯 Recommendations:")
            for i, rec in enumerate(recommendations_data['recommendations'], 1):
                print(f"  {i}. {rec}")
            
            if recommendations_data['milestones']:
                print(f"\n🏆 Milestones Reached:")
                for milestone in recommendations_data['milestones']:
                    print(f"  • {milestone}")
                    
        except Exception as e:
            print(f"Error showing progress: {e}")
    
    def do_visualize_outline(self, arg):
        """Generate a text-based visualization of the outline structure.
        
        Usage: visualize_outline
        """
        try:
            viz = self.app.visualize_outline_structure()
            print(viz)
        except Exception as e:
            print(f"Error visualizing outline: {e}")
    
    def do_create_relationship(self, arg):
        """Create a relationship between two content items.
        
        Usage: create_relationship <source_id> <target_id> <relationship_type> [strength]
        """
        args = arg.split()
        if len(args) < 3:
            print("Error: Source ID, target ID, and relationship type are required")
            return
        
        source_id = args[0]
        target_id = args[1]
        relationship_type = args[2]
        strength = float(args[3]) if len(args) > 3 else 1.0
        
        try:
            success = self.app.create_content_relationship(source_id, target_id, relationship_type, strength)
            if success:
                print(f"Relationship created successfully: {source_id} -> {target_id} ({relationship_type})")
            else:
                print("Failed to create relationship")
        except Exception as e:
            print(f"Error creating relationship: {e}")
    
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