"""
Book Writer System - Outline Module
Handles creation and management of book outlines in YAML/JSON format with AI assistance
"""
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

import yaml

from book_writer.model_manager import model_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BookOutline:
    """Class for managing book outlines with hierarchical structure."""
    
    def __init__(self, title: str, author: str = "", description: str = ""):
        """Initialize a new book outline.
        
        Args:
            title: The title of the book
            author: The author of the book
            description: A brief description of the book
        """
        self.title = title
        self.author = author
        self.description = description
        self.parts = []
        self.uuid = str(uuid.uuid4())
        self.created_at = None
        self.updated_at = None
        # Additional metadata for enhanced organization
        self.metadata = {
            "topic_keywords": [],
            "importance_weights": {},  # Per part/chapter/subtopic
            "prerequisites": {},       # Content that should come before
            "dependencies": {},        # Content that depends on this
            "relationships": {},       # Relationships between content elements
            "suggested_reading_order": [],
            "content_progression": {}  # How content builds on previous concepts
        }
    
    def add_part(self, title: str, description: str = "") -> Dict:
        """Add a new part to the book.
        
        Args:
            title: The title of the part
            description: A brief description of the part
            
        Returns:
            The newly created part as a dictionary
        """
        part_id = str(uuid.uuid4())
        part = {
            "id": part_id,
            "title": title,
            "description": description,
            "chapters": []
        }
        self.parts.append(part)
        return part
    
    def add_chapter(self, part_id: str, title: str, description: str = "") -> Dict:
        """Add a new chapter to a part.
        
        Args:
            part_id: The ID of the part to add the chapter to
            title: The title of the chapter
            description: A brief description of the chapter
            
        Returns:
            The newly created chapter as a dictionary
        """
        for part in self.parts:
            if part["id"] == part_id:
                chapter_id = str(uuid.uuid4())
                chapter = {
                    "id": chapter_id,
                    "title": title,
                    "description": description,
                    "subtopics": []
                }
                part["chapters"].append(chapter)
                return chapter
        raise ValueError(f"Part with ID {part_id} not found")
    
    def add_subtopic(self, chapter_id: str, title: str, description: str = "") -> Dict:
        """Add a new subtopic to a chapter.
        
        Args:
            chapter_id: The ID of the chapter to add the subtopic to
            title: The title of the subtopic
            description: A brief description of the subtopic
            
        Returns:
            The newly created subtopic as a dictionary
        """
        for part in self.parts:
            for chapter in part["chapters"]:
                if chapter["id"] == chapter_id:
                    subtopic_id = str(uuid.uuid4())
                    subtopic = {
                        "id": subtopic_id,
                        "title": title,
                        "description": description,
                        "content_ids": []  # Will store IDs of content chunks
                    }
                    chapter["subtopics"].append(subtopic)
                    return subtopic
        raise ValueError(f"Chapter with ID {chapter_id} not found")
    
    def to_dict(self) -> Dict:
        """Convert the outline to a dictionary.
        
        Returns:
            The outline as a dictionary
        """
        return {
            "title": self.title,
            "author": self.author,
            "description": self.description,
            "uuid": self.uuid,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parts": self.parts,
            "metadata": self.metadata
        }
    
    def generate_from_topic(self, topic: str, style: str = "academic", model_name: str = None) -> None:
        """Generate a complete book outline from a topic using AI.
        
        Args:
            topic: The topic for the book
            style: The style for the book (academic, narrative, technical, etc.)
            model_name: The name of the model to use for generation
        """
        # Create a prompt for the AI to generate an outline
        prompt = f"""
        Create a comprehensive book outline for a book about: {topic}
        
        The book should be in {style} style. Generate a structure with:
        - 2-4 main parts
        - 3-5 chapters per part
        - 2-4 subtopics per chapter
        
        Please return the outline in this JSON format:
        {{
            "parts": [
                {{
                    "id": "unique_id",
                    "title": "Part title",
                    "description": "Brief description of the part",
                    "chapters": [
                        {{
                            "id": "unique_id",
                            "title": "Chapter title",
                            "description": "Brief description of the chapter",
                            "subtopics": [
                                {{
                                    "id": "unique_id",
                                    "title": "Subtopic title",
                                    "description": "Brief description of the subtopic"
                                }}
                            ]
                        }}
                    ]
                }}
            ]
        }}
        """
        
        try:
            # Use the model manager with the organization model to generate the outline
            logger.info(f"Generating outline for topic: {topic}")
            result = model_manager.generate_response(
                prompt=prompt,
                task="outline_generation",
                model_name=model_name,
                format_json=True,
                temperature=0.4,  # Lower temperature for more consistent output
                max_tokens=800,   # Optimized token count for faster generation
                top_p=0.7         # Adjusted for more focused generation
            )
            logger.info(f"AI response: {result}")
            
            if isinstance(result, dict) and "parts" in result:
                # Clear existing parts
                self.parts = []
                
                # Add the generated parts, chapters, and subtopics
                for part_data in result["parts"]:
                    # Create the part
                    part_id = part_data.get("id", str(uuid.uuid4()))
                    part = {
                        "id": part_id,
                        "title": part_data["title"],
                        "description": part_data["description"],
                        "chapters": []
                    }
                    
                    # Add chapters to the part
                    for chapter_data in part_data["chapters"]:
                        chapter_id = chapter_data.get("id", str(uuid.uuid4()))
                        chapter = {
                            "id": chapter_id,
                            "title": chapter_data["title"],
                            "description": chapter_data["description"],
                            "subtopics": []
                        }
                        
                        # Add subtopics to the chapter
                        for subtopic_data in chapter_data["subtopics"]:
                            subtopic_id = subtopic_data.get("id", str(uuid.uuid4()))
                            subtopic = {
                                "id": subtopic_id,
                                "title": subtopic_data["title"],
                                "description": subtopic_data["description"],
                                "content_ids": []
                            }
                            
                            chapter["subtopics"].append(subtopic)
                        
                        part["chapters"].append(chapter)
                    
                    self.parts.append(part)
            else:
                print("AI-generated outline format was not as expected. Using fallback.")
                # Fallback to manual creation
                self._create_fallback_outline(topic)
                
        except Exception as e:
            print(f"Error generating outline from topic: {e}")
            # Fallback to manual creation
            self._create_fallback_outline(topic)
    
    def _create_fallback_outline(self, topic: str) -> None:
        """Create a basic outline when AI generation fails.
        
        Args:
            topic: The topic for the book
        """
        # Create a simple outline structure as a fallback
        part1 = self.add_part(f"Introduction to {topic}", f"An introduction to the topic of {topic}")
        part2 = self.add_part(f"Main Concepts of {topic}", f"The main concepts and principles of {topic}")
        part3 = self.add_part(f"Advanced Topics in {topic}", f"Advanced topics and applications of {topic}")
        
        # Add chapters to Part 1
        chapter1 = self.add_chapter(part1["id"], f"What is {topic}?", f"Understanding the basics of {topic}")
        chapter2 = self.add_chapter(part1["id"], f"History of {topic}", f"A brief history of {topic}")
        
        # Add chapters to Part 2
        chapter3 = self.add_chapter(part2["id"], f"Key Principles", f"The fundamental principles of {topic}")
        chapter4 = self.add_chapter(part2["id"], f"Common Applications", f"How {topic} is commonly applied")
        
        # Add chapters to Part 3
        chapter5 = self.add_chapter(part3["id"], f"Advanced Techniques", f"Advanced techniques in {topic}")
        chapter6 = self.add_chapter(part3["id"], f"Future of {topic}", f"The future of {topic}")
        
        # Add subtopics to chapters
        self.add_subtopic(chapter1["id"], f"Basic Concepts", f"The most basic concepts of {topic}")
        self.add_subtopic(chapter1["id"], f"Terminology", f"Key terminology in {topic}")
        self.add_subtopic(chapter2["id"], f"Early Developments", f"Early developments in {topic}")
        self.add_subtopic(chapter2["id"], f"Modern Evolution", f"How {topic} has evolved")
        self.add_subtopic(chapter3["id"], f"Core Principles", f"The core principles of {topic}")
        self.add_subtopic(chapter3["id"], f"Secondary Principles", f"Secondary principles of {topic}")
        self.add_subtopic(chapter4["id"], f"Common Use Cases", f"Common use cases of {topic}")
        self.add_subtopic(chapter4["id"], f"Industry Applications", f"Industry applications of {topic}")
        self.add_subtopic(chapter5["id"], f"Advanced Methods", f"Advanced methods in {topic}")
        self.add_subtopic(chapter5["id"], f"Expert Techniques", f"Expert-level techniques in {topic}")
        self.add_subtopic(chapter6["id"], f"Emerging Trends", f"Emerging trends in {topic}")
        self.add_subtopic(chapter6["id"], f"Future Challenges", f"Future challenges for {topic}")
    
    def set_importance_weight(self, element_id: str, weight: float, element_type: str = "auto"):
        """
        Set importance weight for a specific element (part, chapter, or subtopic).
        
        Args:
            element_id: ID of the element
            weight: Importance weight (0.0 to 1.0)
            element_type: Type of element ('part', 'chapter', 'subtopic', or 'auto')
        """
        element_type = element_type.lower()
        
        # If auto-detect, check across all levels
        if element_type == "auto":
            element_type = self._get_element_type(element_id)
        
        if element_type and element_type in ["part", "chapter", "subtopic"]:
            self.metadata["importance_weights"][element_id] = max(0.0, min(1.0, weight))
    
    def get_importance_weight(self, element_id: str) -> float:
        """
        Get importance weight for a specific element.
        
        Args:
            element_id: ID of the element
            
        Returns:
            Importance weight (0.0 to 1.0)
        """
        return self.metadata["importance_weights"].get(element_id, 0.5)  # Default to 0.5
    
    def _get_element_type(self, element_id: str) -> Optional[str]:
        """
        Helper method to determine the type of an element by its ID.
        
        Args:
            element_id: The ID to check
            
        Returns:
            The type of element ('part', 'chapter', 'subtopic') or None if not found
        """
        for part in self.parts:
            if part["id"] == element_id:
                return "part"
            for chapter in part["chapters"]:
                if chapter["id"] == element_id:
                    return "chapter"
                for subtopic in chapter["subtopics"]:
                    if subtopic["id"] == element_id:
                        return "subtopic"
        return None
    
    def add_topic_keywords(self, keywords: List[str]):
        """
        Add topic keywords to the outline metadata.
        
        Args:
            keywords: List of topic keywords
        """
        for keyword in keywords:
            if keyword not in self.metadata["topic_keywords"]:
                self.metadata["topic_keywords"].append(keyword)
    
    def add_relationship(self, source_id: str, target_id: str, relationship_type: str, strength: float = 1.0):
        """
        Add a relationship between two elements in the outline.
        
        Args:
            source_id: ID of the source element
            target_id: ID of the target element
            relationship_type: Type of relationship (e.g., 'prerequisite', 'sequential', 'complementary')
            strength: Strength of the relationship (0.0 to 1.0)
        """
        if source_id not in self.metadata["relationships"]:
            self.metadata["relationships"][source_id] = []
        
        # Check if relationship already exists and update it
        existing_idx = None
        for i, rel in enumerate(self.metadata["relationships"][source_id]):
            if rel["target_id"] == target_id and rel["type"] == relationship_type:
                existing_idx = i
                break
        
        new_relationship = {
            "target_id": target_id,
            "type": relationship_type,
            "strength": max(0.0, min(1.0, strength))
        }
        
        if existing_idx is not None:
            self.metadata["relationships"][source_id][existing_idx] = new_relationship
        else:
            self.metadata["relationships"][source_id].append(new_relationship)
    
    def get_related_elements(self, element_id: str, relationship_type: Optional[str] = None) -> List[Dict]:
        """
        Get elements related to the specified element.
        
        Args:
            element_id: ID of the element
            relationship_type: Optional filter for specific relationship type
            
        Returns:
            List of related elements with relationship information
        """
        related = []
        
        # Check for relationships where this element is the source
        if element_id in self.metadata["relationships"]:
            for rel in self.metadata["relationships"][element_id]:
                if relationship_type is None or rel["type"] == relationship_type:
                    related.append(rel)
        
        # Check for relationships where this element is the target
        for source_id, relationships in self.metadata["relationships"].items():
            for rel in relationships:
                if rel["target_id"] == element_id and (relationship_type is None or rel["type"] == relationship_type):
                    related.append({
                        "target_id": source_id,
                        "type": rel["type"],
                        "strength": rel["strength"],
                        "direction": "reverse"
                    })
        
        return related
    
    def get_prerequisites(self, element_id: str) -> List[str]:
        """
        Get prerequisite elements for the specified element.
        
        Args:
            element_id: ID of the element
            
        Returns:
            List of prerequisite element IDs
        """
        prerequisites = []
        for source_id, relationships in self.metadata["relationships"].items():
            for rel in relationships:
                if rel["target_id"] == element_id and rel["type"] == "prerequisite":
                    prerequisites.append(source_id)
        return prerequisites
    
    def validate_content_progression(self) -> List[Dict]:
        """
        Validate that content follows logical progression based on prerequisites.
        
        Returns:
            List of validation issues found
        """
        issues = []
        
        # Check prerequisites for each element
        for part in self.parts:
            for chapter in part["chapters"]:
                for subtopic in chapter["subtopics"]:
                    # Check if prerequisites have been satisfied
                    prereqs = self.get_prerequisites(subtopic["id"])
                    for prereq_id in prereqs:
                        if not self._is_prereq_satisfied(prereq_id, subtopic["id"]):
                            issues.append({
                                "element_id": subtopic["id"],
                                "element_title": subtopic["title"],
                                "prereq_id": prereq_id,
                                "issue": "Prerequisite not properly positioned before this element"
                            })
        
        return issues
    
    def _is_prereq_satisfied(self, prereq_id: str, target_id: str) -> bool:
        """
        Check if a prerequisite element appears before the target element in the outline.
        
        Args:
            prereq_id: ID of the prerequisite element
            target_id: ID of the target element
            
        Returns:
            True if prerequisite is satisfied, False otherwise
        """
        # Find positions of both elements in the outline
        prereq_position = self._get_element_position(prereq_id)
        target_position = self._get_element_position(target_id)
        
        if prereq_position is None or target_position is None:
            return False  # Can't determine positions
        
        # Check if prerequisite comes before target
        return prereq_position < target_position
    
    def _get_element_position(self, element_id: str) -> Optional[int]:
        """
        Get the position of an element in the outline hierarchy.
        
        Args:
            element_id: ID of the element
            
        Returns:
            Position value or None if not found
        """
        position = 0
        for part_idx, part in enumerate(self.parts):
            if part["id"] == element_id:
                return position
            position += 1
            
            for chapter_idx, chapter in enumerate(part["chapters"]):
                if chapter["id"] == element_id:
                    return position
                position += 1
                
                for subtopic_idx, subtopic in enumerate(chapter["subtopics"]):
                    if subtopic["id"] == element_id:
                        return position
                    position += 1
        
        return None
    
    @classmethod
    def from_dict(cls, data: Dict) -> "BookOutline":
        """Create an outline from a dictionary.
        
        Args:
            data: The dictionary containing outline data
            
        Returns:
            A new BookOutline instance
        """
        outline = cls(data["title"], data.get("author", ""), data.get("description", ""))
        outline.uuid = data.get("uuid", str(uuid.uuid4()))
        outline.created_at = data.get("created_at")
        outline.updated_at = data.get("updated_at")
        outline.parts = data.get("parts", [])
        outline.metadata = data.get("metadata", {
            "topic_keywords": [],
            "importance_weights": {},
            "prerequisites": {},
            "dependencies": {},
            "relationships": {},
            "suggested_reading_order": [],
            "content_progression": {}
        })
        return outline
    
    def save(self, file_path: Union[str, Path], format: str = "yaml") -> None:
        """Save the outline to a file.
        
        Args:
            file_path: The path to save the outline to
            format: The format to save in ('yaml' or 'json')
        """
        data = self.to_dict()
        
        if format.lower() == "yaml":
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif format.lower() == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "BookOutline":
        """Load an outline from a file.
        
        Args:
            file_path: The path to load the outline from
            
        Returns:
            A new BookOutline instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif file_path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return cls.from_dict(data)


class OutlineManager:
    """Class for managing book outlines."""
    
    def __init__(self, project_path: Union[str, Path]):
        """Initialize a new outline manager.
        
        Args:
            project_path: The path to the project directory
        """
        self.project_path = Path(project_path)
        self.outlines_dir = self.project_path / "outlines"
        self.outlines_dir.mkdir(exist_ok=True, parents=True)
    
    def create_outline(self, title: str, author: str = "", description: str = "") -> BookOutline:
        """Create a new book outline.
        
        Args:
            title: The title of the book
            author: The author of the book
            description: A brief description of the book
            
        Returns:
            A new BookOutline instance
        """
        outline = BookOutline(title, author, description)
        return outline
    
    def save_outline(self, outline: BookOutline, format: str = "yaml") -> Path:
        """Save an outline to a file.
        
        Args:
            outline: The outline to save
            format: The format to save in ('yaml' or 'json')
            
        Returns:
            The path to the saved file
        """
        if format.lower() == "yaml":
            file_path = self.outlines_dir / f"{outline.title.replace(' ', '_').lower()}.yaml"
        elif format.lower() == "json":
            file_path = self.outlines_dir / f"{outline.title.replace(' ', '_').lower()}.json"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        outline.save(file_path, format)
        return file_path
    
    def load_outline(self, file_name: str) -> BookOutline:
        """Load an outline from a file.
        
        Args:
            file_name: The name of the file to load
            
        Returns:
            A BookOutline instance
        """
        file_path = self.outlines_dir / file_name
        return BookOutline.load(file_path)
    
    def list_outlines(self) -> List[str]:
        """List all outlines in the project.
        
        Returns:
            A list of outline file names
        """
        return [f.name for f in self.outlines_dir.glob("*.yaml")] + [f.name for f in self.outlines_dir.glob("*.json")]


def create_sample_outline(topic: str = "Artificial Intelligence", title: str = None) -> BookOutline:
    """Create a sample book outline using AI.
    
    Args:
        topic: The topic for the sample outline (default: "Artificial Intelligence")
        title: The title for the book (if provided, overrides the default format)
    
    Returns:
        A sample BookOutline instance
    """
    # Use provided title or default format
    book_title = title if title is not None else f"Understanding {topic}"
    description = f"An AI-generated guide to {topic}"
    
    outline = BookOutline(book_title, "AI Assistant", description)
    
    try:
        # Try with the phi3.5 model first (as specified in config)
        outline.generate_from_topic(topic, style="academic")
        if not outline.parts:
            # If the primary model fails, try with the secondary model
            logger.warning("Primary model failed to generate outline, trying secondary model.")
            outline.generate_from_topic(topic, style="academic", model_name="stable-beluga:13b")
    except Exception as e:
        logger.error(f"Error generating outline: {e}")

    if not outline.parts:
        # If both models fail, create a fallback outline
        logger.error("All models failed to generate outline, creating a fallback outline.")
        outline._create_fallback_outline(f"{topic} (AI Generation Failed)")
        outline.description = "This is a template outline. AI generation failed. Please fill in the details."

    return outline