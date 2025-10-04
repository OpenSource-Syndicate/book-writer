"""
Book Writer System - Note Processing Module
Handles embedding and storage of notes using BGE-M3 model and ChromaDB
"""
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


class NoteProcessor:
    """Class for processing and embedding notes."""
    
    def __init__(self, project_path: Union[str, Path]):
        """Initialize a new note processor.
        
        Args:
            project_path: The path to the project directory
        """
        self.project_path = Path(project_path)
        self.notes_dir = self.project_path / "notes"
        self.notes_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize embedding model
        self.model_name = "BAAI/bge-m3"
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Warning: Could not load BGE-M3 model. Using default model instead. Error: {e}")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize ChromaDB
        self.db_path = self.project_path / "db"
        self.db_path.mkdir(exist_ok=True, parents=True)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Create collections if they don't exist
        self.notes_collection = self.client.get_or_create_collection(
            name="notes",
            metadata={"description": "Raw notes with embeddings"}
        )
        self.content_collection = self.client.get_or_create_collection(
            name="content",
            metadata={"description": "Expanded content chunks"}
        )
    
    def process_note(self, 
                     text: str, 
                     source: str = "conversation", 
                     potential_topics: List[str] = None,
                     metadata: Dict = None) -> str:
        """Process a new note by embedding it and storing in ChromaDB.
        
        Args:
            text: The text of the note
            source: The source of the note
            potential_topics: List of potential topics for the note
            metadata: Additional metadata for the note
            
        Returns:
            The ID of the processed note
        """
        # Generate a unique ID for the note
        note_id = str(uuid.uuid4())
        
        # Create metadata
        if metadata is None:
            metadata = {}
        
        # Convert potential_topics to a comma-separated string if it's a list, otherwise use empty string
        topics_str = ",".join(potential_topics) if potential_topics else ""
        
        full_metadata = {
            "source": source,
            "timestamp": time.time(),
            "potential_topics": topics_str,
            **metadata
        }
        
        # Generate embedding
        embedding = self.embed_text(text)
        
        # Filter out any None values from metadata as ChromaDB doesn't accept them
        filtered_metadata = {k: v for k, v in full_metadata.items() if v is not None}
        
        # Store in ChromaDB
        self.notes_collection.add(
            ids=[note_id],
            embeddings=[embedding.tolist()],
            metadatas=[filtered_metadata],
            documents=[text]
        )
        
        # Generate embedding
        embedding = self.embed_text(text)
        
        # Store in ChromaDB
        self.notes_collection.add(
            ids=[note_id],
            embeddings=[embedding.tolist()],
            metadatas=[full_metadata],
            documents=[text]
        )
        
        # Save raw note to file
        self._save_note_to_file(note_id, text, full_metadata)
        
        return note_id
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding as a numpy array
        """
        return self.model.encode(text)
    
    def _save_note_to_file(self, note_id: str, text: str, metadata: Dict) -> None:
        """Save a note to a file.
        
        Args:
            note_id: The ID of the note
            text: The text of the note
            metadata: The metadata for the note
        """
        note_data = {
            "id": note_id,
            "text": text,
            "metadata": metadata
        }
        
        file_path = self.notes_dir / f"{note_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(note_data, f, indent=2, ensure_ascii=False)
    
    def retrieve_similar_notes(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve notes similar to the query.
        
        Args:
            query: The query text
            n_results: The number of results to return
            
        Returns:
            A list of similar notes with their metadata
        """
        # Generate embedding for the query
        query_embedding = self.embed_text(query)
        
        # Query ChromaDB
        results = self.notes_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            # Convert potential_topics back to a list if it exists as a string
            if "potential_topics" in metadata and metadata["potential_topics"] is not None:
                # If potential_topics is an empty string, return an empty list
                if metadata["potential_topics"] == "":
                    metadata["potential_topics"] = []
                else:
                    metadata["potential_topics"] = metadata["potential_topics"].split(",")
            # If potential_topics key doesn't exist, set it to an empty list for consistency
            elif "potential_topics" not in metadata:
                metadata["potential_topics"] = []
            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": metadata,
                "distance": results["distances"][0][i]
            })
        
        return formatted_results
    
    def classify_note(self, note_id: str, chapter_id: str, subtopic_id: str) -> None:
        """Classify a note into a specific chapter and subtopic.
        
        Args:
            note_id: The ID of the note
            chapter_id: The ID of the chapter
            subtopic_id: The ID of the subtopic
        """
        # Get the note from ChromaDB
        results = self.notes_collection.get(
            ids=[note_id],
            include=["metadatas"]
        )
        
        if not results["ids"]:
            raise ValueError(f"Note with ID {note_id} not found")
        
        # Update metadata
        metadata = results["metadatas"][0]
        metadata["chapter_id"] = chapter_id
        metadata["subtopic_id"] = subtopic_id
        
        # Convert potential_topics back to string format for storage if it exists as a list
        if "potential_topics" in metadata and isinstance(metadata["potential_topics"], list):
            metadata["potential_topics"] = ",".join(metadata["potential_topics"]) if metadata["potential_topics"] else ""
        
        # Filter out any None values from metadata as ChromaDB doesn't accept them
        filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Update in ChromaDB
        self.notes_collection.update(
            ids=[note_id],
            metadatas=[filtered_metadata]
        )
        
        # Update file
        file_path = self.notes_dir / f"{note_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                note_data = json.load(f)
            
            note_data["metadata"] = metadata
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(note_data, f, indent=2, ensure_ascii=False)


class ContentManager:
    """Class for managing expanded content chunks."""
    
    def __init__(self, project_path: Union[str, Path], note_processor: NoteProcessor):
        """Initialize a new content manager.
        
        Args:
            project_path: The path to the project directory
            note_processor: A NoteProcessor instance for embedding content
        """
        self.project_path = Path(project_path)
        self.content_dir = self.project_path / "content"
        self.content_dir.mkdir(exist_ok=True, parents=True)
        self.note_processor = note_processor
    
    def store_content(self, 
                      content: str, 
                      title: str, 
                      chapter_id: str, 
                      subtopic_id: str,
                      source_note_ids: List[str] = None) -> str:
        """Store expanded content.
        
        Args:
            content: The expanded content text
            title: The title of the content
            chapter_id: The ID of the chapter
            subtopic_id: The ID of the subtopic
            source_note_ids: List of source note IDs
            
        Returns:
            The ID of the stored content
        """
        # Generate a unique ID for the content
        content_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = {
            "title": title,
            "chapter_id": chapter_id,
            "subtopic_id": subtopic_id,
            "timestamp": time.time(),
            "source_note_ids": ",".join(source_note_ids) if source_note_ids else ""
        }
        
        # Generate embedding
        embedding = self.note_processor.embed_text(content)
        
        # Store in ChromaDB
        self.note_processor.content_collection.add(
            ids=[content_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[content]
        )
        
        # Save content to file
        self._save_content_to_file(content_id, content, metadata)
        
        return content_id

    def update_content(self, content_id: str, new_text: str):
        """Update the content of an existing content file.

        Args:
            content_id: The ID of the content to update.
            new_text: The new text to write to the file.
        """
        content_path = self.content_dir / f"{content_id}.md"
        if not content_path.exists():
            raise FileNotFoundError(f"Content with ID {content_id} not found.")
        
        with open(content_path, "w", encoding="utf-8") as f:
            f.write(new_text)

    def get_content(self, content_id: str) -> str:
        """Retrieve the content of a specific content file.

        Args:
            content_id: The ID of the content to retrieve.

        Returns:
            The content of the file as a string.
        """
        content_path = self.content_dir / f"{content_id}.md"
        if not content_path.exists():
            raise FileNotFoundError(f"Content with ID {content_id} not found.")
        
        with open(content_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def _save_content_to_file(self, content_id: str, content: str, metadata: Dict) -> None:
        """Save content to a file.
        
        Args:
            content_id: The ID of the content
            content: The content text
            metadata: The metadata for the content
        """
        content_data = {
            "id": content_id,
            "content": content,
            "metadata": metadata
        }
        
        file_path = self.content_dir / f"{content_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(content_data, f, indent=2, ensure_ascii=False)
    
    def retrieve_content_by_subtopic(self, subtopic_id: str) -> List[Dict]:
        """Retrieve all content for a specific subtopic.
        
        Args:
            subtopic_id: The ID of the subtopic
            
        Returns:
            A list of content items with their metadata
        """
        # Query ChromaDB
        results = self.note_processor.content_collection.get(
            where={"subtopic_id": subtopic_id},
            include=["documents", "metadatas"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"])):
            formatted_results.append({
                "id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i]
            })
        
        return formatted_results
    
    def retrieve_similar_content(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve content similar to the query.
        
        Args:
            query: The query text
            n_results: The number of results to return
            
        Returns:
            A list of similar content items with their metadata
        """
        # Generate embedding for the query
        query_embedding = self.note_processor.embed_text(query)
        
        # Query ChromaDB
        results = self.note_processor.content_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return formatted_results