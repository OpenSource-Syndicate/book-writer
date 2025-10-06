"""
Book Writer System - Book Assembly Module
Handles assembly of book content into a complete manuscript
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import markdown
from tqdm import tqdm

from book_writer.note_processor import ContentManager
from book_writer.outline import BookOutline

# Set up logging for the assembly module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssemblyConfig:
    """
    Configuration class for customizing the assembly process.
    """
    
    def __init__(self, 
                 include_title_page: bool = True,
                 include_toc: bool = True,
                 include_part_descriptions: bool = True,
                 include_chapter_descriptions: bool = True,
                 include_subtopic_descriptions: bool = True,
                 content_sorting: str = 'timestamp',  # Options: 'timestamp', 'alpha', 'none'
                 output_format: str = 'markdown',
                 custom_separators: Dict[str, str] = None,
                 processor_options: Dict[str, Any] = None):
        """
        Initialize assembly configuration.
        
        Args:
            include_title_page: Whether to include the title page
            include_toc: Whether to include table of contents
            include_part_descriptions: Whether to include part descriptions
            include_chapter_descriptions: Whether to include chapter descriptions
            include_subtopic_descriptions: Whether to include subtopic descriptions
            content_sorting: How to sort content ('timestamp', 'alpha', 'none')
            output_format: Default output format
            custom_separators: Custom separators for different sections
            processor_options: Options for specific processors
        """
        self.include_title_page = include_title_page
        self.include_toc = include_toc
        self.include_part_descriptions = include_part_descriptions
        self.include_chapter_descriptions = include_chapter_descriptions
        self.include_subtopic_descriptions = include_subtopic_descriptions
        self.content_sorting = content_sorting
        self.output_format = output_format
        self.custom_separators = custom_separators or {
            'part': "\n---\n\n",
            'default': "\n\n"
        }
        self.processor_options = processor_options or {}


class AssemblyContext:
    """
    Holds the context for the assembly process, including outline, content,
    and intermediate results.
    """
    
    def __init__(self, outline: BookOutline, config: AssemblyConfig = None):
        self.outline = outline
        self.content = []  # List of content blocks to be assembled
        self.metadata = {}  # Additional assembly metadata
        self.options = {}   # Assembly options and settings
        self.config = config or AssemblyConfig()  # Assembly configuration


class ContentProcessor:
    """
    Base class for content processors in the assembly pipeline.
    Each processor can modify the assembly context during different stages.
    """
    
    def process(self, context: AssemblyContext) -> AssemblyContext:
        """
        Process the assembly context and return the (potentially modified) context.
        
        Args:
            context: The current assembly context
            
        Returns:
            The processed assembly context
        """
        # Default implementation returns context unchanged
        return context
    
    @property
    def name(self) -> str:
        """Get the name of this processor (defaults to class name)."""
        return self.__class__.__name__


class TitleProcessor(ContentProcessor):
    """
    Adds title page information to the book content.
    """
    
    def process(self, context: AssemblyContext) -> AssemblyContext:
        if context.config.include_title_page:
            # Add title page
            context.content.append(f"# {context.outline.title}\n")
            if context.outline.author:
                context.content.append(f"## By {context.outline.author}\n")
            if context.outline.description:
                context.content.append(f"*{context.outline.description}*\n")
            
            context.content.append("\n---\n\n")
        return context


class TOCProcessor(ContentProcessor):
    """
    Adds a table of contents to the book content.
    """
    
    def process(self, context: AssemblyContext) -> AssemblyContext:
        if context.config.include_toc:
            context.content.append("# Table of Contents\n")
            
            for i, part in enumerate(context.outline.parts, 1):
                context.content.append(f"{i}. {part['title']}\n")
                
                for j, chapter in enumerate(part['chapters'], 1):
                    context.content.append(f"   {i}.{j}. {chapter['title']}\n")
            
            context.content.append("\n---\n\n")
        return context


class ContentProcessor(ContentProcessor):
    """
    Processes the main content from outline parts, chapters, and subtopics.
    """
    
    def __init__(self, content_manager: ContentManager):
        self.content_manager = content_manager
    
    def process(self, context: AssemblyContext) -> AssemblyContext:
        # Add parts, chapters, and content
        for part_index, part in enumerate(tqdm(context.outline.parts, desc="Processing parts"), 1):
            context.content.append(f"# Part {part_index}: {part['title']}\n")
            
            if context.config.include_part_descriptions and part.get('description'):
                context.content.append(f"*{part['description']}*\n\n")
            
            for chapter_index, chapter in enumerate(part['chapters'], 1):
                context.content.append(f"## Chapter {part_index}.{chapter_index}: {chapter['title']}\n")
                
                if context.config.include_chapter_descriptions and chapter.get('description'):
                    context.content.append(f"*{chapter['description']}*\n\n")
                
                for subtopic in chapter['subtopics']:
                    context.content.append(f"### {subtopic['title']}\n")
                    
                    if context.config.include_subtopic_descriptions and subtopic.get('description'):
                        context.content.append(f"*{subtopic['description']}*\n\n")
                    
                    # Get content for this subtopic
                    try:
                        subtopic_content = self.content_manager.retrieve_content_by_subtopic(subtopic['id'])
                        
                        if subtopic_content:
                            # Sort content based on configuration
                            if context.config.content_sorting == 'timestamp':
                                subtopic_content.sort(
                                    key=lambda x: x['metadata'].get('timestamp', 0)
                                )
                            elif context.config.content_sorting == 'alpha':
                                subtopic_content.sort(
                                    key=lambda x: x['content'][:50]  # Sort by first 50 chars of content
                                )
                            # If 'none', no sorting is applied
                            
                            # Add all content for this subtopic
                            for content_item in subtopic_content:
                                context.content.append(f"{content_item['content']}\n\n")
                        else:
                            context.content.append("*No content available for this subtopic.*\n\n")
                    except Exception as e:
                        logger.warning(f"Error retrieving content for subtopic '{subtopic['title']}': {str(e)}")
                        context.content.append(f"*Error retrieving content for this subtopic: {str(e)}*\n\n")
                
                # Add separator between chapters
                separator = context.config.custom_separators.get('part', "\n---\n\n")
                context.content.append(separator)
        
        return context


class ContentFlowOptimizer(ContentProcessor):
    """
    Optimizes content flow within chapters and subtopics based on semantic similarity and coherence.
    """
    
    def __init__(self, content_manager: ContentManager):
        self.content_manager = content_manager
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        self.cosine_similarity = cosine_similarity
    
    def process(self, context: AssemblyContext) -> AssemblyContext:
        """
        Optimize content flow by reordering content items for better coherence.
        """
        print("Optimizing content flow...")
        
        # For each part, chapter, and subtopic, optimize the content order
        optimized_content = []
        content_idx = 0
        
        for part_index, part in enumerate(context.outline.parts, 1):
            # Add part header
            if f"# Part {part_index}: {part['title']}" in context.content[content_idx]:
                optimized_content.append(context.content[content_idx])
                content_idx += 1
            
            # Add part description if present
            if context.config.include_part_descriptions and part.get('description'):
                if f"*{part['description']}*" in context.content[content_idx]:
                    optimized_content.append(context.content[content_idx])
                    content_idx += 1
            
            for chapter_index, chapter in enumerate(part['chapters'], 1):
                # Add chapter header
                if f"## Chapter {part_index}.{chapter_index}: {chapter['title']}" in context.content[content_idx]:
                    optimized_content.append(context.content[content_idx])
                    content_idx += 1
                
                # Add chapter description if present
                if context.config.include_chapter_descriptions and chapter.get('description'):
                    if f"*{chapter['description']}*" in context.content[content_idx]:
                        optimized_content.append(context.content[content_idx])
                        content_idx += 1
                
                for subtopic in chapter['subtopics']:
                    # Add subtopic header
                    if f"### {subtopic['title']}" in context.content[content_idx]:
                        optimized_content.append(context.content[content_idx])
                        content_idx += 1
                    
                    # Add subtopic description if present
                    if context.config.include_subtopic_descriptions and subtopic.get('description'):
                        if f"*{subtopic['description']}*" in context.content[content_idx]:
                            optimized_content.append(context.content[content_idx])
                            content_idx += 1
                    
                    # Find content for this subtopic and optimize its order
                    subtopic_content_start = content_idx
                    subtopic_content = []
                    
                    # Extract content for this subtopic until we hit the next section
                    while (content_idx < len(context.content) and 
                           not context.content[content_idx].startswith('#') and
                           not context.content[content_idx].startswith('*No content available') and
                           not context.content[content_idx].startswith('*Error retrieving')):
                        line = context.content[content_idx]
                        if line.strip() != "":
                            subtopic_content.append((content_idx, line))
                        content_idx += 1
                    
                    # Optimize the order of content in this subtopic
                    if len(subtopic_content) > 1:
                        # Extract just the content text for optimization
                        content_texts = [item[1] for item in subtopic_content]
                        
                        # Optimize content order for better flow
                        optimized_indices = self._optimize_content_order(content_texts)
                        
                        # Add the optimized content
                        for opt_idx in optimized_indices:
                            optimized_content.append(subtopic_content[opt_idx][1])
                    else:
                        # If no optimization needed, just add the content as is
                        for _, content_line in subtopic_content:
                            optimized_content.append(content_line)
                    
                    # Add the content back to the optimized content list
                    while (content_idx < len(context.content) and 
                           context.content[content_idx].startswith('*No content available') or
                           context.content[content_idx].startswith('*Error retrieving') or
                           context.content[content_idx].startswith('\n---')):
                        optimized_content.append(context.content[content_idx])
                        content_idx += 1
                
                # Skip the separator if we just added it from the original content
                if (content_idx < len(context.content) and 
                    context.content[content_idx].startswith('\n---')):
                    content_idx += 1
        
        # Replace the context content with the optimized version
        context.content = optimized_content
        
        return context
    
    def _optimize_content_order(self, content_items: List[str]) -> List[int]:
        """
        Optimize the order of content items for better flow and coherence.
        
        Args:
            content_items: List of content strings
            
        Returns:
            List of indices in optimized order
        """
        if len(content_items) <= 1:
            return list(range(len(content_items)))
        
        # Create TF-IDF vectors for content items
        try:
            tfidf_matrix = self.vectorizer.fit_transform(content_items)
            
            # Calculate similarity matrix
            similarity_matrix = self.cosine_similarity(tfidf_matrix)
            
            # Use a greedy algorithm to order content for maximum coherence
            n_items = len(content_items)
            unvisited = set(range(n_items))
            order = []
            
            # Start with the first item
            current = 0
            order.append(current)
            unvisited.remove(current)
            
            # Greedily select the next most similar item
            while unvisited:
                best_next = -1
                best_similarity = -2  # Cosine similarity ranges from -1 to 1
                
                for candidate in unvisited:
                    sim = similarity_matrix[current, candidate]
                    if sim > best_similarity:
                        best_similarity = sim
                        best_next = candidate
                
                if best_next != -1:
                    current = best_next
                    order.append(current)
                    unvisited.remove(current)
                else:
                    # If no similar items found, just pick one randomly
                    current = unvisited.pop()
                    order.append(current)
            
            return order
        except Exception:
            # Fallback to original order if optimization fails
            return list(range(len(content_items)))


class ContentGapDetector(ContentProcessor):
    """
    Detects content gaps in the outline and suggests areas that need more content.
    """
    
    def __init__(self, content_manager: ContentManager):
        self.content_manager = content_manager
    
    def process(self, context: AssemblyContext) -> AssemblyContext:
        """
        Detect content gaps and add notes about missing content.
        """
        print("Detecting content gaps...")
        
        gap_report = []
        total_subtopics = 0
        empty_subtopics = 0
        
        for part in context.outline.parts:
            for chapter in part['chapters']:
                for subtopic in chapter['subtopics']:
                    total_subtopics += 1
                    subtopic_content = self.content_manager.retrieve_content_by_subtopic(subtopic['id'])
                    if not subtopic_content:
                        empty_subtopics += 1
                        gap_report.append(f"- Missing content for: {part['title']} -> {chapter['title']} -> {subtopic['title']}")
        
        if gap_report:
            gap_summary = [
                "\n## Content Gap Report\n",
                f"Total subtopics: {total_subtopics}\n",
                f"Subtopics with no content: {empty_subtopics}\n",
                f"Completion rate: {((total_subtopics - empty_subtopics) / total_subtopics * 100):.1f}%\n\n",
                "Subtopics that need content:\n"
            ] + gap_report + ["\n"]
            
            # Add gap report to the end of content
            context.content.extend(gap_summary)
        
        return context


class AssemblyPipeline:
    """
    Orchestrates the book assembly process by running processors in sequence.
    """
    
    def __init__(self, content_manager: ContentManager, processors: List[ContentProcessor] = None):
        self.content_manager = content_manager
        self.processors = processors or [
            TitleProcessor(),
            TOCProcessor(),
            ContentProcessor(content_manager),
            ContentFlowOptimizer(content_manager),  # Added for intelligent content flow
            ContentGapDetector(content_manager)     # Added for gap detection
        ]
    
    def add_processor(self, processor: ContentProcessor, index: int = None):
        """
        Add a processor to the pipeline.
        
        Args:
            processor: The processor to add
            index: Position to insert the processor (defaults to end)
        """
        if index is None:
            self.processors.append(processor)
        else:
            self.processors.insert(index, processor)
    
    def remove_processor(self, processor_name: str) -> bool:
        """
        Remove a processor from the pipeline by name.
        
        Args:
            processor_name: Name of the processor to remove
            
        Returns:
            True if processor was found and removed, False otherwise
        """
        for i, processor in enumerate(self.processors):
            if processor.name == processor_name:
                del self.processors[i]
                return True
        return False
    
    def run(self, outline: BookOutline, config: AssemblyConfig = None) -> AssemblyContext:
        """
        Run the assembly pipeline on the given outline.
        
        Args:
            outline: The book outline to assemble
            config: Assembly configuration to use
            
        Returns:
            The final assembly context containing the processed content
        """
        context = AssemblyContext(outline, config)
        
        for processor in self.processors:
            context = processor.process(context)
        
        return context


class BookAssembler:
    """Class for assembling book content into a complete manuscript."""
    
    def __init__(self, project_path: Union[str, Path], content_manager: ContentManager, 
                 default_config: AssemblyConfig = None):
        """Initialize a new book assembler.
        
        Args:
            project_path: The path to the project directory
            content_manager: A ContentManager instance for retrieving content
            default_config: Default assembly configuration
        """
        self.project_path = Path(project_path)
        self.content_manager = content_manager
        self.output_dir = self.project_path / "output"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set default configuration
        self.default_config = default_config or AssemblyConfig()
        
        # Create the default pipeline
        self.pipeline = AssemblyPipeline(content_manager)
    
    def build_book(self, outline: BookOutline, output_format: str = "markdown", 
                   config: AssemblyConfig = None) -> Path:
        """Build a complete book manuscript from the outline and content.
        
        Args:
            outline: The book outline
            output_format: The output format (markdown, html)
            config: Optional assembly configuration (uses default if not provided)
            
        Returns:
            The path to the built book
        """
        logger.info(f"Building book: {outline.title}")
        
        try:
            # Use provided config or default
            assembly_config = config or self.default_config

            # Create a timestamp for the output file
            timestamp = int(time.time())
            
            # Create the output file path
            if output_format.lower() == "markdown":
                output_file = self.output_dir / f"{outline.title.replace(' ', '_').lower()}_{timestamp}.md"
            elif output_format.lower() == "html":
                output_file = self.output_dir / f"{outline.title.replace(' ', '_').lower()}_{timestamp}.html"
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            # Run the pipeline to get assembled content
            logger.info("Starting assembly pipeline...")
            context = self.pipeline.run(outline, assembly_config)
            logger.info(f"Pipeline completed, assembled {len(context.content)} content segments")
            
            # Join all content segments
            full_content = "".join(context.content)
            
            # Write the content to the output file
            logger.info(f"Writing content to {output_file}")
            with open(output_file, "w", encoding="utf-8") as f:
                if output_format.lower() == "html":
                    html_content = markdown.markdown(full_content)
                    f.write(html_content)
                else:
                    f.write(full_content)
            
            logger.info(f"Book built successfully: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error building book '{outline.title}': {str(e)}")
            raise
    
    def add_processor(self, processor: ContentProcessor, index: int = None):
        """
        Add a processor to the default pipeline.
        
        Args:
            processor: The processor to add
            index: Position to insert the processor (defaults to end)
        """
        self.pipeline.add_processor(processor, index)
    
    def remove_processor(self, processor_name: str) -> bool:
        """
        Remove a processor from the default pipeline.
        
        Args:
            processor_name: Name of the processor to remove
            
        Returns:
            True if processor was found and removed, False otherwise
        """
        return self.pipeline.remove_processor(processor_name)
    
    def get_pipeline(self) -> AssemblyPipeline:
        """
        Get the current assembly pipeline.
        
        Returns:
            The current AssemblyPipeline instance
        """
        return self.pipeline
    
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