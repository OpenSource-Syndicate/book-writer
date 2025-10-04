"""
Book Writer System - Modular Assembly Module
Provides a flexible, pipeline-based approach to book assembly
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator

import markdown
from tqdm import tqdm

from book_writer.note_processor import ContentManager
from book_writer.outline import BookOutline
from book_writer.book_assembly import (AssemblyConfig, AssemblyContext, ContentProcessor, 
                                      AssemblyPipeline, TitleProcessor, TOCProcessor)


class FormattingProcessor(ContentProcessor):
    """
    Applies formatting and styling to the assembled content.
    """
    
    def process(self, context: AssemblyContext) -> AssemblyContext:
        # Apply any formatting transformations if needed
        # This is a placeholder for future formatting enhancements
        return context


class ModularBookAssembler:
    """
    A more modular version of the book assembler using the pipeline architecture.
    Maintains compatibility with the original BookAssembler interface.
    """
    
    def __init__(self, project_path: Union[str, Path], content_manager: ContentManager, 
                 default_config: AssemblyConfig = None):
        """
        Initialize a new modular book assembler.
        
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
                   custom_pipeline: AssemblyPipeline = None, 
                   config: AssemblyConfig = None) -> Path:
        """
        Build a complete book manuscript from the outline and content.
        
        Args:
            outline: The book outline
            output_format: The output format (markdown, html)
            custom_pipeline: Optional custom pipeline to use instead of default
            config: Optional assembly configuration (uses default if not provided)
            
        Returns:
            The path to the built book
        """
        print(f"Building book: {outline.title}")
        
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
        pipeline = custom_pipeline or self.pipeline
        context = AssemblyContext(outline, assembly_config)
        
        for processor in pipeline.processors:
            context = processor.process(context)
        
        # Join all content segments
        full_content = "".join(context.content)
        
        # Write the content to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            if output_format.lower() == "html":
                html_content = markdown.markdown(full_content)
                f.write(html_content)
            else:
                f.write(full_content)
        
        print(f"Book built successfully: {output_file}")
        return output_file
    
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