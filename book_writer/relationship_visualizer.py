"""
Book Writer System - Relationship Visualization Module
Provides tools to visualize content relationships in text format and generate simple visualizations
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import defaultdict

from book_writer.outline import BookOutline
from book_writer.note_processor import ContentManager
from book_writer.organized_content import ContentOrganizer, ContentRelationship


class RelationshipVisualizer:
    """
    Visualizes relationships between content elements in the book.
    Provides both text-based and simple ASCII visualizations.
    """
    
    def __init__(self, 
                 project_path: Union[str, Path], 
                 content_manager: ContentManager,
                 content_organizer: ContentOrganizer):
        """
        Initialize the relationship visualizer.
        
        Args:
            project_path: The path to the project directory
            content_manager: Content manager instance
            content_organizer: Content organizer instance
        """
        self.project_path = Path(project_path)
        self.content_manager = content_manager
        self.content_organizer = content_organizer
        self.viz_dir = self.project_path / "visualizations"
        self.viz_dir.mkdir(exist_ok=True, parents=True)
    
    def visualize_outline_structure(self, outline: BookOutline) -> str:
        """
        Visualizes the hierarchical structure of the book outline.
        
        Args:
            outline: The book outline to visualize
            
        Returns:
            String representation of the outline structure
        """
        viz = []
        viz.append(f"Book: {outline.title}")
        viz.append(f"Author: {outline.author}")
        if outline.description:
            viz.append(f"Description: {outline.description}")
        viz.append("")
        
        for part_idx, part in enumerate(outline.parts, 1):
            viz.append(f"Part {part_idx}: {part['title']}")
            if part.get('description'):
                viz.append(f"  Description: {part['description']}")
            
            for chapter_idx, chapter in enumerate(part['chapters'], 1):
                viz.append(f"  Chapter {part_idx}.{chapter_idx}: {chapter['title']}")
                if chapter.get('description'):
                    viz.append(f"    Description: {chapter['description']}")
                
                for subtopic_idx, subtopic in enumerate(chapter['subtopics'], 1):
                    viz.append(f"    Subtopic {part_idx}.{chapter_idx}.{subtopic_idx}: {subtopic['title']}")
                    if subtopic.get('description'):
                        viz.append(f"      Description: {subtopic['description']}")
                    
                    # Show content count for this subtopic
                    content_items = self.content_manager.retrieve_content_by_subtopic(subtopic['id'])
                    viz.append(f"      Content items: {len(content_items)}")
                    
                    # Show content titles if available
                    for item in content_items[:3]:  # Show first 3 items
                        title = item['metadata'].get('title', 'Untitled')
                        viz.append(f"        - {title[:50]}{'...' if len(title) > 50 else ''}")
                    
                    if len(content_items) > 3:
                        viz.append(f"        ... and {len(content_items) - 3} more")
                    
                    viz.append("")
        
        return "\n".join(viz)
    
    def visualize_content_relationships(self, outline: BookOutline) -> str:
        """
        Visualizes relationships between content elements.
        
        Args:
            outline: The book outline
            
        Returns:
            String representation of content relationships
        """
        relationships = self.content_organizer.relationships
        if not relationships:
            return "No explicit relationships defined in the content organization."
        
        viz = []
        viz.append("Content Relationships:")
        viz.append("=" * 20)
        
        # Group relationships by type
        relationships_by_type = defaultdict(list)
        for rel in relationships:
            relationships_by_type[rel.relationship_type].append(rel)
        
        for rel_type, rel_list in relationships_by_type.items():
            viz.append(f"\n{rel_type.upper()} RELATIONSHIPS:")
            viz.append("-" * (len(rel_type) + 13))
            
            for rel in rel_list:
                try:
                    # Get source content
                    source_content = self.content_manager.get_content(rel.source_id)
                    source_preview = source_content[:30] + "..." if len(source_content) > 30 else source_content
                    
                    # Get target content
                    target_content = self.content_manager.get_content(rel.target_id)
                    target_preview = target_content[:30] + "..." if len(target_content) > 30 else target_content
                    
                    viz.append(f"  From: {rel.source_id[:8]}... -> To: {rel.target_id[:8]}... (strength: {rel.strength:.2f})")
                    viz.append(f"    \"{source_preview}\"")
                    viz.append(f"    -> \"{target_preview}\"")
                    viz.append("")
                except FileNotFoundError:
                    # Handle cases where content may have been deleted
                    viz.append(f"  From: {rel.source_id[:8]}... -> To: {rel.target_id[:8]}... (strength: {rel.strength:.2f})")
                    viz.append(f"    [Content not found - may have been deleted]")
                    viz.append("")
        
        return "\n".join(viz)
    
    def visualize_content_clusters(self) -> str:
        """
        Visualizes content clusters identified by the content organizer.
        
        Returns:
            String representation of content clusters
        """
        clusters = self.content_organizer.clusters
        if not clusters:
            return "No content clusters identified."
        
        viz = []
        viz.append("Content Clusters:")
        viz.append("=" * 16)
        
        for cluster in clusters:
            viz.append(f"\nCluster: {cluster.title}")
            viz.append(f"Description: {cluster.description}")
            viz.append(f"Topic Keywords: {', '.join(cluster.topic_keywords[:5])}")  # Show first 5 keywords
            viz.append(f"Confidence: {cluster.confidence:.2f}")
            viz.append(f"Content Items: {len(cluster.content_ids)}")
            
            # Show previews of content in the cluster
            for content_id in cluster.content_ids[:3]:  # Show first 3 items
                try:
                    content = self.content_manager.get_content(content_id)
                    preview = content[:50] + "..." if len(content) > 50 else content
                    viz.append(f"  - {preview}")
                except FileNotFoundError:
                    viz.append(f"  - [Content {content_id[:8]}... not found]")
            
            if len(cluster.content_ids) > 3:
                viz.append(f"  ... and {len(cluster.content_ids) - 3} more")
        
        return "\n".join(viz)
    
    def visualize_content_flow(self, outline: BookOutline) -> str:
        """
        Visualizes the logical flow of content through the book.
        
        Args:
            outline: The book outline
            
        Returns:
            String representation of content flow
        """
        viz = []
        viz.append("Content Flow Analysis:")
        viz.append("=" * 21)
        
        # Analyze organization metrics
        metrics = self.content_organizer.calculate_organization_metrics(outline)
        
        viz.append(f"Completeness Score: {metrics.completeness_score:.2f}")
        viz.append(f"Flow Score: {metrics.flow_score:.2f}")
        viz.append(f"Cohesion Score: {metrics.cohesion_score:.2f}")
        viz.append(f"Detected Gaps: {metrics.gap_count}")
        viz.append(f"Reorganization Suggestions: {metrics.suggestions_count}")
        
        # Show reorganization suggestions if any
        suggestions = self.content_organizer.suggest_content_reorganization(outline)
        if suggestions:
            viz.append("\nReorganization Suggestions:")
            viz.append("-" * 26)
            
            for suggestion in suggestions[:5]:  # Show first 5 suggestions
                viz.append(f"- Content '{suggestion['content_id'][:8]}...' in '{suggestion['current_location']}'")
                viz.append(f"  Suggested move to: '{suggestion['suggested_location']}'")
                viz.append(f"  Reason: {suggestion['reason']}")
                viz.append("")
        
        return "\n".join(viz)
    
    def generate_summary_report(self, outline: BookOutline) -> str:
        """
        Generates a comprehensive summary report of the book's organization.
        
        Args:
            outline: The book outline
            
        Returns:
            Complete organization summary report
        """
        report = []
        report.append("=" * 60)
        report.append("BOOK ORGANIZATION SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append(self.visualize_outline_structure(outline))
        report.append("")
        
        report.append(self.visualize_content_clusters())
        report.append("")
        
        report.append(self.visualize_content_relationships(outline))
        report.append("")
        
        report.append(self.visualize_content_flow(outline))
        report.append("")
        
        # Show organization metrics
        metrics = self.content_organizer.calculate_organization_metrics(outline)
        report.append("QUICK STATS:")
        report.append("-" * 12)
        report.append(f"- Completeness: {metrics.completeness_score:.1%}")
        report.append(f"- Content Flow: {metrics.flow_score:.1%}")
        report.append(f"- Cohesion: {metrics.cohesion_score:.1%}")
        report.append(f"- Gaps: {metrics.gap_count}")
        report.append(f"- Suggestions: {metrics.suggestions_count}")
        
        report.append("")
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_visualization(self, name: str, content: str) -> Path:
        """
        Saves a visualization to a file.
        
        Args:
            name: Name for the visualization (will be used in filename)
            content: Content to save
            
        Returns:
            Path to the saved file
        """
        filename = self.viz_dir / f"{name.replace(' ', '_').lower()}_visualization.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return filename
    
    def export_outline_as_tree(self, outline: BookOutline) -> str:
        """
        Exports the outline structure as a tree-like text visualization.
        
        Args:
            outline: The book outline to visualize
            
        Returns:
            Tree-like string representation of the outline
        """
        tree = []
        tree.append(outline.title)
        
        for part_idx, part in enumerate(outline.parts, 1):
            tree.append(f"├── Part {part_idx}: {part['title']}")
            
            for chapter_idx, chapter in enumerate(part['chapters'], 1):
                is_last_chapter = (chapter_idx == len(part['chapters']))
                chapter_prefix = "└──" if is_last_chapter else "├──"
                tree.append(f"│   {chapter_prefix} Chapter {chapter_idx}: {chapter['title']}")
                
                for subtopic_idx, subtopic in enumerate(chapter['subtopics'], 1):
                    is_last_subtopic = (subtopic_idx == len(chapter['subtopics']))
                    subtopic_prefix = "└──" if is_last_subtopic else "├──"
                    
                    # Determine the vertical bar pattern based on the level
                    if not is_last_chapter:
                        tree.append(f"│   │   {subtopic_prefix} {subtopic['title']}")
                    else:
                        tree.append(f"│       {subtopic_prefix} {subtopic['title']}")
            
            # Add a separator line after each part except the last one
            if part_idx < len(outline.parts):
                tree.append("│")
        
        return "\n".join(tree)