"""
Book Writer System - Export Module
Handles export of book manuscripts to various formats
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import ebooklib
from ebooklib import epub
import markdown
import re
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


class BookExporter:
    """Class for exporting book manuscripts to various formats."""
    
    def __init__(self, project_path: Union[str, Path]):
        """Initialize a new book exporter.
        
        Args:
            project_path: The path to the project directory
        """
        self.project_path = Path(project_path)
        self.exports_dir = self.project_path / "exports"
        self.exports_dir.mkdir(exist_ok=True, parents=True)
    
    def export_to_pdf(self, markdown_path: Union[str, Path]) -> Path:
        """Export a markdown manuscript to PDF.
        
        Args:
            markdown_path: The path to the markdown manuscript
            
        Returns:
            The path to the exported PDF
        """
        markdown_path = Path(markdown_path)
        
        # Read the markdown content
        with open(markdown_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        
        # Create output path
        pdf_path = self.exports_dir / f"{markdown_path.stem}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=72)
        
        # Get sample styles and customize
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.black,
        )
        
        heading1_style = ParagraphStyle(
            'CustomH1',
            parent=styles['Heading1'],
            fontSize=18,
            spaceBefore=24,
            spaceAfter=12,
            keepWithNext=True,
        )
        
        heading2_style = ParagraphStyle(
            'CustomH2',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
        )
        
        heading3_style = ParagraphStyle(
            'CustomH3',
            parent=styles['Heading3'],
            fontSize=14,
            spaceBefore=16,
            spaceAfter=8,
        )
        
        normal_style = ParagraphStyle(
            'JustifiedBody',
            parent=styles['Normal'],
            fontSize=12,
            leading=18,  # line height
            alignment=4,  # Justified alignment
            spaceAfter=12,
        )
        
        elements = []
        
        # Process markdown content line by line
        lines = markdown_content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('# '):  # H1
                elements.append(Paragraph(line[2:], title_style))
                elements.append(PageBreak())
            elif line.startswith('## '):  # H2
                elements.append(Paragraph(line[3:], heading1_style))
            elif line.startswith('### '):  # H3
                elements.append(Paragraph(line[4:], heading2_style))
            elif line.startswith('#### '):  # H4
                elements.append(Paragraph(line[5:], heading3_style))
            elif line.startswith('```'):  # Code block
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    if i < len(lines):
                        code_lines.append(lines[i])
                    i += 1
                code_content = '\n'.join(code_lines)
                elements.append(Preformatted(code_content, styles['Code']))
            elif line == '':  # Empty line
                elements.append(Spacer(1, 12))
            else:  # Regular paragraph
                # Add support for basic markdown formatting
                formatted_line = self._format_markdown_line(line)
                elements.append(Paragraph(formatted_line, normal_style))
            
            i += 1
        
        # Build the PDF
        doc.build(elements)
        
        print(f"PDF exported successfully: {pdf_path}")
        return pdf_path
    
    def _format_markdown_line(self, line: str) -> str:
        """Convert basic markdown formatting to ReportLab-compatible HTML."""
        import re
        
        # Replace bold formatting
        line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
        line = re.sub(r'__(.*?)__', r'<b>\1</b>', line)
        
        # Replace italic formatting
        line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', line)
        line = re.sub(r'_(.*?)_', r'<i>\1</i>', line)
        
        # Replace links
        line = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', line)
        
        return line
    
    def export_to_epub(self, markdown_path: Union[str, Path]) -> Path:
        """Export a markdown manuscript to ePub.
        
        Args:
            markdown_path: The path to the markdown manuscript
            
        Returns:
            The path to the exported ePub
        """
        markdown_path = Path(markdown_path)
        
        # Read the markdown content
        with open(markdown_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        
        # Create a new ePub book
        book = epub.EpubBook()
        
        # Set metadata
        book.set_identifier(f"bookwriter-{markdown_path.stem}")
        book.set_title(markdown_path.stem.replace('_', ' ').title())
        book.set_language('en')
        
        # Parse the markdown to extract title, author, and chapters
        lines = markdown_content.split('\n')
        title = lines[0].strip('# ') if lines and lines[0].startswith('# ') else markdown_path.stem
        author = lines[1].strip('## By ') if len(lines) > 1 and lines[1].startswith('## By ') else "Unknown Author"
        
        book.add_author(author)
        
        # Create CSS style
        style = '''
        @namespace epub "http://www.idpf.org/2007/ops";
        body {
            font-family: Georgia, serif;
            line-height: 1.6;
            margin: 5%;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: Arial, sans-serif;
            margin-top: 2em;
            margin-bottom: 0.5em;
        }
        h1 {
            font-size: 2em;
            text-align: center;
        }
        h2 {
            font-size: 1.5em;
        }
        h3 {
            font-size: 1.2em;
        }
        p {
            margin-bottom: 1em;
        }
        '''
        
        # Add CSS file
        nav_css = epub.EpubItem(
            uid="style_nav",
            file_name="style/nav.css",
            media_type="text/css",
            content=style
        )
        book.add_item(nav_css)
        
        # Create chapters
        chapters = []
        current_chapter = []
        chapter_title = "Introduction"
        
        for line in lines:
            if line.startswith('## Chapter'):
                # Save previous chapter
                if current_chapter:
                    chapter_content = '\n'.join(current_chapter)
                    chapter_html = markdown.markdown(chapter_content, extensions=['tables', 'fenced_code'])
                    
                    c = epub.EpubHtml(
                        title=chapter_title,
                        file_name=f"chapter_{len(chapters)}.xhtml",
                        lang='en'
                    )
                    c.content = f'<html><head><link rel="stylesheet" href="style/nav.css" /></head><body>{chapter_html}</body></html>'
                    book.add_item(c)
                    chapters.append(c)
                    
                    current_chapter = []
                
                # Start new chapter
                chapter_title = line.strip('## ')
                current_chapter.append(f"# {chapter_title}")
            else:
                current_chapter.append(line)
        
        # Add the last chapter
        if current_chapter:
            chapter_content = '\n'.join(current_chapter)
            chapter_html = markdown.markdown(chapter_content, extensions=['tables', 'fenced_code'])
            
            c = epub.EpubHtml(
                title=chapter_title,
                file_name=f"chapter_{len(chapters)}.xhtml",
                lang='en'
            )
            c.content = f'<html><head><link rel="stylesheet" href="style/nav.css" /></head><body>{chapter_html}</body></html>'
            book.add_item(c)
            chapters.append(c)
        
        # Add chapters to the book
        book.toc = chapters
        
        # Add navigation files
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        # Define the book spine
        book.spine = ['nav'] + chapters
        
        # Create output path
        epub_path = self.exports_dir / f"{markdown_path.stem}.epub"
        
        # Write the ePub file
        epub.write_epub(str(epub_path), book, {})
        
        print(f"ePub exported successfully: {epub_path}")
        return epub_path