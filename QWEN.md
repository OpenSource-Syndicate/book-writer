# Book Writer System - QWEN.md

## Project Overview

The Book Writer System is a comprehensive book development application that helps authors create, organize, and refine book content through a structured workflow. The system combines traditional book writing processes with AI-powered content generation using local Ollama models.

### Key Features
- **Outline Creation**: Create hierarchical book outlines in YAML/JSON format with parts, chapters, and subtopics
- **Note Processing Pipeline**: Embed notes using BGE-M3 model and store in ChromaDB with metadata
- **AI-Powered Content Expansion**: Expand raw notes into polished text using AI models
- **Retrieval-Augmented Writing (RAG)**: Context-aware writing with Ollama models
- **AI Function Calling & Tool Integration**: Support for AI models to call specific tools and functions
- **Content Organization**: Cluster content, detect gaps, suggest reorganization, and visualize content structure
- **Book Assembly**: Merge chapters according to outline and generate complete manuscripts
- **Multiple Export Options**: Export to PDF and ePub formats
- **Dual Interface**: Interactive CLI and web-based Gradio UI

### Architecture

The system follows a modular architecture with clear separation of concerns:

- **app.py**: Main application interface that ties all components together
- **outline.py**: Handles book outline creation and management
- **note_processor.py**: Manages note processing and storage in ChromaDB
- **content_expansion.py**: Expands notes into polished content using AI models
- **rag_writing.py**: Implements RAG loop for context-aware writing
- **book_assembly.py**: Combines content into complete manuscripts
- **export.py**: Handles PDF and ePub export functionality
- **model_manager.py**: Manages different Ollama models for various tasks
- **organized_content.py**: Provides tools for content organization, including the `ContentOrganizer` and `ContentOrganization` classes.
- **ui.py**: Gradio web interface implementation
- **tool_registry.py**: Manages AI tool integration for function calling

## Building and Running

### Prerequisites
1. Install [Ollama](https://ollama.ai/) on your system
2. Pull the required models:
   ```bash
   ollama pull stable-beluga:13b
   ollama pull deepseek-r1:8b
   ```

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application

#### Interactive CLI Mode
```bash
python main.py
```

#### Web-based Gradio UI
```bash
python main.py --ui
```

### Configuration
Model configuration can be customized in `book_writer/model_config.yaml`:
- Content Expansion & RAG: `stable-beluga:13b` (optimized for generating rich, detailed content)
- Outline Generation & Organization: `deepseek-r1:8b` (optimized for structure and organization)

## Development Conventions

### AI Model Usage
- **Stable Beluga 13B**: Used for content generation tasks (expanding notes into polished text, RAG generation)
- **DeepSeek R1 8B**: Used for organizational tasks (outline generation, content classification)

### Tool Integration
The system includes a flexible tool registry that allows AI models to call specific functions:
- `search_notes`: Search for notes based on a query
- `classify_content`: Classify content into appropriate chapter and subtopic
- `expand_note`: Expand a note into detailed content
- `validate_outline`: Validate if an outline is properly structured

### Data Storage
- Notes and content are stored with embeddings in ChromaDB for semantic search
- Outlines are stored in YAML/JSON format in the `outlines/` directory
- Content is stored in markdown format in the `content/` directory
- Project configuration is stored in `book_config.yaml`

### Streaming Implementation
The system implements real-time streaming of Ollama output to the web UI. The `expand_note_stream_with_real_time` function in the UI progressively yields content as it's received from Ollama, allowing users to see responses as they are generated.

## Usage Examples

### Interactive CLI Commands
- `create_outline <title> [author] [description]`: Create a new book outline
- `add_note <text> [source] [topic1,topic2,...]`: Add a new note to the system
- `expand_note <note_id> [style]`: Expand a note into polished text
- `organize_content <content_id1> <content_id2> ...`: Organize content and get a summary
- `get_suggestions`: Get suggestions for content gaps and reorganization
- `visualize_content <output_path>`: Generate a visualization of the content structure
- `build_book`: Build the current book according to the outline
- `export_book <format>`: Export the book to PDF or ePub format

### Web Interface Features
- Project creation and loading
- Note adding with real-time AI expansion
- Writing style selection (academic, narrative, technical, conversational)
- Content editing capabilities
- Simplified content organization with a step-by-step process:
    - Organize content and get a summary
    - Review organization suggestions
    - Visualize content structure
- Outline visualization and management
- Book building and export functionality