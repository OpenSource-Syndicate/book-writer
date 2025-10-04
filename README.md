# Book Writer System

A comprehensive book development system that helps authors create, organize, and refine book content through a structured workflow.

## Features

1. **Outline Creation**
   - Create hierarchical book outlines in YAML/JSON format
   - Structure books with parts, chapters, and subtopics
   - AI-powered outline generation using DeepSeek R1:8B model

2. **Note Processing Pipeline**
   - Embed notes using BGE-M3 model
   - Store in ChromaDB with metadata for organization

3. **Content Expansion**
   - Expand raw notes into polished text using Stable Beluga 13B model
   - Auto-classify content into appropriate sections using DeepSeek R1:8B model
   - Multiple writing styles supported (academic, narrative, technical, conversational)

4. **Retrieval-Augmented Writing (RAG)**
   - Advanced RAG loop for context-aware writing using Ollama models
   - Maintains content coherence throughout the book
   - Uses Stable Beluga 13B model for content generation

5. **AI Function Calling & Tool Integration**
   - Support for AI models to call specific tools and functions
   - Predefined tools for note searching, content classification, and outline validation
   - Extensible tool registry for adding custom functions

6. **Book Assembly**
   - Merge chapters according to outline
   - Generate complete Markdown manuscript
   - Support continuous refinement

7. **Export Options**
   - Export to PDF format
   - Export to ePub format

## Model Configuration

The system uses Ollama to run local language models. By default, it uses:

- **Content Expansion & RAG**: `stable-beluga:13b` - Optimized for generating rich, detailed content
- **Outline Generation & Organization**: `deepseek-r1:8b` - Optimized for structure and organization

### Prerequisites

1. Install [Ollama](https://ollama.ai/) on your system
2. Pull the required models:

```bash
ollama pull stable-beluga:13b
ollama pull deepseek-r1:8b
```

### Configuration

The model configuration can be customized in `book_writer/model_config.yaml`. The default configuration is:

```yaml
models:
  content_expansion:
    model_name: "stable-beluga:13b"
    api_type: "ollama"
    temperature: 0.7
    max_tokens: 1024
    top_p: 0.9
  outline_generation:
    model_name: "deepseek-r1:8b"
    api_type: "ollama"
    temperature: 0.5
    max_tokens: 512
    top_p: 0.8
  organization:
    model_name: "deepseek-r1:8b"
    api_type: "ollama"
    temperature: 0.4
    max_tokens: 512
    top_p: 0.8
ollama:
  base_url: "http://localhost:11434"
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode

```bash
python main.py
```

Follow the interactive prompts to create and manage your book project.

### Testing the Ollama Integration

To test if your Ollama models are properly configured:

```bash
python test_ollama_models.py
```

This will run comprehensive tests including:
- Model connectivity
- Content expansion functionality
- Outline generation
- Organization/classification
- Full integration
- RAG functionality
- Function calling capabilities

## Model-Specific Task Assignment

- **Stable Beluga 13B**: Used for content generation tasks (expanding notes into polished text, RAG generation)
- **DeepSeek R1 8B**: Used for organizational tasks (outline generation, content classification)

## AI Tool Integration

The system includes a flexible tool registry that allows AI models to call specific functions:

### Available Tools:
- `search_notes`: Search for notes based on a query
- `classify_content`: Classify content into appropriate chapter and subtopic
- `expand_note`: Expand a note into detailed content
- `validate_outline`: Validate if an outline is properly structured

### Adding Custom Tools:
You can register custom tools with the `ToolRegistry` to extend the system's capabilities.

This division allows the system to leverage the strengths of each model for the most appropriate tasks.