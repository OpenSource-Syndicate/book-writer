"""
Test script to verify Ollama model integration with the book writer system
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from book_writer.config import model_config
from book_writer.model_manager import model_manager
from book_writer.outline import BookOutline, create_sample_outline
from book_writer.note_processor import NoteProcessor, ContentManager
from book_writer.content_expansion import ContentExpander


def test_model_connectivity():
    """Test if Ollama models are accessible."""
    print("Testing Ollama model connectivity...")
    
    if model_manager.health_check():
        print("✓ Ollama service is accessible")
        
        # List available models
        available_models = model_manager.list_available_models()
        print(f"Available models: {available_models}")
        
        return True
    else:
        print("✗ Ollama service is not accessible")
        print("Please make sure Ollama is running and accessible at the configured URL")
        return False


def test_content_expansion_model():
    """Test the content expansion model (stable-beluga:13b)."""
    print("\nTesting content expansion model...")
    
    # Create a simple test note
    test_note = "Artificial intelligence is a branch of computer science that aims to create software or machines that exhibit human-like intelligence."
    
    try:
        response = model_manager.generate_response(
            prompt=f"Expand this concept into a comprehensive paragraph: {test_note}",
            task="content_expansion",
            temperature=0.7,
            max_tokens=512
        )
        
        print(f"✓ Content expansion test completed")
        print(f"Response: {response[:200]}...")
        return True
    except Exception as e:
        print(f"✗ Content expansion test failed: {e}")
        return False


def test_outline_generation_model():
    """Test the outline generation model (deepseek-r1:8b)."""
    print("\nTesting outline generation model...")
    
    test_topic = "Machine Learning Fundamentals"
    
    try:
        response = model_manager.generate_response(
            prompt=f"Create a detailed book outline for '{test_topic}' with 2-3 parts, 3-4 chapters per part, and 2-3 subtopics per chapter.",
            task="outline_generation",
            format_json=True,
            temperature=0.5,
            max_tokens=1024
        )
        
        print(f"✓ Outline generation test completed")
        if isinstance(response, dict):
            print(f"Generated {len(response.get('parts', []))} parts")
        else:
            print(f"Response: {str(response)[:200]}...")
        return True
    except Exception as e:
        print(f"✗ Outline generation test failed: {e}")
        return False


def test_organization_model():
    """Test the organization model (deepseek-r1:8b)."""
    print("\nTesting organization model...")
    
    test_content = "This chapter discusses various machine learning algorithms including supervised, unsupervised, and reinforcement learning approaches."
    test_outline = {
        "parts": [
            {
                "id": "part1",
                "title": "Introduction",
                "chapters": [
                    {
                        "id": "chap1",
                        "title": "Getting Started with ML",
                        "subtopics": [
                            {"id": "sub1", "title": "Basic Concepts"}
                        ]
                    },
                    {
                        "id": "chap2", 
                        "title": "Machine Learning Algorithms",
                        "subtopics": [
                            {"id": "sub2", "title": "Supervised Learning"},
                            {"id": "sub3", "title": "Unsupervised Learning"}
                        ]
                    }
                ]
            }
        ]
    }
    
    try:
        prompt = f"""
        Analyze the following content and classify it into the most appropriate chapter and subtopic from the provided outline structure.
        
        Content to classify:
        {test_content}
        
        Book Outline Structure:
        {str(test_outline)}
        
        Please return the classification in the following JSON format:
        {{
            "chapter_id": "the ID of the most appropriate chapter",
            "chapter_title": "the title of the most appropriate chapter", 
            "subtopic_id": "the ID of the most appropriate subtopic",
            "subtopic_title": "the title of the most appropriate subtopic",
            "confidence": "a confidence score between 0 and 1"
        }}
        """
        
        response = model_manager.generate_response(
            prompt=prompt,
            task="organization",
            format_json=True,
            temperature=0.4
        )
        
        print(f"✓ Organization model test completed")
        if isinstance(response, dict):
            print(f"Classification: {response.get('chapter_title', 'N/A')} - {response.get('subtopic_title', 'N/A')}")
        else:
            print(f"Response: {str(response)[:200]}...")
        return True
    except Exception as e:
        print(f"✗ Organization model test failed: {e}")
        return False


def test_full_integration():
    """Test full integration with the book writer system."""
    print("\nTesting full integration...")
    
    try:
        # Create a temporary project directory
        test_project_path = project_root / "test_project"
        test_project_path.mkdir(exist_ok=True)
        
        # Create the necessary components
        note_processor = NoteProcessor(test_project_path)
        content_manager = ContentManager(test_project_path, note_processor)
        content_expander = ContentExpander(test_project_path, note_processor, content_manager)
        
        # Create a sample outline using AI
        print("Creating sample outline...")
        outline = create_sample_outline("Natural Language Processing")
        print(f"Created outline with {len(outline.parts)} parts")
        
        # Add a test note
        print("Adding test note...")
        note_id = note_processor.process_note(
            text="Transformers are a type of neural network architecture that has revolutionized natural language processing.",
            source="test",
            potential_topics=["NLP", "Machine Learning", "AI"]
        )
        print(f"Added note with ID: {note_id}")
        
        # Expand the note using the AI model
        print("Expanding note...")
        expanded_text, metadata = content_expander.expand_note(note_id, style="academic")
        print(f"Expanded text length: {len(expanded_text)} characters")
        
        # Classify the content
        print("Classifying content...")
        classification = content_expander.classify_content(expanded_text, outline.to_dict())
        print(f"Classification: Chapter={classification['chapter']['title'] if classification['chapter'] else 'N/A'}, Subtopic={classification['subtopic']['title'] if classification['subtopic'] else 'N/A'}")
        
        # Clean up
        import shutil
        shutil.rmtree(test_project_path)
        
        print("✓ Full integration test completed successfully")
        return True
    except Exception as e:
        print(f"✗ Full integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_functionality():
    """Test the updated RAG functionality with Ollama models."""
    print("\nTesting RAG functionality...")
    
    try:
        # Create a temporary project directory
        test_project_path = project_root / "test_project"
        test_project_path.mkdir(exist_ok=True)
        
        # Create the necessary components
        note_processor = NoteProcessor(test_project_path)
        content_manager = ContentManager(test_project_path, note_processor)
        rag_writer = RAGWriter(test_project_path, note_processor, content_manager)
        
        # Add a test note for retrieval
        test_note_id = note_processor.process_note(
            text="Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
            source="test_source",
            potential_topics=["AI", "Neural Networks", "Machine Learning"]
        )
        
        # Test context retrieval
        context = rag_writer.retrieve_context("neural networks")
        print(f"Retrieved {len(context.get('notes', []))} notes and {len(context.get('content', []))} content items")
        
        # Test content generation with context
        prompt = "Explain the concept of deep learning in simple terms"
        generated_content = rag_writer.generate_with_context(prompt, context)
        
        print(f"✓ RAG generation completed")
        print(f"Generated content length: {len(generated_content)} characters")
        
        # Clean up
        import shutil
        shutil.rmtree(test_project_path)
        
        return True
    except Exception as e:
        print(f"✗ RAG functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_function_calling():
    """Test the function calling capabilities."""
    print("\nTesting function calling...")
    
    try:
        # Register a simple test function
        def test_function(message: str) -> str:
            return f"Processed: {message}"
        
        from book_writer.tool_registry import tool_registry
        tool_registry.register_tool(
            name="test_function",
            description="A simple test function",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message to process"}
                },
                "required": ["message"]
            },
            function=test_function
        )
        
        # Test function execution
        result = model_manager._execute_single_function("test_function", {"message": "Hello, world!"})
        print(f"Function result: {result}")
        
        # Test with the RAG writer's registered tools
        result = model_manager._execute_single_function("search_notes", {"query": "test"})
        print(f"Search notes result: {result}")
        
        print("✓ Function calling test completed")
        return True
    except Exception as e:
        print(f"✗ Function calling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Starting Ollama model integration tests...\n")
    
    # Run connectivity test first
    if not test_model_connectivity():
        print("\nCannot proceed with other tests - Ollama not accessible")
        return
    
    # Run individual model tests
    results = []
    results.append(test_content_expansion_model())
    results.append(test_outline_generation_model())
    results.append(test_organization_model())
    results.append(test_full_integration())
    results.append(test_rag_functionality())
    results.append(test_function_calling())
    
    # Summary
    print(f"\nTest Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed.")


if __name__ == "__main__":
    main()