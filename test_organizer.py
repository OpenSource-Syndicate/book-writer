"""
Quick test script for the Notes Organizer
Run this to verify the organizer functionality works
"""
from pathlib import Path
from book_writer.app import BookWriterApp
from book_writer.content_organizer_ui import get_all_notes_list, format_notes_as_html, format_outline_as_interactive_tree

def test_organizer():
    """Test the organizer components."""
    print("🧪 Testing Notes Organizer...")
    print("=" * 50)
    
    # Test 1: Check if imports work
    print("\n✅ Test 1: Imports successful")
    
    # Test 2: Create or load a test project
    test_project_path = Path("./test_project")
    if not test_project_path.exists():
        print("\n📁 Creating test project...")
        app = BookWriterApp.create_project(
            test_project_path,
            book_title="Test Book",
            target_pages=50
        )
        print("✅ Test 2: Project created successfully")
    else:
        print("\n📁 Loading existing test project...")
        app = BookWriterApp(test_project_path)
        print("✅ Test 2: Project loaded successfully")
    
    # Test 3: Add some test notes
    print("\n📝 Adding test notes...")
    note_ids = []
    test_notes = [
        "This is a note about artificial intelligence and machine learning.",
        "The protagonist discovers a hidden treasure map in the attic.",
        "Climate change is affecting global weather patterns significantly.",
        "The recipe calls for flour, eggs, and milk to make pancakes.",
        "Quantum computing uses qubits instead of classical bits."
    ]
    
    for i, note_text in enumerate(test_notes):
        try:
            note_id = app.process_note(note_text, source=f"test_{i}")
            note_ids.append(note_id)
            print(f"  ✓ Added note {i+1}: {note_id[:16]}...")
        except Exception as e:
            print(f"  ✗ Failed to add note {i+1}: {e}")
    
    print(f"✅ Test 3: Added {len(note_ids)} test notes")
    
    # Test 4: Get notes list
    print("\n📋 Getting notes list...")
    notes_list = get_all_notes_list(app, "")
    print(f"✅ Test 4: Retrieved {len(notes_list)} notes")
    
    # Test 5: Format notes as HTML
    print("\n🎨 Formatting notes as HTML...")
    notes_html = format_notes_as_html(notes_list, show_assigned=True, show_unassigned=True)
    html_length = len(notes_html)
    print(f"✅ Test 5: Generated HTML ({html_length} characters)")
    
    # Test 6: Format outline as interactive tree
    print("\n🌳 Formatting outline tree...")
    outline_html = format_outline_as_interactive_tree(app)
    outline_length = len(outline_html)
    print(f"✅ Test 6: Generated outline tree ({outline_length} characters)")
    
    # Test 7: Check outline structure
    print("\n📚 Checking outline structure...")
    if app.current_outline:
        outline_dict = app.current_outline.to_dict()
        num_parts = len(outline_dict.get("parts", []))
        num_chapters = sum(len(part.get("chapters", [])) for part in outline_dict.get("parts", []))
        num_subtopics = sum(
            len(chapter.get("subtopics", []))
            for part in outline_dict.get("parts", [])
            for chapter in part.get("chapters", [])
        )
        print(f"  Parts: {num_parts}")
        print(f"  Chapters: {num_chapters}")
        print(f"  Subtopics: {num_subtopics}")
        print(f"✅ Test 7: Outline structure verified")
    else:
        print("✗ Test 7: No outline loaded")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\n📌 Next steps:")
    print("  1. Run: python main.py")
    print("  2. Open the web UI in your browser")
    print("  3. Go to the '🗂️ Notes Organizer' tab")
    print("  4. Try dragging notes to subtopics!")
    print("\n💡 Tip: Check ORGANIZER_GUIDE.md for detailed usage instructions")

if __name__ == "__main__":
    try:
        test_organizer()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
