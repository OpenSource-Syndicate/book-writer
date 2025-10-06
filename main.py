
"""
Main entry point for the Book Writer System
"""
import argparse
import os
from pathlib import Path

# Handle the readline issue on Windows before importing other modules
try:
    import readline
    # On Windows, the readline module may not have a backend attribute
    # which causes issues in Python's cmd module
    if not hasattr(readline, 'backend'):
        # If backend attribute doesn't exist, set a default value to avoid AttributeError
        readline.backend = 'builtin'  # or another appropriate default
except ImportError:
    # readline may not be available on all systems
    pass

from book_writer.app import BookWriterApp
from book_writer.ui import create_ui

def main():
    """Main function to run the Book Writer System."""
    parser = argparse.ArgumentParser(description="Book Writer System")
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the web-based Gradio UI instead of the interactive CLI."
    )
    parser.add_argument(
        "--project-path",
        type=str,
        default=os.getcwd(),
        help="Path to the book project directory."
    )
    args = parser.parse_args()

    project_path = Path(args.project_path)

    if args.ui:
        print("Launching Gradio Web UI...")
        # The UI manages its own project loading, so we just launch it
        web_ui = create_ui()
        web_ui.launch(share=True) # Using share=True for easy access
    else:
        print(f"Starting interactive CLI for project at: {project_path}")
        if not project_path.exists():
            print(f"Project path not found. Creating a new project at {project_path}")
            
            # Prompt for book title
            book_title = input("What is the title of your book? (default: 'My Book'): ").strip()
            if not book_title:
                book_title = "My Book"
            
            # Prompt for target number of pages
            target_pages_input = input("How many pages would you like to write for this book? (default: 100): ").strip()
            try:
                target_pages = int(target_pages_input) if target_pages_input else 100
            except ValueError:
                target_pages = 100
                print(f"Invalid input. Using default value of {target_pages} pages.")
            
            app = BookWriterApp.create_project(project_path, book_title=book_title, target_pages=target_pages)
        else:
            app = BookWriterApp(project_path)
        
        app.start_interactive_mode()

if __name__ == "__main__":
    main()