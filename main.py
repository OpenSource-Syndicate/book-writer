
"""
Main entry point for the Book Writer System
"""
import argparse
import os
from pathlib import Path

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
            app = BookWriterApp.create_project(project_path)
        else:
            app = BookWriterApp(project_path)
        
        app.start_interactive_mode()

if __name__ == "__main__":
    main()