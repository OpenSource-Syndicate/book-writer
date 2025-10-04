
"""
Book Writer System - Gradio Web UI
Provides a web-based interface for the Book Writer application.
"""
import gradio as gr
import os
from pathlib import Path
import yaml

from book_writer.app import BookWriterApp

# --- State Management ---

def get_project_path(app_state):
    if app_state and app_state.project_path:
        return str(app_state.project_path)
    return "No project loaded."

def get_outline_display(app_state):
    if app_state and app_state.current_outline:
        return yaml.dump(app_state.current_outline.to_dict(), allow_unicode=True, default_flow_style=False, sort_keys=False)
    return "No outline loaded."

# --- UI Event Handlers ---

def create_or_load_project(project_path_str: str):
    """Creates or loads a project."""
    project_path = Path(project_path_str)
    if not project_path.exists():
        gr.Info(f"Project path not found. Creating new project at: {project_path}")
        app = BookWriterApp.create_project(project_path)
    else:
        gr.Info(f"Loading existing project from: {project_path}")
        app = BookWriterApp(project_path)
    
    outline_display = get_outline_display(app)
    project_path_display = get_project_path(app)
    
    return app, project_path_display, outline_display

async def add_note_async(app_state, note_text, source):
    """Asynchronously adds a new note."""
    if not app_state:
        gr.Warning("Please load a project first.")
        return None, "No project loaded."
    if not note_text:
        gr.Warning("Note text cannot be empty.")
        return app_state, "Note text was empty."

    gr.Info("Processing note...")
    note_id = app_state.process_note(note_text, source)
    gr.Success(f"Note added successfully! ID: {note_id}")
    return app_state, f"Note added: {note_id}"

def expand_note_stream_with_real_time(app_state, note_id, style):
    """Streams the expansion of a note into full content in real-time."""
    if not app_state:
        gr.Warning("Please load a project first.")
        yield app_state, None, "No project loaded."
        return
    if not note_id:
        gr.Warning("Note ID is required.")
        yield app_state, None, "Note ID is required."
        return

    gr.Info(f"Expanding note {note_id} (streaming)...")
    
    try:
        # Get the note first
        results = app_state.note_processor.notes_collection.get(
            ids=[note_id],
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            raise ValueError(f"Note with ID {note_id} not found")
        
        note_text = results["documents"][0]
        metadata = results["metadatas"][0]
        
        # Create prompt
        expander = app_state.content_expander
        prompt = expander._create_expansion_prompt(note_text, style)
        
        # Get model configuration
        model_cfg = app_state.content_expander.model_manager.config.get_model_config("content_expansion")
        model_name = model_cfg["model_name"]
        
        # Initialize the streaming response
        full_response = ""
        
        # Create the stream with the Ollama client
        stream = app_state.content_expander.model_manager.ollama_client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.7,
                "top_p": 0.9,
            },
            stream=True
        )
        
        # Process the stream and yield content as it comes in
        for chunk in stream:
            if chunk.get('done', False):
                break
            content = chunk.get('message', {}).get('content', '')
            if content:
                full_response += content
                # Yield the accumulated content immediately to show streaming
                yield app_state, None, full_response
        
        # Now classify and store the content after full generation
        if not app_state.current_outline:
            raise ValueError("No outline loaded. Please load or create an outline first.")
        
        classification = expander.classify_content(full_response, app_state.current_outline.to_dict())
        
        if not classification["chapter"] or not classification["subtopic"]:
            raise ValueError("Could not classify content. Please specify chapter and subtopic manually.")
        
        # Store the expanded content
        content_id = app_state.content_manager.store_content(
            content=full_response,
            title=f"Expanded note from {metadata.get('source', 'unknown')}",
            chapter_id=classification["chapter"]["id"],
            subtopic_id=classification["subtopic"]["id"],
            source_note_ids=[note_id]
        )
        
        gr.Success(f"Content generated! ID: {content_id}")
        # Yield the final result with content ID
        yield app_state, content_id, full_response
        
    except Exception as e:
        gr.Error(f"Failed to expand note: {e}")
        yield app_state, None, f"Error: {e}"

async def save_edited_content_async(app_state, current_content_id, new_text):
    """Asynchronously saves the edited content."""
    if not app_state:
        gr.Warning("Please load a project first.")
        return
    if not current_content_id:
        gr.Warning("No content is currently loaded in the editor.")
        return

    gr.Info(f"Saving changes to content ID: {current_content_id}...")
    try:
        app_state.content_manager.update_content(current_content_id, new_text)
        gr.Success("Changes saved successfully!")
    except Exception as e:
        gr.Error(f"Failed to save changes: {e}")

async def build_and_export_async(app_state, export_format):
    """Asynchronously builds and exports the book."""
    if not app_state:
        gr.Warning("Please load a project first.")
        return None, None
    
    gr.Info(f"Building and exporting book to {export_format}...")
    try:
        export_path = app_state.export_book(export_format)
        gr.Success(f"Book exported successfully!")
        return app_state, export_path
    except Exception as e:
        gr.Error(f"Failed to export book: {e}")
        return app_state, None

# --- UI Layout ---

def create_ui():
    """Creates the Gradio UI for the Book Writer application."""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Book Writer Pro") as ui:
        app_state = gr.State()
        current_content_id_state = gr.State()

        gr.Markdown("# 📖 Book Writer Pro")
        gr.Markdown("An AI-powered assistant to help you write, organize, and export your book.")

        with gr.Tabs():
            with gr.TabItem("🚀 Project"):
                gr.Markdown("## Project Management")
                project_path_input = gr.Textbox(label="Project Directory", placeholder="/path/to/your/book_project", lines=1)
                load_project_btn = gr.Button("Create or Load Project", variant="primary")
                project_path_display = gr.Textbox(label="Current Project Path", interactive=False)

            with gr.TabItem("✍️ Writer's Desk"):
                gr.Markdown("## AI-Powered Content Generation")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Step 1: Add a Note")
                        note_text_input = gr.Textbox(label="Your Note", placeholder="e.g., 'The protagonist discovers a hidden map.'", lines=5)
                        note_source_input = gr.Textbox(label="Source (optional)", value="web-ui", lines=1)
                        add_note_btn = gr.Button("Add Note", variant="secondary")
                        note_status_output = gr.Textbox(label="Status", interactive=False)
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### Step 2: Expand Note into Content")
                        expand_note_id_input = gr.Textbox(label="Note ID to Expand", placeholder="Enter the note ID from the status above", lines=1)
                        writing_style_dd = gr.Dropdown(label="Writing Style", choices=["academic", "narrative", "technical", "conversational"], value="narrative")
                        expand_note_btn = gr.Button("Expand Note", variant="primary")
                        
                gr.Markdown("### Generated Content (Editable)")
                content_output_and_edit = gr.Textbox(label="Content", lines=20, interactive=True, placeholder="Generated content will appear here...")
                save_content_btn = gr.Button("Save Changes", variant="primary")
                save_content_btn.click(
                    fn=save_edited_content_async, 
                    inputs=[app_state, current_content_id_state, content_output_and_edit], 
                    outputs=None
                )

            with gr.TabItem("📚 Outline & Assembly"):
                gr.Markdown("## Book Outline and Export")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Current Outline")
                        outline_display = gr.Textbox(label="Outline (YAML)", lines=30, interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown("### Build & Export")
                        export_format_dd = gr.Dropdown(label="Export Format", choices=["pdf", "epub"], value="pdf")
                        export_btn = gr.Button("Build and Export Book", variant="primary")
                        exported_file_output = gr.File(label="Download Your Book")
                        
        # --- Event Wiring ---
        load_project_btn.click(
            fn=create_or_load_project,
            inputs=[project_path_input],
            outputs=[app_state, project_path_display, outline_display]
        )
        
        add_note_btn.click(
            fn=add_note_async,
            inputs=[app_state, note_text_input, note_source_input],
            outputs=[app_state, note_status_output]
        )
        
        expand_note_btn.click(
            fn=expand_note_stream_with_real_time,
            inputs=[app_state, expand_note_id_input, writing_style_dd],
            outputs=[app_state, current_content_id_state, content_output_and_edit]
        )
        
        export_btn.click(
            fn=build_and_export_async,
            inputs=[app_state, export_format_dd],
            outputs=[app_state, exported_file_output]
        )

    return ui

if __name__ == "__main__":
    web_ui = create_ui()
    web_ui.launch()
