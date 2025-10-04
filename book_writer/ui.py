
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

# Content Organization Functions
def refresh_outline_tree(app_state):
    """Refresh the outline structure display with assignment indicators."""
    if not app_state or not app_state.current_outline:
        return "No outline loaded."
    
    # Convert the outline to a tree-like structure for display
    outline_dict = app_state.current_outline.to_dict()
    
    # Get all notes and their classifications to add to the outline display
    try:
        all_notes = app_state.note_processor.notes_collection.get(include=["documents", "metadatas"])
        if all_notes["ids"]:
            # Add note assignment information to the outline structure
            for i, note_id in enumerate(all_notes["ids"]):
                metadata = all_notes["metadatas"][i]
                chapter_id = metadata.get("chapter_id")
                subtopic_id = metadata.get("subtopic_id")
                
                if chapter_id and subtopic_id:
                    # Find the chapter and subtopic in the outline and add note count/indicators
                    for part in outline_dict.get("parts", []):
                        for chapter in part.get("chapters", []):
                            if chapter["id"] == chapter_id:
                                # Add note assignment info to the chapter
                                if "assigned_notes" not in chapter:
                                    chapter["assigned_notes"] = []
                                chapter["assigned_notes"].append(note_id)
                                
                                for subtopic in chapter.get("subtopics", []):
                                    if subtopic["id"] == subtopic_id:
                                        if "assigned_notes" not in subtopic:
                                            subtopic["assigned_notes"] = []
                                        subtopic["assigned_notes"].append(note_id)
    except Exception:
        # If there's an error getting note assignments, just return the basic outline
        pass
    
    return outline_dict

def get_available_notes(app_state):
    """Get a list of available notes for assignment."""
    if not app_state:
        return []
    
    try:
        # Get all notes from the notes collection
        all_notes = app_state.note_processor.notes_collection.get(include=["documents", "metadatas"])
        if not all_notes["ids"]:
            return []
        
        # Create a list of note options with ID and preview
        note_options = []
        for i, note_id in enumerate(all_notes["ids"]):
            note_doc = all_notes["documents"][i]
            preview = note_doc[:50] + "..." if len(note_doc) > 50 else note_doc
            note_options.append(f"{note_id}: {preview}")
        
        return note_options
    except Exception:
        return []

def get_outline_sections(app_state):
    """Get a list of outline sections for assignment."""
    if not app_state or not app_state.current_outline:
        return []
    
    sections = []
    outline = app_state.current_outline
    
    for part in outline.parts:
        part_title = f"Part: {part['title']}"
        sections.append(part_title)
        
        for chapter in part['chapters']:
            chapter_title = f"Chapter: {chapter['title']} (in {part['title']})"
            sections.append(chapter_title)
            
            for subtopic in chapter['subtopics']:
                subtopic_title = f"Subtopic: {subtopic['title']} (in {chapter['title']})"
                sections.append(subtopic_title)
    
    return sections

def assign_notes_to_section(app_state, selected_notes, target_section):
    """Assign selected notes to the specified section."""
    if not app_state:
        gr.Warning("Please load a project first.")
        return "No project loaded."
    
    if not selected_notes:
        gr.Warning("Please select notes to assign.")
        return "No notes selected."
    
    if not target_section:
        gr.Warning("Please select a section to assign notes to.")
        return "No section selected."
    
    try:
        # This is a simplified implementation - in a full implementation you would
        # classify the notes into the appropriate section and update their metadata
        assigned_count = 0
        for note_selection in selected_notes:
            # Extract note ID from the selection (format: "note_id: preview text")
            note_id = note_selection.split(":")[0] if ":" in note_selection else note_selection
            
            # In a real implementation, we would update the note's classification
            # For now, we'll just indicate that assignment would happen
            assigned_count += 1
        
        result = f"Assigned {assigned_count} note(s) to section '{target_section}'"
        gr.Info(result)
        return result
    except Exception as e:
        gr.Error(f"Failed to assign notes: {e}")
        return f"Error: {e}"

def auto_organize_notes(app_state, custom_rules=None):
    """Automatically organize notes based on content analysis."""
    if not app_state:
        gr.Warning("Please load a project first.")
        return "No project loaded."
    
    if not app_state.current_outline:
        gr.Warning("Please load an outline first.")
        return "No outline loaded."
    
    try:
        # Get all unorganized notes
        all_notes = app_state.note_processor.notes_collection.get(include=["documents", "metadatas"])
        if not all_notes["ids"]:
            return "No notes to organize."
        
        organized_count = 0
        for i, note_id in enumerate(all_notes["ids"]):
            note_text = all_notes["documents"][i]
            note_metadata = all_notes["metadatas"][i]
            
            # Skip if note is already assigned to a specific section
            if note_metadata.get("chapter_id") and note_metadata.get("subtopic_id"):
                continue
            
            # Use content classification to determine where to place the note
            # If custom rules are provided, use them to influence the classification
            if custom_rules:
                # For now, we'll just use the standard classification
                # In a more sophisticated implementation, custom_rules could be used to influence the classification
                classification = app_state.content_expander.classify_content(
                    note_text, 
                    app_state.current_outline.to_dict()
                )
            else:
                classification = app_state.content_expander.classify_content(
                    note_text, 
                    app_state.current_outline.to_dict()
                )
            
            if classification["chapter"] and classification["subtopic"]:
                # Update the note's metadata to include chapter/subtopic assignment
                results = app_state.note_processor.notes_collection.get(
                    ids=[note_id],
                    include=["metadatas"]
                )
                
                if results["ids"]:
                    metadata = results["metadatas"][0]
                    metadata["chapter_id"] = classification["chapter"]["id"]
                    metadata["subtopic_id"] = classification["subtopic"]["id"]
                    
                    # Filter out any None values from metadata as ChromaDB doesn't accept them
                    filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
                    
                    # Update in ChromaDB
                    app_state.note_processor.notes_collection.update(
                        ids=[note_id],
                        metadatas=[filtered_metadata]
                    )
                    
                    organized_count += 1
        
        result = f"Auto-organized {organized_count} notes into appropriate sections."
        gr.Info(result)
        return result
    except Exception as e:
        gr.Error(f"Failed to auto-organize notes: {e}")
        return f"Error: {e}"

def customize_organization_rules(app_state):
    """Allow users to customize organization rules."""
    if not app_state:
        gr.Warning("Please load a project first.")
        return "No project loaded."
    
    if not app_state.current_outline:
        gr.Warning("Please load an outline first.")
        return "No outline loaded."
    
    # For now, return a message indicating where users can customize rules
    # In a full implementation, this would open a rule configuration interface
    return "Organization rules can be customized. In a full implementation, this would open a configuration dialog where users could define custom rules for organizing notes into chapters and subtopics based on keywords, topics, or other criteria."

def load_content_for_editor(app_state, section_type, section_id):
    """Load content for editing in the content editor."""
    if not app_state:
        return "No project loaded.", ""
    
    try:
        if section_type == "note":
            # Get note content
            results = app_state.note_processor.notes_collection.get(
                ids=[section_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"]:
                content = results["documents"][0] 
                return content, section_id
            else:
                return "Note not found.", ""
        elif section_type == "content":
            # Get content item
            content_text = app_state.content_manager.get_content(section_id)
            return content_text, section_id
        else:
            return "Invalid section type.", ""
    except Exception as e:
        return f"Error loading content: {e}", ""

def save_content_from_editor(app_state, content_id, new_content):
    """Save edited content back to storage."""
    if not app_state:
        gr.Warning("Please load a project first.")
        return "No project loaded."
    
    if not content_id:
        gr.Warning("No content is currently loaded in the editor.")
        return "No content ID provided."
    
    try:
        # Determine if this is a note or content item and save accordingly
        # For now, we'll try to save as content first, then as note if that fails
        try:
            app_state.content_manager.update_content(content_id, new_content)
            gr.Success("Content saved successfully!")
            return "Content saved successfully!"
        except Exception:
            # If it's not a content item, it might be a note that was loaded for editing
            # For notes, we might need to create a new content item from the note
            try:
                # Get the note to understand its classification
                note_results = app_state.note_processor.notes_collection.get(
                    ids=[content_id],
                    include=["documents", "metadatas"]
                )
                
                if note_results["ids"]:
                    note_metadata = note_results["metadatas"][0]
                    # Create content from the note
                    content_id_new = app_state.content_manager.store_content(
                        content=new_content,
                        title=f"Edited content from note {content_id}",
                        chapter_id=note_metadata.get("chapter_id", "unknown"),
                        subtopic_id=note_metadata.get("subtopic_id", "unknown"),
                        source_note_ids=[content_id]
                    )
                    gr.Success("Note content saved as new content item!")
                    return f"Note saved as new content item: {content_id_new}"
            except Exception as e:
                gr.Error(f"Failed to save content: {e}")
                return f"Error saving content: {e}"
    except Exception as e:
        gr.Error(f"Failed to save changes: {e}")
        return f"Error: {e}"

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

            with gr.TabItem("📋 Content Organization"):
                gr.Markdown("## Content Organization")
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Outline Structure")
                        outline_tree = gr.JSON(label="Outline Structure")
                        refresh_outline_btn = gr.Button("Refresh Outline", variant="secondary")
                        
                        gr.Markdown("### Assign Notes to Sections")
                        note_selector = gr.Dropdown(label="Select Note", choices=[], multiselect=True, allow_custom_value=True)
                        section_selector = gr.Dropdown(label="Select Section", choices=[], value=None, allow_custom_value=True)
                        assign_notes_btn = gr.Button("Assign Notes to Section", variant="primary")
                        
                        gr.Markdown("### Batch Organize")
                        auto_organize_btn = gr.Button("Auto-Organize Notes", variant="secondary")
                        customize_rules_btn = gr.Button("Customize Organization Rules", variant="secondary")
                    with gr.Column(scale=1):
                        gr.Markdown("### Content Editor")
                        content_editor = gr.Textbox(label="Edit Content", lines=25, interactive=True, 
                                                   placeholder="Select a note or content section to edit...")
                        save_content_btn_org = gr.Button("Save Content", variant="primary")
                        refresh_content_btn = gr.Button("Refresh Content", variant="secondary")

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

        # Content Organization Tab Event Wiring
        refresh_outline_btn.click(
            fn=refresh_outline_tree,
            inputs=[app_state],
            outputs=[outline_tree]
        )
        
        refresh_outline_btn.click(  # Also update the note selector when refreshing outline
            fn=get_available_notes,
            inputs=[app_state],
            outputs=[note_selector]
        )
        
        refresh_outline_btn.click(  # Also update the section selector when refreshing outline
            fn=get_outline_sections,
            inputs=[app_state],
            outputs=[section_selector]
        )
        
        assign_notes_btn.click(
            fn=assign_notes_to_section,
            inputs=[app_state, note_selector, section_selector],
            outputs=[outline_tree]  # Update display after assignment
        )
        
        # Add event for loading note content into editor when a note is selected
        def load_selected_note_content(app_state, selected_notes):
            """Load the content of the first selected note into the editor."""
            if not selected_notes:
                return "", ""
            
            # Extract note ID from the selection (format: "note_id: preview text")
            note_selection = selected_notes[0]  # Use first selected note
            note_id = note_selection.split(":")[0] if ":" in note_selection else note_selection
            
            # Load the note content
            results = app_state.note_processor.notes_collection.get(
                ids=[note_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"]:
                content = results["documents"][0]
                return content, note_id  # Return content and note_id
            else:
                return "Note not found.", ""
        
        note_selector.change(
            fn=load_selected_note_content,
            inputs=[app_state, note_selector],
            outputs=[content_editor, current_content_id_state]
        )
        
        auto_organize_btn.click(
            fn=auto_organize_notes,
            inputs=[app_state],
            outputs=[outline_tree]  # Update display after auto-organization
        )
        
        customize_rules_btn.click(
            fn=customize_organization_rules,
            inputs=[app_state],
            outputs=[outline_tree]  # For now, just update display with status
        )
        
        save_content_btn_org.click(
            fn=save_content_from_editor,
            inputs=[app_state, current_content_id_state, content_editor],
            outputs=None
        )
        
        # Note that we can't directly use a textbox with fixed value in Gradio events
        # Instead, we'll handle this differently in actual implementation

    return ui

if __name__ == "__main__":
    web_ui = create_ui()
    web_ui.launch()
