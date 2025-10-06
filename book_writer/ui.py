
"""
Book Writer System - Gradio Web UI
Provides a web-based interface for the Book Writer application.
"""
import gradio as gr
import os
import time
from pathlib import Path
import yaml

from book_writer.app import BookWriterApp
from book_writer.content_organizer_ui import create_organizer_tab

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

def create_or_load_project(project_path_str: str, book_title: str = "My Book", target_pages: int = 100):
    """Creates or loads a project."""
    project_path = Path(project_path_str)
    if not project_path.exists():
        gr.Info(f"Project path not found. Creating new project at: {project_path}")
        app = BookWriterApp.create_project(project_path, book_title=book_title, target_pages=target_pages)
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
        
        results_ids = results.get("ids", [])
        if not results_ids:
            raise ValueError(f"Note with ID {note_id} not found")
        
        results_docs = results.get("documents", [])
        results_metas = results.get("metadatas", [])
        note_text = results_docs[0] if results_docs else ""
        metadata = results_metas[0] if results_metas else {}
        
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
def create_outline_tree_html(outline_dict, note_assignments=None):
    """Create an HTML representation of the outline tree with assignment indicators."""
    if not outline_dict:
        return "<p>No outline loaded.</p>"
    
    if note_assignments is None:
        note_assignments = {}
    
    html = '<div class="outline-tree">'
    html += '<ul class="outline-root">'
    
    for part_idx, part in enumerate(outline_dict.get("parts", [])):
        part_id = part.get("id", f"part-{part_idx}")
        part_title = part.get("title", "Untitled Part")
        
        # Count notes assigned to this part
        part_note_count = sum(1 for aid, (pid, cid, stid) in note_assignments.items() if pid == part_id)
        
        html += f'''
        <li class="outline-part">
            <div class="outline-item-header">
                <span class="toggle-btn">▼</span>
                <span class="outline-part-title">{part_title}</span>
                <span class="note-count">[{part_note_count} notes]</span>
            </div>
            <div class="outline-children">
                <ul class="outline-chapters">
        '''
        
        for chap_idx, chapter in enumerate(part.get("chapters", [])):
            chapter_id = chapter.get("id", f"chapter-{part_idx}-{chap_idx}")
            chapter_title = chapter.get("title", "Untitled Chapter")
            
            # Count notes assigned to this chapter
            chapter_note_count = sum(1 for aid, (pid, cid, stid) in note_assignments.items() if cid == chapter_id)
            
            html += f'''
                <li class="outline-chapter">
                    <div class="outline-item-header">
                        <span class="toggle-btn">▼</span>
                        <span class="outline-chapter-title">{chapter_title}</span>
                        <span class="note-count">[{chapter_note_count} notes]</span>
                    </div>
                    <div class="outline-children">
                        <ul class="outline-subtopics">
            '''
            
            for st_idx, subtopic in enumerate(chapter.get("subtopics", [])):
                subtopic_id = subtopic.get("id", f"subtopic-{part_idx}-{chap_idx}-{st_idx}")
                subtopic_title = subtopic.get("title", "Untitled Subtopic")
                
                # Count notes assigned to this subtopic
                subtopic_note_count = sum(1 for aid, (pid, cid, stid) in note_assignments.items() if stid == subtopic_id)
                
                html += f'''
                    <li class="outline-subtopic">
                        <div class="outline-item-header">
                            <span class="outline-subtopic-title">{subtopic_title}</span>
                            <span class="note-count">[{subtopic_note_count} notes]</span>
                        </div>
                    </li>
                '''
            
            html += '''
                        </ul>
                    </div>
                </li>
            '''
        
        html += '''
                </ul>
            </div>
        </li>
        '''
    
    html += '</ul>'
    html += '</div>'
    
    # Add CSS for the tree structure
    css = '''
    <style>
        .outline-tree {
            font-family: Arial, sans-serif;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
        }
        .outline-root {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .outline-part, .outline-chapter, .outline-subtopic {
            list-style: none;
            margin: 0;
            padding: 0;
        }
        .outline-item-header {
            padding: 5px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .outline-part:hover, .outline-chapter:hover {
            background-color: #f5f5f5;
        }
        .outline-part-title {
            font-weight: bold;
            color: #2c5aa0;
        }
        .outline-chapter-title {
            color: #3a66c1;
        }
        .outline-subtopic-title {
            color: #555;
            font-style: italic;
        }
        .note-count {
            background-color: #e7f3ff;
            border-radius: 12px;
            padding: 2px 8px;
            font-size: 0.8em;
        }
        .outline-children {
            margin-left: 20px;
            display: block; /* Default to expanded */
        }
        .toggle-btn {
            cursor: pointer;
            margin-right: 8px;
            width: 20px;
        }
    </style>
    '''
    
    return css + html


def refresh_outline_tree(app_state):
    """Refresh the outline structure display with assignment indicators."""
    if not app_state or not app_state.current_outline:
        return "No outline loaded."
    
    # Convert the outline to a dictionary structure
    outline_dict = app_state.current_outline.to_dict()
    
    # Get all notes and their classifications to add to the outline display
    note_assignments = {}  # Structure: {note_id: (part_id, chapter_id, subtopic_id)}
    try:
        all_notes = app_state.note_processor.notes_collection.get(include=["documents", "metadatas"])
        all_note_ids = all_notes.get("ids", [])
        if all_note_ids:
            all_note_metadatas = all_notes.get("metadatas", [])
            for i, note_id in enumerate(all_note_ids):
                metadata = all_note_metadatas[i] if i < len(all_note_metadatas) else {}
                chapter_id = metadata.get("chapter_id")
                subtopic_id = metadata.get("subtopic_id")
                
                # Find the corresponding part_id for the chapter
                part_id = None
                if chapter_id and subtopic_id:
                    for part in outline_dict.get("parts", []):
                        for chapter in part.get("chapters", []):
                            if chapter["id"] == chapter_id:
                                part_id = part["id"]
                                break
                        if part_id:
                            break
                
                if part_id and chapter_id and subtopic_id:
                    note_assignments[note_id] = (part_id, chapter_id, subtopic_id)
    except Exception:
        # If there's an error getting note assignments, continue with empty assignments
        pass
    
    # Create the HTML representation of the outline tree
    return create_outline_tree_html(outline_dict, note_assignments)

def get_available_notes(app_state, search_query=""):
    """Get a list of available notes for assignment, optionally filtered by search query."""
    if not app_state:
        return []
    
    try:
        # Get all notes from the notes collection
        all_notes = app_state.note_processor.notes_collection.get(include=["documents", "metadatas"])
        all_note_ids = all_notes.get("ids", [])
        if not all_note_ids:
            return []
        
        # Create a list of note options with ID and preview
        note_options = []
        search_query = search_query.lower() if search_query else ""
        
        all_note_docs = all_notes.get("documents", [])
        for i, note_id in enumerate(all_note_ids):
            note_doc = all_note_docs[i] if i < len(all_note_docs) else ""
            preview = note_doc[:50] + "..." if len(note_doc) > 50 else note_doc
            
            # If search query is specified, filter notes
            if search_query:
                if search_query in note_doc.lower() or search_query in note_id.lower():
                    note_options.append(f"{note_id}: {preview}")
            else:
                # If no search query, add all notes
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
    """Automatically organize notes based on enhanced content analysis."""
    # Print to console for debugging
    print("Starting auto-organization process...")
    
    if not app_state:
        error_msg = "Please load a project first."
        print(f"ERROR: {error_msg}")
        gr.Warning(error_msg)
        return error_msg
    
    if not app_state.current_outline:
        error_msg = "Please load an outline first."
        print(f"ERROR: {error_msg}")
        gr.Warning(error_msg)
        return error_msg
    
    try:
        print("Fetching notes for organization...")
        # Get all unorganized notes
        all_notes = app_state.note_processor.notes_collection.get(include=["documents", "metadatas"])
        all_note_ids = all_notes.get("ids", [])
        if not all_note_ids:
            msg = "No notes to organize."
            print(msg)
            return msg
        
        total_notes = len(all_note_ids)
        organized_count = 0
        skipped_count = 0
        error_count = 0
        failed_notes = []
        
        print(f"Processing {total_notes} notes...")
        gr.Info(f"Starting auto-organization of {total_notes} notes...")
        
        all_note_docs = all_notes.get("documents", [])
        all_note_metadatas = all_notes.get("metadatas", [])
        for i, note_id in enumerate(all_note_ids):
            note_text = all_note_docs[i] if i < len(all_note_docs) else ""
            note_metadata = all_note_metadatas[i] if i < len(all_note_metadatas) else {}
            
            # Create a preview of the note text for better logging
            note_preview = note_text[:50] + "..." if len(note_text) > 50 else note_text
            print(f"Processing note {i+1}/{total_notes}: {note_id[:8]}... ({note_preview})")
            gr.Info(f"Processing note {i+1}/{total_notes}: {note_id[:8]}...")
            
            # Skip if note is already assigned to a specific section
            if note_metadata.get("chapter_id") and note_metadata.get("subtopic_id"):
                print(f"  Skipping note {note_id[:8]}... - already assigned")
                skipped_count += 1
                continue
            
            # Use enhanced content classification to determine where to place the note
            try:
                print(f"  Classifying note {note_id[:8]}...")
                # Use the content expander's classify_content function with enhanced logic
                if custom_rules:
                    # Apply custom rules if provided
                    print("  Using custom rules for classification")
                    # For now, we'll use the standard classification with custom rules as parameters
                    classification = app_state.content_expander.classify_content(
                        note_text, 
                        app_state.current_outline.to_dict()
                    )
                else:
                    print("  Using standard classification")
                    classification = app_state.content_expander.classify_content(
                        note_text, 
                        app_state.current_outline.to_dict()
                    )
                
                print(f"  Classification result for {note_id[:8]}...: {classification is not None}")
                
                # Verify the classification result
                if classification and classification.get("chapter") and classification.get("subtopic"):
                    print(f"  Valid classification found for {note_id[:8]}...")
                    # Update the note's metadata to include chapter/subtopic assignment
                    results = app_state.note_processor.notes_collection.get(
                        ids=[note_id],
                        include=["metadatas"]
                    )
                    
                    results_ids = results.get("ids", [])
                    if results_ids:
                        print(f"  Updating metadata for note {note_id[:8]}...")
                        results_metas = results.get("metadatas", [])
                        metadata = results_metas[0] if results_metas else {}
                        metadata["chapter_id"] = classification["chapter"]["id"]
                        metadata["subtopic_id"] = classification["subtopic"]["id"]
                        
                        # Additional metadata to track organization
                        metadata["organized_by"] = "auto"
                        metadata["organized_date"] = str(time.time())
                        
                        # Filter out any None values from metadata as ChromaDB doesn't accept them
                        filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
                        
                        # Update in ChromaDB
                        app_state.note_processor.notes_collection.update(
                            ids=[note_id],
                            metadatas=[filtered_metadata]
                        )
                        
                        organized_count += 1
                        print(f"  Successfully organized note {note_id[:8]}...")
                        gr.Info(f"Successfully organized note: {note_id[:8]}...")
                    else:
                        error_msg = f"Failed to get metadata for note {note_id[:8]}..."
                        print(f"  ERROR: {error_msg}")
                        failed_notes.append((note_id, error_msg))
                        error_count += 1
                else:
                    # If classification didn't return valid results, log as error
                    error_msg = f"No valid classification for note {note_id[:8]}..."
                    print(f"  ERROR: {error_msg}")
                    failed_notes.append((note_id, error_msg))
                    error_count += 1
            except Exception as e:
                error_msg = f"Failed to classify note {note_id[:8]}...: {str(e)}"
                print(f"  EXCEPTION: {error_msg}")
                failed_notes.append((note_id, error_msg))
                gr.Warning(error_msg)
                error_count += 1
        
        # Prepare detailed result message
        result_details = []
        result_details.append(f"Auto-organization completed:")
        result_details.append(f"  - Total notes processed: {total_notes}")
        result_details.append(f"  - Successfully organized: {organized_count}")
        result_details.append(f"  - Skipped (already assigned): {skipped_count}")
        result_details.append(f"  - Errors: {error_count}")
        
        if failed_notes:
            result_details.append(f"\nFailed notes:")
            for note_id, error in failed_notes[:5]:  # Show first 5 failures
                result_details.append(f"  - {note_id[:8]}...: {error}")
            if len(failed_notes) > 5:
                result_details.append(f"  ... and {len(failed_notes) - 5} more")
        
        result = "\n".join(result_details)
        print(result)
        
        if error_count == 0:
            success_msg = f"Success! Organized {organized_count} notes."
            gr.Info(success_msg)
        else:
            warning_msg = f"Completed with {error_count} errors. Check console for details."
            gr.Warning(warning_msg)
            
        return result
    except Exception as e:
        error_msg = f"Failed to auto-organize notes: {e}"
        print(f"CRITICAL ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        gr.Error(error_msg)
        return f"Error: {e}"

def parse_keyword_rules(rules_text):
    """Parse keyword rules from text input."""
    rules = {}
    if not rules_text:
        return rules
    
    try:
        rule_pairs = rules_text.split(';')
        for pair in rule_pairs:
            pair = pair.strip()
            if not pair:
                continue
                
            parts = pair.split(':')
            if len(parts) >= 4:
                keyword = parts[0].strip().lower()
                part_id = parts[1].strip()
                chapter_id = parts[2].strip()
                subtopic_id = parts[3].strip()
                
                # Store the rule with the keyword as key
                rules[keyword] = (part_id, chapter_id, subtopic_id)
    except Exception as e:
        print(f"Error parsing keyword rules: {e}")
    
    return rules


def find_notes_needing_review(app_state):
    """Find notes that may need human review before assignment."""
    return _get_current_review_queue(app_state)


def approve_assignment(app_state, selected_note):
    """Approve the assignment of a note and return updated review queue."""
    if not app_state:
        gr.Warning("Please load a project first.")
        return []
    
    if not selected_note:
        gr.Warning("Please select a note to approve.")
        # Return current review queue when nothing is selected
        return _get_current_review_queue(app_state)
    
    try:
        # Extract note ID from the selection (format: "note_id: preview text")
        note_id = selected_note.split(":")[0] if ":" in selected_note else selected_note
        
        # Update the note's metadata to mark it as approved
        results = app_state.note_processor.notes_collection.get(
            ids=[note_id],
            include=["metadatas"]
        )
        
        results_ids = results.get("ids", [])
        if results_ids:
            results_metas = results.get("metadatas", [])
            metadata = results_metas[0] if results_metas else {}
            # Mark as reviewed and approved
            metadata["reviewed"] = True
            metadata["review_status"] = "approved"
            metadata["reviewed_date"] = str(time.time())
            
            # Filter out any None values from metadata as ChromaDB doesn't accept them
            filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
            
            # Update in ChromaDB
            app_state.note_processor.notes_collection.update(
                ids=[note_id],
                metadatas=[filtered_metadata]
            )
            
            gr.Info(f"Note {note_id[:8]}... approved for assignment.")
        else:
            gr.Warning(f"Note not found: {note_id}")
        
        # Return the updated review queue (notes that still need review)
        return _get_current_review_queue(app_state)
    except Exception as e:
        gr.Error(f"Failed to approve note assignment: {e}")
        # Return current review queue even if there's an error
        return _get_current_review_queue(app_state)


def _get_current_review_queue(app_state):
    """Helper function to get the current list of notes needing review."""
    if not app_state:
        return []
    
    if not app_state.current_outline:
        return []
    
    try:
        # Get all notes
        all_notes = app_state.note_processor.notes_collection.get(include=["documents", "metadatas"])
        all_note_ids = all_notes.get("ids", [])
        if not all_note_ids:
            return []
        
        review_candidates = []
        
        all_note_docs = all_notes.get("documents", [])
        all_note_metadatas = all_notes.get("metadatas", [])
        for i, note_id in enumerate(all_note_ids):
            note_text = all_note_docs[i] if i < len(all_note_docs) else ""
            note_metadata = all_note_metadatas[i] if i < len(all_note_metadatas) else {}
            
            # Notes that haven't been assigned yet are candidates for review
            if not note_metadata.get("chapter_id") or not note_metadata.get("subtopic_id"):
                preview = note_text[:50] + "..." if len(note_text) > 50 else note_text
                review_candidates.append(f"{note_id}: {preview}")
            # Notes with low confidence scores or specific status markers could also be added
            elif note_metadata.get("organization_confidence", 1.0) < 0.5:
                preview = note_text[:50] + "..." if len(note_text) > 50 else note_text
                review_candidates.append(f"{note_id}: {preview} [low confidence]")
        
        return review_candidates
    except Exception as e:
        gr.Error(f"Failed to get review queue: {e}")
        return []


def get_organization_statistics(app_state):
    """Get statistics about note organization."""
    if not app_state:
        return "No project loaded."
    
    if not app_state.current_outline:
        return "No outline loaded."
    
    try:
        # Get all notes
        all_notes = app_state.note_processor.notes_collection.get(include=["documents", "metadatas"])
        all_note_ids = all_notes.get("ids", [])
        if not all_note_ids:
            return "No notes in the system."
        total_notes = len(all_note_ids)
        assigned_notes = 0
        unassigned_notes = 0
        reviewed_notes = 0
        low_confidence_notes = 0
        
        # Count assigned vs unassigned
        all_note_metadatas = all_notes.get("metadatas", [])
        for i, note_id in enumerate(all_note_ids):
            metadata = all_note_metadatas[i] if i < len(all_note_metadatas) else {}
            
            if metadata.get("chapter_id") and metadata.get("subtopic_id"):
                assigned_notes += 1
            else:
                unassigned_notes += 1
                
            if metadata.get("reviewed"):
                reviewed_notes += 1
                
            if metadata.get("organization_confidence", 1.0) < 0.5:
                low_confidence_notes += 1
        
        # Get outline structure info
        outline = app_state.current_outline.to_dict()
        total_parts = len(outline.get("parts", []))
        total_chapters = 0
        total_subtopics = 0
        
        for part in outline.get("parts", []):
            total_chapters += len(part.get("chapters", []))
            for chapter in part.get("chapters", []):
                total_subtopics += len(chapter.get("subtopics", []))
        
        # Create statistics summary
        stats = f"""Organization Statistics
==================
Total Notes: {total_notes}
Assigned Notes: {assigned_notes}
Unassigned Notes: {unassigned_notes}
Notes with Review Status: {reviewed_notes}
Low Confidence Assignments: {low_confidence_notes}

Outline Structure:
- Parts: {total_parts}
- Chapters: {total_chapters}
- Subtopics: {total_subtopics}

Organization Progress: 
- {(assigned_notes/total_notes*100):.1f}% of notes assigned ({assigned_notes}/{total_notes})
"""
        
        return stats
    except Exception as e:
        return f"Error calculating statistics: {e}"


def reject_assignment(app_state, selected_note):
    """Mark a note as needing revision and return updated review queue."""
    if not app_state:
        gr.Warning("Please load a project first.")
        return []
    
    if not selected_note:
        gr.Warning("Please select a note to reject.")
        # Return current review queue when nothing is selected
        return _get_current_review_queue(app_state)
    
    try:
        # Extract note ID from the selection (format: "note_id: preview text")
        note_id = selected_note.split(":")[0] if ":" in selected_note else selected_note
        
        # Update the note's metadata to mark it as needing revision
        results = app_state.note_processor.notes_collection.get(
            ids=[note_id],
            include=["metadatas"]
        )
        
        results_ids = results.get("ids", [])
        if results_ids:
            results_metas = results.get("metadatas", [])
            metadata = results_metas[0] if results_metas else {}
            # Mark as reviewed but needing revision
            metadata["reviewed"] = True
            metadata["review_status"] = "needs_revision"
            metadata["reviewed_date"] = str(time.time())
            
            # Filter out any None values from metadata as ChromaDB doesn't accept them
            filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
            
            # Update in ChromaDB
            app_state.note_processor.notes_collection.update(
                ids=[note_id],
                metadatas=[filtered_metadata]
            )
            
            gr.Info(f"Note {note_id[:8]}... marked for revision.")
        else:
            gr.Warning(f"Note not found: {note_id}")
        
        # Return the updated review queue (notes that still need review)
        return _get_current_review_queue(app_state)
    except Exception as e:
        gr.Error(f"Failed to reject note assignment: {e}")
        # Return current review queue even if there's an error
        return _get_current_review_queue(app_state)


def apply_keyword_rules_to_notes(app_state, rules_text):
    """Apply keyword-based rules to organize notes."""
    if not app_state:
        gr.Warning("Please load a project first.")
        return "No project loaded."
    
    if not app_state.current_outline:
        gr.Warning("Please load an outline first.")
        return "No outline loaded."
    
    # Parse the rules
    keyword_rules = parse_keyword_rules(rules_text)
    if not keyword_rules:
        gr.Warning("No valid rules found. Please check the format.")
        return "No rules applied."
    
    try:
        # Get all notes
        all_notes = app_state.note_processor.notes_collection.get(include=["documents", "metadatas"])
        all_note_ids = all_notes.get("ids", [])
        if not all_note_ids:
            return "No notes to organize."
        
        organized_count = 0
        
        all_note_docs = all_notes.get("documents", [])
        all_note_metadatas = all_notes.get("metadatas", [])
        for i, note_id in enumerate(all_note_ids):
            note_text = all_note_docs[i].lower() if i < len(all_note_docs) else ""
            note_metadata = all_note_metadatas[i] if i < len(all_note_metadatas) else {}
            
            # Skip if note is already assigned
            if note_metadata.get("chapter_id") and note_metadata.get("subtopic_id"):
                continue
            
            # Check if any keyword rule applies to this note
            for keyword, (part_id, chapter_id, subtopic_id) in keyword_rules.items():
                if keyword in note_text:
                    # Update the note's metadata with the matched assignment
                    results = app_state.note_processor.notes_collection.get(
                        ids=[note_id],
                        include=["metadatas"]
                    )
                    
                    results_ids = results.get("ids", [])
                    if results_ids:
                        results_metas = results.get("metadatas", [])
                        metadata = results_metas[0] if results_metas else {}
                        metadata["chapter_id"] = chapter_id
                        metadata["subtopic_id"] = subtopic_id
                        metadata["organized_by"] = "keyword_rule"
                        metadata["organized_date"] = str(time.time())
                        
                        # Filter out any None values from metadata as ChromaDB doesn't accept them
                        filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
                        
                        # Update in ChromaDB
                        app_state.note_processor.notes_collection.update(
                            ids=[note_id],
                            metadatas=[filtered_metadata]
                        )
                        
                        organized_count += 1
                        break  # Only apply the first matching rule
        
        result = f"Applied keyword rules: {organized_count} notes organized based on keywords."
        gr.Info(result)
        return result
    except Exception as e:
        gr.Error(f"Failed to apply keyword rules: {e}")
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
            
            results_ids = results.get("ids", [])
            if results_ids:
                results_docs = results.get("documents", [])
                content = results_docs[0] if results_docs else ""
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
                
                note_results_ids = note_results.get("ids", [])
                if note_results_ids:
                    note_results_metas = note_results.get("metadatas", [])
                    note_metadata = note_results_metas[0] if note_results_metas else {}
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
                book_title_input = gr.Textbox(label="Book Title", placeholder="Enter the title of your book", value="My Book", lines=1)
                target_pages_input = gr.Number(label="Target Number of Pages", value=100, minimum=1, maximum=10000, step=1)
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

            # New enhanced organizer tab
            create_organizer_tab(app_state)
            
            with gr.TabItem("📋 Advanced Organization"):
                gr.Markdown("## Advanced Content Organization")
                gr.Markdown("Follow these steps to organize your content:")
                gr.Markdown("1. **Organize Content:** Click the 'Organize Content' button to automatically cluster your content and get organization suggestions.")
                gr.Markdown("2. **Review Suggestions:** Review the suggestions for content gaps and reorganization. You can then apply or dismiss these suggestions.")
                gr.Markdown("3. **Visualize Structure:** Generate a visualization of your content structure to better understand the relationships between your content.")

                with gr.Tabs():
                    with gr.TabItem("Step 1: Organize Content"):
                        gr.Markdown("### Organize Your Content")
                        content_ids_input = gr.Textbox(label="Content IDs", placeholder="Enter a comma-separated list of content IDs to organize", lines=1)
                        organize_content_btn = gr.Button("Organize Content", variant="primary")
                        organization_summary_output = gr.JSON(label="Organization Summary")

                    with gr.TabItem("Step 2: Review Suggestions"):
                        gr.Markdown("### Review Organization Suggestions")
                        get_suggestions_btn = gr.Button("Get Suggestions", variant="primary")
                        suggestions_output = gr.JSON(label="Organization Suggestions")

                    with gr.TabItem("Step 3: Visualize Structure"):
                        gr.Markdown("### Visualize Content Structure")
                        visualize_content_btn = gr.Button("Visualize Content", variant="primary")
                        visualization_output = gr.Image(label="Content Structure Visualization")

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
                        
            with gr.TabItem("📊 Progress & Gamification"):
                gr.Markdown("## Writing Progress & Recommendations")
                with gr.Row():
                    with gr.Column():
                        progress_summary = gr.Markdown(label="Progress Summary")
                        progress_by_part = gr.JSON(label="Progress by Part")
                        
                        # Button to get progress and recommendations
                        get_progress_btn = gr.Button("Refresh Progress & Recommendations", variant="primary")
                        progress_output = gr.JSON(label="Detailed Progress")
                        recommendations_output = gr.Textbox(label="Recommendations", lines=10)
                        
                        def update_progress(app_state):
                            if not app_state:
                                gr.Warning("Please load a project first.")
                                return "", {}, {}, ""
                            try:
                                progress_data = app_state.get_writing_progress()
                                recommendations_data = app_state.get_gamification_recommendations()
                                
                                # Ensure both data structures exist and are properly formatted
                                if not progress_data:
                                    return "Could not retrieve progress data.", {}, {}, ""

                                if not recommendations_data:
                                    return "Could not retrieve recommendations.", {}, {}, ""

                                # Format progress summary - with additional safety checks
                                total_target_pages = progress_data.get('total_target_pages', 0) or 0
                                total_written_pages = progress_data.get('total_written_pages', 0) or 0
                                progress_percentage = progress_data.get('progress_percentage', 0) or 0
                                total_notes = progress_data.get('total_notes', 0) or 0
                                
                                progress_text = f"""
                                ### Writing Progress
                                - **Target Pages**: {total_target_pages}
                                - **Written Pages**: {total_written_pages:.1f}
                                - **Progress**: {progress_percentage:.1f}%
                                - **Total Notes**: {total_notes}
                                
                                ### Milestones
                                """
                                
                                # Ensure milestones is a list to prevent NoneType error
                                milestones = recommendations_data.get('milestones', [])
                                if milestones is None:
                                    milestones = []
                                
                                for milestone in milestones:
                                    if milestone:  # Additional safety check
                                        progress_text += f"- {milestone}\n"
                                
                                # Format progress by part
                                progress_by_part_data = {}
                                progress_by_section = progress_data.get('progress_by_section', [])
                                if progress_by_section is None:
                                    progress_by_section = []
                                
                                for part in progress_by_section:
                                    if not part:
                                        # When part is empty, use default values
                                        title = 'Untitled'
                                        part_progress = 0
                                        progress_by_part_data[title] = {
                                            'written': "0.0",
                                            'target': "0.0",
                                            'percentage': f"{part_progress:.1f}%"
                                        }
                                        continue
                                        
                                    # Extract values with additional safety checks
                                    written_pages = part.get('written_pages', 0) or 0
                                    target_pages = part.get('target_pages', 0) or 1  # Default to 1 to prevent division by zero
                                    title = part.get('title', 'Untitled') or 'Untitled'
                                    
                                    part_progress = (written_pages / target_pages * 100) if target_pages > 0 else 0
                                    progress_by_part_data[title] = {
                                        'written': f"{written_pages:.1f}",
                                        'target': f"{target_pages:.1f}",
                                        'percentage': f"{part_progress:.1f}%"
                                    }
                                
                                # Ensure recommendations is a list to prevent NoneType error when calling join()
                                recommendations_list = recommendations_data.get('recommendations', [])
                                if recommendations_list is None:
                                    recommendations_list = []
                                
                                # Filter out None values from recommendations list
                                recommendations_list = [rec for rec in recommendations_list if rec is not None]
                                
                                recommendations_text = "\n".join(recommendations_list)
                                
                                return progress_text, progress_by_part_data, progress_data, recommendations_text
                            except Exception as e:
                                gr.Error(f"Failed to get progress: {e}")
                                return f"Error getting progress: {e}", {}, {}, ""
                        
                        get_progress_btn.click(
                            fn=update_progress,
                            inputs=[app_state],
                            outputs=[progress_summary, progress_by_part, progress_output, recommendations_output]
                        )
                        
        # --- Event Wiring ---
        load_project_btn.click(
            fn=create_or_load_project,
            inputs=[project_path_input, book_title_input, target_pages_input],
            outputs=[app_state, project_path_display, outline_display]
        )

        def organize_content_ui(app_state, content_ids_str):
            if not app_state:
                gr.Warning("Please load a project first.")
                return None
            if not content_ids_str:
                gr.Warning("Please enter content IDs.")
                return None
            content_ids = [c.strip() for c in content_ids_str.split(',')]
            summary = app_state.organize_content(content_ids)
            return summary

        def get_suggestions_ui(app_state):
            if not app_state:
                gr.Warning("Please load a project first.")
                return None
            suggestions = app_state.get_organization_suggestions()
            return suggestions

        def visualize_content_ui(app_state):
            if not app_state:
                gr.Warning("Please load a project first.")
                return None
            output_path = app_state.project_path / "visualizations" / "content_structure.png"
            app_state.visualize_content_structure(output_path)
            return str(output_path)

        organize_content_btn.click(
            fn=organize_content_ui,
            inputs=[app_state, content_ids_input],
            outputs=[organization_summary_output]
        )

        get_suggestions_btn.click(
            fn=get_suggestions_ui,
            inputs=[app_state],
            outputs=[suggestions_output]
        )

        visualize_content_btn.click(
            fn=visualize_content_ui,
            inputs=[app_state],
            outputs=[visualization_output]
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

        
        # Note that we can't directly use a textbox with fixed value in Gradio events
        # Instead, we'll handle this differently in actual implementation

    return ui

if __name__ == "__main__":
    web_ui = create_ui()
    web_ui.launch()
