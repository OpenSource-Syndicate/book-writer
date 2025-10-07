"""
Enhanced Content Organizer UI for Book Writer
Provides drag-and-drop-like interface for organizing notes into outline structure
"""
import gradio as gr
import json
import time
from typing import List, Dict, Optional, Tuple


def get_all_notes_list(app_state, search_filter: str = "") -> List[Dict]:
    """Get all notes with preview text for display."""
    if not app_state:
        return []
    
    try:
        all_notes = app_state.note_processor.notes_collection.get(
            include=["documents", "metadatas"]
        )
        all_note_ids = all_notes.get("ids", [])
        if not all_note_ids:
            return []
        
        notes_list = []
        all_note_docs = all_notes.get("documents", [])
        all_note_metadatas = all_notes.get("metadatas", [])
        
        for i, note_id in enumerate(all_note_ids):
            note_text = all_note_docs[i] if i < len(all_note_docs) else ""
            note_metadata = all_note_metadatas[i] if i < len(all_note_metadatas) else {}
            
            # Apply search filter
            if search_filter:
                if search_filter.lower() not in note_text.lower() and search_filter.lower() not in note_id.lower():
                    continue
            
            # Get assignment status
            chapter_id = note_metadata.get("chapter_id")
            subtopic_id = note_metadata.get("subtopic_id")
            is_assigned = bool(chapter_id and subtopic_id)
            
            # Create preview
            preview = note_text[:80] + "..." if len(note_text) > 80 else note_text
            
            notes_list.append({
                "id": note_id,
                "preview": preview,
                "full_text": note_text,
                "is_assigned": is_assigned,
                "chapter_id": chapter_id,
                "subtopic_id": subtopic_id
            })
        
        return notes_list
    except Exception as e:
        print(f"Error getting notes: {e}")
        return []


def format_notes_as_html(notes_list: List[Dict], show_assigned: bool = True, show_unassigned: bool = True) -> str:
    """Format notes list as interactive HTML with drag handles."""
    if not notes_list:
        return "<p style='color: #888; padding: 20px;'>No notes found. Add some notes in the Writer's Desk tab.</p>"
    
    html = """
    <style>
        .notes-container {
            max-height: 600px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: #fafafa;
        }
        .note-item {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 8px;
            cursor: move;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .note-item:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-color: #4CAF50;
        }
        .note-item.assigned {
            border-left: 4px solid #4CAF50;
            background: #f1f8f4;
        }
        .note-item.unassigned {
            border-left: 4px solid #ff9800;
        }
        .drag-handle {
            font-size: 20px;
            color: #999;
            cursor: grab;
        }
        .drag-handle:active {
            cursor: grabbing;
        }
        .note-content {
            flex: 1;
        }
        .note-id {
            font-size: 11px;
            color: #666;
            font-family: monospace;
            margin-bottom: 4px;
        }
        .note-preview {
            font-size: 14px;
            color: #333;
            line-height: 1.4;
        }
        .note-status {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 500;
        }
        .status-assigned {
            background: #c8e6c9;
            color: #2e7d32;
        }
        .status-unassigned {
            background: #ffe0b2;
            color: #e65100;
        }
    </style>
    <div class="notes-container">
    """
    
    for note in notes_list:
        # Filter by assignment status
        if note["is_assigned"] and not show_assigned:
            continue
        if not note["is_assigned"] and not show_unassigned:
            continue
        
        status_class = "assigned" if note["is_assigned"] else "unassigned"
        status_label = "Assigned" if note["is_assigned"] else "Unassigned"
        status_badge_class = "status-assigned" if note["is_assigned"] else "status-unassigned"
        
        html += f"""
        <div class="note-item {status_class}" data-note-id="{note['id']}" draggable="true">
            <div class="drag-handle">⋮⋮</div>
            <div class="note-content">
                <div class="note-id">{note['id'][:16]}...</div>
                <div class="note-preview">{note['preview']}</div>
            </div>
            <span class="note-status {status_badge_class}">{status_label}</span>
        </div>
        """
    
    html += "</div>"
    return html


def format_outline_as_interactive_tree(app_state) -> str:
    """Format outline as an interactive tree that can receive dropped notes."""
    if not app_state or not app_state.current_outline:
        return "<p style='color: #888; padding: 20px;'>No outline loaded. Load a project first.</p>"
    
    outline_dict = app_state.current_outline.to_dict()
    
    html = """
    <style>
        .outline-tree-container {
            max-height: 600px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: #fafafa;
        }
        .outline-part {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .outline-part-header {
            font-weight: bold;
            font-size: 16px;
            color: #1976d2;
            margin-bottom: 8px;
            padding: 8px;
            background: #e3f2fd;
            border-radius: 4px;
        }
        .outline-chapter {
            margin-left: 15px;
            margin-bottom: 8px;
            padding: 8px;
            background: #f5f5f5;
            border-radius: 4px;
        }
        .outline-chapter-header {
            font-weight: 600;
            font-size: 14px;
            color: #0288d1;
            margin-bottom: 6px;
        }
        .outline-subtopic {
            margin-left: 15px;
            padding: 8px 12px;
            margin-bottom: 4px;
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .outline-subtopic:hover {
            background: #fff9c4;
            border-color: #fbc02d;
        }
        .outline-subtopic.drop-target {
            background: #c8e6c9;
            border: 2px dashed #4caf50;
        }
        .subtopic-title {
            font-size: 13px;
            color: #424242;
        }
        .note-count-badge {
            display: inline-block;
            background: #e0e0e0;
            color: #616161;
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 10px;
            margin-left: 8px;
        }
    </style>
    <div class="outline-tree-container">
    """
    
    # Get note assignments
    note_assignments = {}
    try:
        all_notes = app_state.note_processor.notes_collection.get(include=["metadatas"])
        all_note_ids = all_notes.get("ids", [])
        all_note_metadatas = all_notes.get("metadatas", [])
        
        for i, note_id in enumerate(all_note_ids):
            metadata = all_note_metadatas[i] if i < len(all_note_metadatas) else {}
            chapter_id = metadata.get("chapter_id")
            subtopic_id = metadata.get("subtopic_id")
            if chapter_id and subtopic_id:
                key = f"{chapter_id}_{subtopic_id}"
                note_assignments[key] = note_assignments.get(key, 0) + 1
    except Exception:
        pass
    
    for part in outline_dict.get("parts", []):
        part_id = part.get("id", "")
        part_title = part.get("title", "Untitled Part")
        
        html += f"""
        <div class="outline-part">
            <div class="outline-part-header">📚 {part_title}</div>
        """
        
        for chapter in part.get("chapters", []):
            chapter_id = chapter.get("id", "")
            chapter_title = chapter.get("title", "Untitled Chapter")
            
            html += f"""
            <div class="outline-chapter">
                <div class="outline-chapter-header">📖 {chapter_title}</div>
            """
            
            for subtopic in chapter.get("subtopics", []):
                subtopic_id = subtopic.get("id", "")
                subtopic_title = subtopic.get("title", "Untitled Subtopic")
                
                # Count notes assigned to this subtopic
                key = f"{chapter_id}_{subtopic_id}"
                note_count = note_assignments.get(key, 0)
                
                html += f"""
                <div class="outline-subtopic" 
                     data-chapter-id="{chapter_id}" 
                     data-subtopic-id="{subtopic_id}"
                     ondrop="handleDrop(event)" 
                     ondragover="handleDragOver(event)"
                     ondragleave="handleDragLeave(event)">
                    <span class="subtopic-title">📄 {subtopic_title}</span>
                    <span class="note-count-badge">{note_count} notes</span>
                </div>
                """
            
            html += "</div>"
        
        html += "</div>"
    
    html += """
    </div>
    <script>
        // Drag and drop handlers
        document.addEventListener('dragstart', function(e) {
            if (e.target.classList.contains('note-item')) {
                e.dataTransfer.setData('note-id', e.target.dataset.noteId);
                e.target.style.opacity = '0.5';
            }
        });
        
        document.addEventListener('dragend', function(e) {
            if (e.target.classList.contains('note-item')) {
                e.target.style.opacity = '1';
            }
        });
        
        function handleDragOver(e) {
            e.preventDefault();
            e.currentTarget.classList.add('drop-target');
        }
        
        function handleDragLeave(e) {
            e.currentTarget.classList.remove('drop-target');
        }
        
        function handleDrop(e) {
            e.preventDefault();
            e.currentTarget.classList.remove('drop-target');
            
            const noteId = e.dataTransfer.getData('note-id');
            const chapterId = e.currentTarget.dataset.chapterId;
            const subtopicId = e.currentTarget.dataset.subtopicId;
            
            // Bridge payload to Gradio via hidden components
            try {
                const payload = JSON.stringify({ note_id: noteId, chapter_id: chapterId, subtopic_id: subtopicId });
                const inputContainer = document.getElementById('drop_receiver');
                const btnContainer = document.getElementById('drop_apply');
                // Find actual input and button elements inside containers
                const inputEl = inputContainer ? (inputContainer.querySelector('textarea, input') || inputContainer) : null;
                const btnEl = btnContainer ? (btnContainer.querySelector('button') || btnContainer) : null;
                if (inputEl && btnEl) {
                    // Set value and dispatch input event so Gradio picks up the change
                    inputEl.value = payload;
                    inputEl.dispatchEvent(new Event('input', { bubbles: true }));
                    // Visual feedback
                    e.currentTarget.style.background = '#c8e6c9';
                    // Trigger backend assignment
                    btnEl.click();
                    setTimeout(() => {
                        e.currentTarget.style.background = 'white';
                    }, 600);
                } else {
                    console.warn('Drop bridge elements not found.');
                }
            } catch (err) {
                console.error('Failed to send drop payload:', err);
            }
        }
    </script>
    """
    
    return html


def assign_note_to_section(app_state, note_id: str, chapter_id: str, subtopic_id: str) -> Tuple[bool, str]:
    """Assign a note to a specific chapter and subtopic."""
    if not app_state:
        return False, "No project loaded."
    
    if not note_id or not chapter_id or not subtopic_id:
        return False, "Missing required parameters."
    
    try:
        # Get the note's current metadata
        results = app_state.note_processor.notes_collection.get(
            ids=[note_id],
            include=["metadatas"]
        )
        
        if not results.get("ids"):
            return False, f"Note {note_id} not found."
        
        # Update metadata
        metadata = results["metadatas"][0] if results.get("metadatas") else {}
        metadata["chapter_id"] = chapter_id
        metadata["subtopic_id"] = subtopic_id
        metadata["organized_by"] = "manual_drag_drop"
        metadata["organized_date"] = str(time.time())
        
        # Filter out None values
        filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Update in ChromaDB
        app_state.note_processor.notes_collection.update(
            ids=[note_id],
            metadatas=[filtered_metadata]
        )
        
        return True, f"Note {note_id[:8]}... assigned successfully!"
    except Exception as e:
        return False, f"Error: {str(e)}"


def batch_assign_notes(app_state, note_ids: List[str], chapter_id: str, subtopic_id: str) -> str:
    """Assign multiple notes to a section at once."""
    if not app_state:
        return "No project loaded."
    
    if not note_ids:
        return "No notes selected."
    
    success_count = 0
    error_count = 0
    
    for note_id in note_ids:
        success, msg = assign_note_to_section(app_state, note_id, chapter_id, subtopic_id)
        if success:
            success_count += 1
        else:
            error_count += 1
    
    return f"Assigned {success_count} notes successfully. {error_count} errors."


def search_and_filter_notes(app_state, search_query: str, show_assigned: bool, show_unassigned: bool) -> str:
    """Search and filter notes, return formatted HTML."""
    notes_list = get_all_notes_list(app_state, search_query)
    return format_notes_as_html(notes_list, show_assigned, show_unassigned)


def get_outline_target_options(app_state) -> Tuple[list, dict]:
    """Return dropdown options for target subtopics and a map to ids.
    Returns (options, mapping) where options is a list of display strings and mapping maps display->(chapter_id, subtopic_id)
    """
    options = []
    mapping = {}
    if not app_state or not app_state.current_outline:
        return options, mapping
    outline = app_state.current_outline.to_dict()
    for part in outline.get("parts", []):
        part_title = part.get("title", "Part")
        for chapter in part.get("chapters", []):
            chapter_title = chapter.get("title", "Chapter")
            chapter_id = chapter.get("id")
            for subtopic in chapter.get("subtopics", []):
                subtopic_title = subtopic.get("title", "Subtopic")
                subtopic_id = subtopic.get("id")
                label = f"{part_title} / {chapter_title} / {subtopic_title}"
                options.append(label)
                mapping[label] = (chapter_id, subtopic_id)
    return options, mapping


def create_organizer_tab(app_state_component):
    """Create the enhanced content organizer tab."""
    with gr.TabItem("🗂️ Notes Organizer"):
        gr.Markdown("## Organize Notes into Outline")
        gr.Markdown("Drag notes from the left panel and drop them onto subtopics in the outline tree on the right.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Available Notes")
                
                # Search and filter controls
                with gr.Row():
                    search_input = gr.Textbox(
                        label="Search Notes",
                        placeholder="Type to search...",
                        scale=3
                    )
                    refresh_notes_btn = gr.Button("🔄", scale=1, size="sm")
                
                with gr.Row():
                    show_assigned_check = gr.Checkbox(label="Show Assigned", value=True)
                    show_unassigned_check = gr.Checkbox(label="Show Unassigned", value=True)
                
                # Notes list (HTML display)
                notes_display = gr.HTML(label="Notes")

                # Non-DnD reliable assignment UI
                gr.Markdown("### Assign Selected (Reliable)")
                notes_multiselect = gr.CheckboxGroup(label="Select Notes", choices=[], value=[])
                target_dropdown = gr.Dropdown(label="Target Subtopic", choices=[], value=None)
                assign_selected_btn = gr.Button("Assign Selected Notes", variant="primary")

                # Hidden bridge elements for JS-driven drops
                drop_receiver = gr.Textbox(
                    label="drop_payload",
                    visible=False,
                    elem_id="drop_receiver"
                )
                drop_apply_btn = gr.Button(
                    "Apply Drop",
                    visible=False,
                    elem_id="drop_apply"
                )
                
                # Manual assignment controls (fallback if drag-drop doesn't work)
                gr.Markdown("### Manual Assignment")
                with gr.Column():
                    selected_note_id = gr.Textbox(
                        label="Note ID",
                        placeholder="Paste note ID here",
                        scale=1
                    )
                    target_chapter_id = gr.Textbox(
                        label="Chapter ID",
                        placeholder="Paste chapter ID",
                        scale=1
                    )
                    target_subtopic_id = gr.Textbox(
                        label="Subtopic ID",
                        placeholder="Paste subtopic ID",
                        scale=1
                    )
                    manual_assign_btn = gr.Button("Assign Note", variant="primary")
                    assignment_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=1):
                gr.Markdown("### 📚 Book Outline")
                
                # Outline tree (HTML display)
                outline_display = gr.HTML(label="Outline Structure")
                
                refresh_outline_btn = gr.Button("🔄 Refresh Outline", variant="secondary")
                
                # Quick actions
                gr.Markdown("### Quick Actions")
                auto_organize_btn = gr.Button("🤖 Auto-Organize All Unassigned Notes", variant="primary")
                auto_organize_status = gr.Textbox(label="Auto-Organize Status", interactive=False, lines=5)
        
        # Statistics panel
        with gr.Row():
            stats_display = gr.Textbox(label="Organization Statistics", lines=8, interactive=False)
            refresh_stats_btn = gr.Button("📊 Refresh Statistics")
        
        # Event handlers
        def refresh_notes_handler(app_state, search, show_assigned, show_unassigned):
            notes_list = get_all_notes_list(app_state, search)
            return format_notes_as_html(notes_list, show_assigned, show_unassigned)
        
        def refresh_outline_handler(app_state):
            return format_outline_as_interactive_tree(app_state)
        
        def manual_assign_handler(app_state, note_id, chapter_id, subtopic_id):
            success, msg = assign_note_to_section(app_state, note_id, chapter_id, subtopic_id)
            # Refresh displays
            notes_html = refresh_notes_handler(app_state, "", True, True)
            outline_html = refresh_outline_handler(app_state)
            return msg, notes_html, outline_html

        def assign_from_drop(app_state, payload_json):
            """Handle drop payload coming from JS and assign the note."""
            try:
                payload = json.loads(payload_json or "{}")
                note_id = payload.get("note_id")
                chapter_id = payload.get("chapter_id")
                subtopic_id = payload.get("subtopic_id")
                if not (note_id and chapter_id and subtopic_id):
                    return "Invalid drop payload.", refresh_notes_handler(app_state, "", True, True), refresh_outline_handler(app_state)
                success, msg = assign_note_to_section(app_state, note_id, chapter_id, subtopic_id)
                # Refresh displays after assignment
                notes_html = refresh_notes_handler(app_state, "", True, True)
                outline_html = refresh_outline_handler(app_state)
                return msg, notes_html, outline_html
            except Exception as e:
                return f"Error handling drop: {e}", refresh_notes_handler(app_state, "", True, True), refresh_outline_handler(app_state)
        
        def auto_organize_handler(app_state):
            """Auto-organize all unassigned notes."""
            if not app_state:
                return "No project loaded.", refresh_notes_handler(app_state, "", True, True), refresh_outline_handler(app_state)
            
            if not app_state.current_outline:
                return "No outline loaded.", refresh_notes_handler(app_state, "", True, True), refresh_outline_handler(app_state)
            
            try:
                # Get all unassigned notes
                all_notes = app_state.note_processor.notes_collection.get(include=["documents", "metadatas"])
                all_note_ids = all_notes.get("ids", [])
                if not all_note_ids:
                    return "No notes to organize.", refresh_notes_handler(app_state, "", True, True), refresh_outline_handler(app_state)
                
                organized_count = 0
                skipped_count = 0
                error_count = 0
                
                all_note_docs = all_notes.get("documents", [])
                all_note_metadatas = all_notes.get("metadatas", [])
                
                for i, note_id in enumerate(all_note_ids):
                    note_text = all_note_docs[i] if i < len(all_note_docs) else ""
                    note_metadata = all_note_metadatas[i] if i < len(all_note_metadatas) else {}
                    
                    # Skip if already assigned
                    if note_metadata.get("chapter_id") and note_metadata.get("subtopic_id"):
                        skipped_count += 1
                        continue
                    
                    # Classify the note
                    try:
                        classification = app_state.content_expander.classify_content(
                            note_text,
                            app_state.current_outline.to_dict()
                        )
                        
                        if classification and classification.get("chapter") and classification.get("subtopic"):
                            # Assign the note
                            success, msg = assign_note_to_section(
                                app_state,
                                note_id,
                                classification["chapter"]["id"],
                                classification["subtopic"]["id"]
                            )
                            if success:
                                organized_count += 1
                            else:
                                error_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        error_count += 1
                        print(f"Error organising note {note_id}: {e}")
                
                result = f"""Auto-Organization Complete:\n- Total notes: {len(all_note_ids)}\n- Organized: {organized_count}\n- Skipped (already assigned): {skipped_count}\n- Errors: {error_count}\n"""
                # Refresh UI elements
                notes_html = refresh_notes_handler(app_state, "", True, True)
                outline_html = refresh_outline_handler(app_state)
                return result, notes_html, outline_html
            except Exception as e:
                return f"Error during auto-organisation: {str(e)}", refresh_notes_handler(app_state, "", True, True), refresh_outline_handler(app_state)
        
        def get_stats_handler(app_state):
            """Get organisation statistics."""
            if not app_state:
                return "No project loaded."
            if not app_state.current_outline:
                return "No outline loaded."
            
            try:
                all_notes = app_state.note_processor.notes_collection.get(include=["metadatas"])
                all_note_ids = all_notes.get("ids", [])
                if not all_note_ids:
                    return "No notes in the system."
                
                total_notes = len(all_note_ids)
                assigned_notes = 0
                unassigned_notes = 0
                
                all_note_metadatas = all_notes.get("metadatas", [])
                for i, note_id in enumerate(all_note_ids):
                    metadata = all_note_metadatas[i] if i < len(all_note_metadatas) else {}
                    
                    if metadata.get("chapter_id") and metadata.get("subtopic_id"):
                        assigned_notes += 1
                    else:
                        unassigned_notes += 1
                
                # Get outline structure info
                outline = app_state.current_outline.to_dict()
                total_parts = len(outline.get("parts", []))
                total_chapters = 0
                total_subtopics = 0
                
                for part in outline.get("parts", []):
                    total_chapters += len(part.get("chapters", []))
                    for chapter in part.get("chapters", []):
                        total_subtopics += len(chapter.get("subtopics", []))
                
                stats = f"""Organization Statistics
==================
Total Notes: {total_notes}
Assigned Notes: {assigned_notes}
Unassigned Notes: {unassigned_notes}

Outline Structure:
- Parts: {total_parts}
- Chapters: {total_chapters}
- Subtopics: {total_subtopics}

Progress: {(assigned_notes/total_notes*100):.1f}% organized
"""
                return stats
            except Exception as e:
                return f"Error calculating statistics: {e}"
        
        # Wire up events
        refresh_notes_btn.click(
            fn=refresh_notes_handler,
            inputs=[app_state_component, search_input, show_assigned_check, show_unassigned_check],
            outputs=[notes_display]
        )
        
        search_input.change(
            fn=lambda app_state, q, sa, su: (
                refresh_notes_handler(app_state, q, sa, su)
            ),
            inputs=[app_state_component, search_input, show_assigned_check, show_unassigned_check],
            outputs=[notes_display]
        )
        
        show_assigned_check.change(
            fn=refresh_notes_handler,
            inputs=[app_state_component, search_input, show_assigned_check, show_unassigned_check],
            outputs=[notes_display]
        )
        
        show_unassigned_check.change(
            fn=refresh_notes_handler,
            inputs=[app_state_component, search_input, show_assigned_check, show_unassigned_check],
            outputs=[notes_display]
        )
        
        refresh_outline_btn.click(
            fn=refresh_outline_handler,
            inputs=[app_state_component],
            outputs=[outline_display]
        )

        # Populate multiselect and targets initially and on search
        app_state_component.change(
            fn=refresh_multiselect_and_targets,
            inputs=[app_state_component, search_input],
            outputs=[notes_multiselect, target_dropdown, gr.State()]
        )
        search_input.change(
            fn=refresh_multiselect_and_targets,
            inputs=[app_state_component, search_input],
            outputs=[notes_multiselect, target_dropdown, gr.State()]
        )

        # Hidden state to carry label->id map
        labels_map_state = gr.State("")
        app_state_component.change(
            fn=lambda app_state, q: refresh_multiselect_and_targets(app_state, q),
            inputs=[app_state_component, search_input],
            outputs=[notes_multiselect, target_dropdown, labels_map_state]
        )
        search_input.change(
            fn=lambda app_state, q: refresh_multiselect_and_targets(app_state, q),
            inputs=[app_state_component, search_input],
            outputs=[notes_multiselect, target_dropdown, labels_map_state]
        )

        assign_selected_btn.click(
            fn=assign_selected_handler,
            inputs=[app_state_component, notes_multiselect, target_dropdown, labels_map_state],
            outputs=[assignment_status, notes_display, outline_display, notes_multiselect]
        )
        
        manual_assign_btn.click(
            fn=manual_assign_handler,
            inputs=[app_state_component, selected_note_id, target_chapter_id, target_subtopic_id],
            outputs=[assignment_status, notes_display, outline_display]
        )
        
        auto_organize_btn.click(
            fn=auto_organize_handler,
            inputs=[app_state_component],
            outputs=[auto_organize_status, notes_display, outline_display]
        )

        # Wire drop apply hidden button to backend assignment
        drop_apply_btn.click(
            fn=assign_from_drop,
            inputs=[app_state_component, drop_receiver],
            outputs=[assignment_status, notes_display, outline_display]
        )
        
        refresh_stats_btn.click(
            fn=get_stats_handler,
            inputs=[app_state_component],
            outputs=[stats_display]
        )
        
        # Initial load
        app_state_component.change(
            fn=lambda app_state: (
                refresh_notes_handler(app_state, "", True, True),
                refresh_outline_handler(app_state),
                get_stats_handler(app_state)
            ),
            inputs=[app_state_component],
            outputs=[notes_display, outline_display, stats_display]
        )
