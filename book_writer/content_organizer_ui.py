"""
Enhanced Content Organizer UI for Book Writer
Provides drag-and-drop-like interface for organizing notes into outline structure
"""
import gradio as gr
import json
import time
import math
from typing import List, Dict, Optional, Tuple

from book_writer.response_handler import validate_json_response


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
            border: 1px solid #374151; /* slate-700 */
            border-radius: 8px;
            background: #111827; /* gray-900 */
        }
        .note-item {
            background: #1f2937; /* gray-800 */
            border: 1px solid #4b5563; /* gray-600 */
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
            box-shadow: 0 2px 8px rgba(0,0,0,0.35);
            border-color: #10b981; /* emerald-500 */
        }
        .note-item.assigned {
            border-left: 4px solid #10b981; /* emerald-500 */
            background: #064e3b; /* emerald-900 */
        }
        .note-item.unassigned {
            border-left: 4px solid #f59e0b; /* amber-500 */
        }
        .drag-handle {
            font-size: 20px;
            color: #9ca3af; /* gray-400 */
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
            color: #d1d5db; /* gray-300 */
            font-family: monospace;
            margin-bottom: 4px;
        }
        .note-preview {
            font-size: 14px;
            color: #f3f4f6; /* gray-100 */
            line-height: 1.4;
        }
        .note-status {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 500;
        }
        .status-assigned {
            background: #065f46; /* emerald-800 */
            color: #a7f3d0; /* emerald-200 */
        }
        .status-unassigned {
            background: #78350f; /* amber-900 */
            color: #fde68a; /* amber-200 */
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
            border: 1px solid #374151; /* slate-700 */
            border-radius: 8px;
            background: #0b1220; /* near dark */
        }
        .outline-part {
            background: #111827; /* gray-900 */
            border: 1px solid #4b5563; /* gray-600 */
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .outline-part-header {
            font-weight: bold;
            font-size: 16px;
            color: #93c5fd; /* blue-300 */
            margin-bottom: 8px;
            padding: 8px;
            background: #1e3a8a; /* blue-900 */
            border-radius: 4px;
        }
        .outline-chapter {
            margin-left: 15px;
            margin-bottom: 8px;
            padding: 8px;
            background: #0f172a; /* slate-900 */
            border-radius: 4px;
        }
        .outline-chapter-header {
            font-weight: 600;
            font-size: 14px;
            color: #60a5fa; /* blue-400 */
            margin-bottom: 6px;
        }
        .outline-subtopic {
            margin-left: 15px;
            padding: 8px 12px;
            margin-bottom: 4px;
            background: #111827; /* gray-900 */
            border: 1px solid #4b5563; /* gray-600 */
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .outline-subtopic:hover {
            background: #1f2937; /* gray-800 */
            border-color: #f59e0b; /* amber-500 */
        }
        .outline-subtopic.drop-target {
            background: #065f46; /* emerald-800 */
            border: 2px dashed #10b981; /* emerald-500 */
        }
        .subtopic-title {
            font-size: 13px;
            color: #e5e7eb; /* gray-200 */
        }
        .note-count-badge {
            display: inline-block;
            background: #374151; /* gray-700 */
            color: #f9fafb; /* gray-50 */
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
                    e.currentTarget.style.background = '#065f46';
                    // Trigger backend assignment
                    btnEl.click();
                    setTimeout(() => {
                        e.currentTarget.style.background = '#111827';
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

        # Suggestions panel (moved from Advanced Organization)
        gr.Markdown("### 🔍 Review Suggestions")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=220):
                get_suggestions_btn = gr.Button("Get Suggestions", variant="secondary")
                suggestions_overview = gr.Markdown("No suggestions yet. Click **Get Suggestions** to analyze the outline.", elem_id="suggestions-overview")
            with gr.Column(scale=3):
                suggestions_output = gr.JSON(label="Suggestions", value={})

        # Actions for suggestions
        gr.Markdown("#### Take Action on Suggestions")
        suggestions_state = gr.State("")
        with gr.Row():
            suggestions_selector = gr.Dropdown(label="Select Suggestion (by section)", choices=[], value=None)
            idea_selector = gr.Dropdown(label="Select Idea/Prompt", choices=[], value=None)
        ai_plan_display = gr.JSON(label="AI Content Plan Preview", value={})
        with gr.Row():
            gen_pages_input = gr.Number(label="Target Pages", value=1, minimum=1, maximum=50, step=1)
            generate_content_btn = gr.Button("Generate Content", variant="primary")
            add_as_note_btn = gr.Button("Add as Note", variant="secondary")
        action_status = gr.Textbox(label="Action Status", interactive=False)
        
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

        def refresh_multiselect_and_targets(app_state, search):
            """Populate multi-select choices and target dropdown, return also labels map state."""
            notes_list = get_all_notes_list(app_state, search)
            choices = [f"{n['id']} : {n['preview'][:60]}" for n in notes_list]
            label_to_id = {f"{n['id']} : {n['preview'][:60]}": n['id'] for n in notes_list}
            target_options, _ = get_outline_target_options(app_state)
            return (
                gr.update(choices=choices, value=[]),
                gr.update(choices=target_options, value=None),
                json.dumps(label_to_id),
            )

        def assign_selected_handler(app_state, selected_labels, target_label, label_to_id_json):
            """Assign selected notes to chosen subtopic reliably (non-DnD)."""
            if not app_state:
                return (
                    "No project loaded.",
                    refresh_notes_handler(app_state, "", True, True),
                    refresh_outline_handler(app_state),
                    gr.update(value=selected_labels or []),
                )
            if not target_label:
                return (
                    "Please select a target subtopic.",
                    refresh_notes_handler(app_state, "", True, True),
                    refresh_outline_handler(app_state),
                    gr.update(value=selected_labels or []),
                )
            try:
                label_to_id = json.loads(label_to_id_json or "{}")
            except Exception:
                label_to_id = {}
            note_ids = [label_to_id.get(lbl, (lbl.split(" : ")[0] if isinstance(lbl, str) else lbl)) for lbl in (selected_labels or [])]
            _, mapping = get_outline_target_options(app_state)
            if target_label not in mapping:
                return (
                    "Invalid target selected.",
                    refresh_notes_handler(app_state, "", True, True),
                    refresh_outline_handler(app_state),
                    gr.update(value=selected_labels or []),
                )
            chapter_id, subtopic_id = mapping[target_label]
            for nid in note_ids:
                assign_note_to_section(app_state, nid, chapter_id, subtopic_id)
            # Refresh and clear selection
            return (
                f"Assigned {len(note_ids)} note(s) to {target_label}",
                refresh_notes_handler(app_state, "", True, True),
                refresh_outline_handler(app_state),
                gr.update(value=[]),
            )

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

        def generate_ai_plan_for_suggestion(app_state, suggestion_item):
            """Generate a structured AI plan with topic, subtopics, and draft content."""
            fallback_plan = {
                "topic": suggestion_item.get("path", "Generated Section"),
                "subtopics": [],
                "draft_content": "",
                "recommended_prompt": "",
            }
            try:
                if not app_state or not getattr(app_state, "content_expander", None):
                    return fallback_plan
                mm = app_state.content_expander.model_manager
                if not mm:
                    return fallback_plan
                pages_needed = max(1, int(math.ceil(suggestion_item.get("pages_needed", 1) or 1)))
                outline_path = suggestion_item.get("path", "Current Section")
                context_lines = []
                if suggestion_item.get("topic_ideas"):
                    ideas_preview = suggestion_item["topic_ideas"][:5]
                    context_lines.append("Ideas already considered: " + "; ".join(ideas_preview))
                if suggestion_item.get("action_items"):
                    context_lines.append("Action items: " + ", ".join(suggestion_item["action_items"][:5]))
                context = "\n".join(context_lines)
                prompt = f"""
You are an expert book writing strategist. Plan a detailed section for the book area "{outline_path}".
Target length: approximately {pages_needed} page(s).
{context}

Return only valid JSON with the following schema:
{{
  "topic": string,                          # clear high-level topic for the section
  "subtopics": [
    {{
      "title": string,
      "summary": string,
      "talking_points": [string],
      "estimated_paragraphs": int,
      "draft_paragraph": string             # 1-2 paragraphs covering this subtopic
    }}
  ],
  "draft_content": string,                  # cohesive draft ~{pages_needed} page(s)
  "recommended_prompt": string              # prompt tailored for a writing model
}}
Ensure every string is plain text (no markdown bullets).
""".strip()
                response = mm.generate_response(
                    prompt=prompt,
                    task="content_expansion",
                    format_json=True,
                    temperature=0.4,
                    top_p=0.8,
                )
                if isinstance(response, dict):
                    plan = response
                else:
                    plan = validate_json_response(response)
                if not isinstance(plan, dict):
                    return fallback_plan
                plan.setdefault("topic", fallback_plan["topic"])
                if not isinstance(plan.get("subtopics"), list):
                    plan["subtopics"] = []
                plan.setdefault("draft_content", "")
                plan.setdefault("recommended_prompt", "")
                return plan
            except Exception as e:
                fallback_plan["error"] = str(e)
                return fallback_plan

        def build_suggestions_summary_markdown(suggestions_data):
            if not suggestions_data:
                return "**Summary**\n- **Suggestions**: None yet\n- **Action**: Click the button to analyze your outline."
            overall = suggestions_data.get("overall") or {}
            gap_items = suggestions_data.get("page_gap_suggestions") or []
            total_needed = overall.get("total_pages_needed")
            remaining = overall.get("remaining_pages_to_target")
            target = overall.get("target_pages")
            written = overall.get("written_pages")
            summary_lines = ["**Summary**"]
            summary_lines.append(f"- **Suggestions**: {len(gap_items)} ready to review")
            if total_needed is not None:
                summary_lines.append(f"- **Gap estimate**: {round(total_needed, 2)} page(s) suggested")
            if target is not None and written is not None:
                summary_lines.append(f"- **Progress**: {round(written or 0, 2)} / {float(target):.0f} pages")
            if remaining is not None:
                summary_lines.append(f"- **Remaining to target**: {round(max(0.0, remaining or 0), 2)} page(s)")
            if not gap_items:
                summary_lines.append("- **Mode**: Enhancement prompts generated (no major gaps detected)")
            return "\n".join(summary_lines)

        def suggestions_state_to_outputs(suggestions_json_str):
            try:
                data = json.loads(suggestions_json_str or "{}")
            except Exception:
                data = {}
            summary_md = build_suggestions_summary_markdown(data)
            return data, summary_md

        def get_suggestions_handler(app_state):
            """Combine organization suggestions with rich, page-target-driven expansion suggestions.
            Augment with AI-refined topic ideas in strict JSON when possible.
            """
            if not app_state:
                return {"error": "No project loaded."}
            try:
                # Existing organization suggestions (if implemented)
                try:
                    org_suggestions = app_state.get_organization_suggestions()
                except Exception as _:
                    org_suggestions = {}

                # Page-gap suggestions from progress
                page_gap_suggestions = []
                total_pages_needed = 0.0
                action_catalog = [
                    "Add a concrete example to illustrate the concept",
                    "Introduce a short case study",
                    "Provide a step-by-step walkthrough",
                    "Clarify key terms and definitions",
                    "Contrast with an alternative approach",
                    "Add a visual description/diagram explanation",
                    "Include a checklist or summary bullets",
                    "Address common pitfalls or misconceptions",
                    "Add a mini-FAQ (2-3 questions)",
                    "Provide a short exercise or reflection prompt",
                ]
                try:
                    progress = app_state.get_writing_progress()
                    for part in (progress.get("progress_by_section") or []):
                        part_title = part.get("title", "Part")
                        for chapter in (part.get("chapters") or []):
                            chapter_title = chapter.get("title", "Chapter")
                            for sub in (chapter.get("subtopics") or []):
                                target = float(sub.get("target_pages", 0) or 0)
                                written = float(sub.get("written_pages", 0) or 0)
                                gap = max(0.0, target - written)
                                if gap > 0.05:  # suggest only when meaningful gap exists
                                    total_pages_needed += gap
                                    # Create multiple actionable items proportional to the gap
                                    # Heuristic: ~2 actionable items per missing page
                                    items_count = max(2, int(math.ceil(gap * 2)))
                                    detailed_actions = []
                                    for i in range(items_count):
                                        # Rotate through the catalog for variety
                                        action = action_catalog[i % len(action_catalog)]
                                        detailed_actions.append(action)

                                    # Generate topic-specific new content ideas proportional to the gap
                                    # Heuristic: ~3 ideas per missing page (capped to avoid overload)
                                    ideas_count = max(3, min(12, int(math.ceil(gap * 3))))
                                    sub_title = sub.get('title', 'Subtopic')
                                    idea_templates = [
                                        "Foundations: Key concepts behind {}",
                                        "Historical context and evolution of {}",
                                        "When and why to use {} (use-cases)",
                                        "Step-by-step tutorial: Getting started with {}",
                                        "Deep dive: Advanced techniques in {}",
                                        "Common pitfalls and how to avoid them in {}",
                                        "Comparative analysis: {} vs alternative approaches",
                                        "Mini case study: Applying {} in a real scenario",
                                        "Checklist: Best practices for {}",
                                        "FAQ: Frequently asked questions about {}",
                                        "Evaluation metrics and benchmarks for {}",
                                        "Future trends and open challenges in {}",
                                    ]
                                    topic_ideas = [idea_templates[i % len(idea_templates)].format(sub_title) for i in range(ideas_count)]

                                    # Provide ready-to-use prompts to generate content
                                    suggested_prompts = [
                                        f"Write ~{int(max(1, math.ceil(gap/ideas_count*2)))} pages elaborating: '{idea}'. Include examples and a short summary." 
                                        for idea in topic_ideas[:min(ideas_count, 10)]
                                    ]

                                    suggestion_item = {
                                        "path": f"{part_title} > {chapter_title} > {sub.get('title','Subtopic')}",
                                        "chapter_id": chapter.get("id"),
                                        "subtopic_id": sub.get("id"),
                                        "target_pages": round(target, 2),
                                        "written_pages": round(written, 2),
                                        "pages_needed": round(gap, 2),
                                        "recommended_actions": [
                                            f"Expand content by ~{int(math.ceil(gap))} page(s)",
                                            "Add/expand notes mapped to this subtopic",
                                        ],
                                        "action_items": detailed_actions,
                                        "topic_ideas": topic_ideas,
                                        "suggested_prompts": suggested_prompts,
                                    }

                                    # Optional: AI-refine topic ideas reliably via JSON
                                    try:
                                        mm = app_state.content_expander.model_manager
                                        model_cfg = mm.config.get_model_config("content_expansion")
                                        model_name = model_cfg["model_name"]
                                        refine_prompt = (
                                            "You are helping expand a book section. Given the section path, propose 5-10 concise, high-impact topic ideas as JSON.\n"
                                            "Return strictly valid JSON with a single key 'ideas' as an array of strings. No extra commentary.\n\n"
                                            f"Section: {suggestion_item['path']}\n"
                                        )
                                        resp = mm.ollama_client.chat(
                                            model=model_name,
                                            messages=[{"role": "user", "content": refine_prompt}],
                                            options={"temperature": 0.6, "top_p": 0.9},
                                        )
                                        content = resp.get("message", {}).get("content", "") if isinstance(resp, dict) else ""
                                        # Safe JSON extraction
                                        refined = {}
                                        try:
                                            refined = json.loads(content)
                                        except Exception:
                                            # Try to find JSON block
                                            start = content.find('{')
                                            end = content.rfind('}')
                                            if start != -1 and end != -1 and end > start:
                                                refined = json.loads(content[start:end+1])
                                        if isinstance(refined, dict) and isinstance(refined.get('ideas'), list):
                                            refined_ideas = [str(x) for x in refined['ideas'] if isinstance(x, (str,))]
                                            # Merge unique
                                            existing = set(suggestion_item["topic_ideas"]) if suggestion_item.get("topic_ideas") else set()
                                            merged = list(existing | set(refined_ideas))
                                            suggestion_item["topic_ideas"] = merged
                                            # Build prompts from refined ideas too
                                            for idea in refined_ideas:
                                                suggestion_item["suggested_prompts"].append(
                                                    f"Write ~1 page elaborating: '{idea}'. Include examples and transitions."
                                                )
                                    except Exception:
                                        pass

                                    page_gap_suggestions.append(suggestion_item)
                except Exception as _:
                    pass

                # Sort by largest gap first
                page_gap_suggestions.sort(key=lambda x: x.get("pages_needed", 0), reverse=True)

                # If no meaningful gaps found, generate enhancement suggestions to keep momentum
                if not page_gap_suggestions:
                    enhancement_candidates = []
                    progress_sections = (progress or {}).get("progress_by_section") if 'progress' in locals() else None
                    if progress_sections:
                        for part in progress_sections or []:
                            part_title = part.get("title", "Part")
                            for chapter in part.get("chapters") or []:
                                chapter_title = chapter.get("title", "Chapter")
                                for sub in chapter.get("subtopics") or []:
                                    target = float(sub.get("target_pages", 0) or 0)
                                    if target <= 0:
                                        continue
                                    written = float(sub.get("written_pages", 0) or 0)
                                    desired_extra = max(1.0, round(max(0.5, target * 0.25), 2))
                                    sub_title = sub.get("title", "Subtopic")
                                    topic_ideas = [
                                        f"Add advanced insights for {sub_title}",
                                        f"Incorporate a case study highlighting {sub_title}",
                                        f"Summarize key takeaways for {sub_title} with actionable steps",
                                    ]
                                    prompts = [
                                        f"Write ~1 page expanding on {sub_title} with new examples and updated research.",
                                        f"Draft a narrative that illustrates {sub_title} through a real-world scenario.",
                                        f"Create a concise summary and checklist for {sub_title} to reinforce learning.",
                                    ]
                                    enhancement_candidates.append({
                                        "path": f"{part_title} > {chapter_title} > {sub_title}",
                                        "chapter_id": chapter.get("id"),
                                        "subtopic_id": sub.get("id"),
                                        "target_pages": round(target, 2),
                                        "written_pages": round(written, 2),
                                        "pages_needed": desired_extra,
                                        "recommended_actions": [
                                            "Polish and extend existing content with richer detail",
                                            "Add illustrative stories, data, or expert quotes",
                                            "Ensure transitions and conclusions feel complete",
                                        ],
                                        "action_items": [
                                            "Review current paragraphs for depth gaps",
                                            "Add at least one fresh example or anecdote",
                                            "Highlight a key takeaway with a short summary",
                                        ],
                                        "topic_ideas": topic_ideas,
                                        "suggested_prompts": prompts,
                                        "suggestion_type": "enhancement",
                                    })
                    if enhancement_candidates:
                        enhancement_candidates.sort(key=lambda x: (x.get("target_pages", 0), -x.get("written_pages", 0)), reverse=True)
                        page_gap_suggestions = enhancement_candidates[:5]

                # Attach AI plans to each suggestion for automatic topic/subtopic/content generation
                for suggestion in page_gap_suggestions:
                    plan = generate_ai_plan_for_suggestion(app_state, suggestion)
                    suggestion["ai_plan"] = plan
                    # Prime prompts/ideas from plan for convenience
                    recommended_prompt = plan.get("recommended_prompt")
                    if recommended_prompt:
                        prompts = suggestion.get("suggested_prompts") or []
                        if recommended_prompt not in prompts:
                            suggestion.setdefault("suggested_prompts", [])
                            suggestion["suggested_prompts"].insert(0, recommended_prompt)
                    subtopics = plan.get("subtopics") or []
                    auto_ideas = []
                    for sub in subtopics:
                        if isinstance(sub, dict):
                            title = sub.get("title")
                            summary = sub.get("summary")
                            if title and summary:
                                auto_ideas.append(f"{title}: {summary}")
                    if auto_ideas:
                        suggestion.setdefault("topic_ideas", [])
                        for idea in auto_ideas:
                            if idea not in suggestion["topic_ideas"]:
                                suggestion["topic_ideas"].append(idea)

                # Overall plan to reach target
                overall = {
                    "total_pages_needed": round(total_pages_needed, 2),
                }
                # Add pacing guidance if target pages exist
                try:
                    total_target_pages = (app_state.config or {}).get("target_pages", 0) or 0
                    total_written_pages = float(progress.get("total_written_pages", 0)) if 'progress' in locals() and progress else 0.0
                    remaining = max(0.0, float(total_target_pages) - total_written_pages)
                    overall.update({
                        "target_pages": float(total_target_pages),
                        "written_pages": round(total_written_pages, 2),
                        "remaining_pages_to_target": round(remaining, 2),
                        "pacing_suggestion": "Aim for ~{} page(s)/day to hit the target in 2 weeks.".format(int(math.ceil(remaining/14))) if remaining > 0 else "Target met or not set.",
                    })
                except Exception:
                    pass

                result = {
                    "overall": overall,
                    "page_gap_suggestions": page_gap_suggestions,
                    "organization_suggestions": org_suggestions,
                }
                return result
            except Exception as e:
                return {"error": str(e)}

        def build_suggestion_selectors(_, suggestions_json_str):
            """Populate selectors from suggestions JSON string."""
            try:
                data = json.loads(suggestions_json_str or "{}")
                items = data.get("page_gap_suggestions", [])
            except Exception:
                items = []
            # Build display labels keyed by index
            labels = []
            first_plan = {}
            for idx, it in enumerate(items):
                path = it.get("path", f"Suggestion {idx+1}")
                gap = it.get("pages_needed", 0)
                labels.append(f"{idx}: {path} (need ~{gap} pages)")
            if items:
                first_plan = items[0].get("ai_plan") or {}
            return (
                gr.update(choices=labels, value=(labels[0] if labels else None)),
                gr.update(choices=[], value=None),
                first_plan,
            )

        def on_select_suggestion(_, suggestions_json_str, selected_label):
            """Populate idea selector when a suggestion is chosen."""
            try:
                data = json.loads(suggestions_json_str or "{}")
                items = data.get("page_gap_suggestions", [])
            except Exception:
                items = []
            if not selected_label:
                return gr.update(choices=[], value=None), {}
            try:
                idx = int(selected_label.split(":", 1)[0])
            except Exception:
                idx = 0
            ideas = []
            plan = {}
            if 0 <= idx < len(items):
                ideas = (items[idx].get("suggested_prompts") or []) + (items[idx].get("topic_ideas") or [])
                plan = items[idx].get("ai_plan") or {}
            # Limit to a reasonable size
            ideas = ideas[:50]
            return gr.update(choices=ideas, value=(ideas[0] if ideas else None)), plan

        def generate_content_from_suggestion(app_state, suggestions_json_str, selected_label, selected_idea, pages):
            if not app_state:
                return "No project loaded."
            try:
                data = json.loads(suggestions_json_str or "{}")
                items = data.get("page_gap_suggestions", [])
                idx = int(selected_label.split(":", 1)[0]) if selected_label else 0
                item = items[idx] if 0 <= idx < len(items) else None
                if not item:
                    return "Invalid selection."
                chapter_id = item.get("chapter_id")
                subtopic_id = item.get("subtopic_id")
                user_prompt = selected_idea or (item.get("suggested_prompts") or ["Expand this section."])[0]
                try:
                    p = int(pages) if pages else 1
                except Exception:
                    p = 1
                p = max(1, min(50, p))
                plan = item.get("ai_plan") or {}
                plan_topic = plan.get("topic")
                plan_subtopics = plan.get("subtopics") if isinstance(plan.get("subtopics"), list) else []
                plan_outline_lines = []
                for i, sub in enumerate(plan_subtopics, 1):
                    if isinstance(sub, dict):
                        title = sub.get("title") or f"Subtopic {i}"
                        summary = sub.get("summary") or ""
                        talking_points = sub.get("talking_points") or []
                        outline_line = f"{i}. {title} — {summary}"
                        if talking_points:
                            outline_line += " | Key points: " + "; ".join(talking_points[:5])
                        plan_outline_lines.append(outline_line)
                plan_outline = "\n".join(plan_outline_lines)
                draft_content = plan.get("draft_content", "")
                prompt_sections = [
                    f"Write ~{p} page(s) for book section: {item.get('path', 'Selected Section')}.",
                    f"User guidance: {user_prompt}",
                ]
                if plan_topic:
                    prompt_sections.append(f"Primary topic focus: {plan_topic}")
                if plan_outline:
                    prompt_sections.append("Outline to follow:\n" + plan_outline)
                if draft_content:
                    prompt_sections.append("Reference draft (revise, improve, and expand where helpful):\n" + draft_content)
                prompt_sections.append("Deliver polished prose with smooth transitions and cohesive narrative.")
                prompt = "\n\n".join(prompt_sections)
                # Call app method to generate content
                content_id = app_state.generate_content(prompt, chapter_id, subtopic_id)
                return f"Generated content: {content_id} for {item.get('path')}"
            except Exception as e:
                return f"Error generating content: {e}"

        def add_note_from_suggestion(app_state, suggestions_json_str, selected_label, selected_idea):
            if not app_state:
                return "No project loaded."
            try:
                data = json.loads(suggestions_json_str or "{}")
                items = data.get("page_gap_suggestions", [])
                idx = int(selected_label.split(":", 1)[0]) if selected_label else 0
                item = items[idx] if 0 <= idx < len(items) else None
                if not item:
                    return "Invalid selection."
                chapter_id = item.get("chapter_id")
                subtopic_id = item.get("subtopic_id")
                plan = item.get("ai_plan") or {}
                subtopics = plan.get("subtopics") if isinstance(plan.get("subtopics"), list) else []
                plan_snippets = []
                for sub in subtopics[:3]:
                    if isinstance(sub, dict):
                        title = sub.get("title")
                        summary = sub.get("summary")
                        if title and summary:
                            plan_snippets.append(f"{title}: {summary}")
                draft_content = plan.get("draft_content", "")
                note_text = selected_idea or draft_content
                if not note_text:
                    note_text = "\n".join(plan_snippets) if plan_snippets else (item.get("topic_ideas") or ["New note idea"])[0]
                note_id = app_state.process_note(note_text, source="suggestion")
                # Assign the note to the selected section
                assign_note_to_section(app_state, note_id, chapter_id, subtopic_id)
                return f"Added note {note_id[:8]}... to {item.get('path')}"
            except Exception as e:
                return f"Error adding note: {e}"
        
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

        # Hidden state to carry label->id map
        labels_map_state = gr.State("")
        # Populate multiselect and targets initially and on search
        app_state_component.change(
            fn=refresh_multiselect_and_targets,
            inputs=[app_state_component, search_input],
            outputs=[notes_multiselect, target_dropdown, labels_map_state]
        )
        search_input.change(
            fn=refresh_multiselect_and_targets,
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

        # Wire suggestions button
        get_suggestions_btn.click(
            fn=lambda app_state: (
                json.dumps(get_suggestions_handler(app_state))
            ),
            inputs=[app_state_component],
            outputs=[suggestions_state]
        )

        # When suggestions state changes, show parsed JSON and populate selectors
        suggestions_state.change(
            fn=suggestions_state_to_outputs,
            inputs=[suggestions_state],
            outputs=[suggestions_output, suggestions_overview]
        )
        suggestions_state.change(
            fn=build_suggestion_selectors,
            inputs=[app_state_component, suggestions_state],
            outputs=[suggestions_selector, idea_selector, ai_plan_display]
        )
        suggestions_selector.change(
            fn=on_select_suggestion,
            inputs=[app_state_component, suggestions_state, suggestions_selector],
            outputs=[idea_selector, ai_plan_display]
        )
        generate_content_btn.click(
            fn=generate_content_from_suggestion,
            inputs=[app_state_component, suggestions_state, suggestions_selector, idea_selector, gen_pages_input],
            outputs=[action_status]
        )
        add_as_note_btn.click(
            fn=add_note_from_suggestion,
            inputs=[app_state_component, suggestions_state, suggestions_selector, idea_selector],
            outputs=[action_status]
        )
        
        # Initial load
        app_state_component.change(
            fn=lambda app_state: (
                refresh_notes_handler(app_state, "", True, True),
                refresh_outline_handler(app_state),
                get_stats_handler(app_state),
                json.dumps(get_suggestions_handler(app_state))
            ),
            inputs=[app_state_component],
            outputs=[notes_display, outline_display, stats_display, suggestions_state]
        )
