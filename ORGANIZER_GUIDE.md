# Notes Organizer Guide

## Overview
The **Notes Organizer** is a new drag-and-drop interface for organizing your notes into your book's outline structure. It provides a visual, intuitive way to assign notes to chapters and subtopics.

## Features

### 🎯 Visual Organization
- **Left Panel**: All your notes with search and filtering
- **Right Panel**: Your book outline tree structure
- **Drag & Drop**: Drag notes from left and drop onto subtopics on the right

### 🔍 Search & Filter
- **Search Box**: Find notes by content or ID
- **Show Assigned**: Toggle visibility of already-assigned notes
- **Show Unassigned**: Toggle visibility of unassigned notes

### 🤖 Auto-Organization
- **One-Click Auto-Organize**: Automatically classify and assign all unassigned notes using AI
- **Smart Classification**: Uses your existing content classification system

### 📊 Statistics
- View organization progress
- See how many notes are assigned vs unassigned
- Track coverage across your outline structure

## How to Use

### Method 1: Drag and Drop (Visual)
1. Open the **🗂️ Notes Organizer** tab
2. Find a note in the left panel
3. Drag it to a subtopic in the right panel
4. Drop it to assign

### Method 2: Manual Assignment (Fallback)
If drag-and-drop doesn't work in your browser:
1. Copy the **Note ID** from the left panel
2. Copy the **Chapter ID** and **Subtopic ID** from the outline
3. Paste them into the "Manual Assignment" section
4. Click **Assign Note**

### Method 3: Auto-Organize (Bulk)
1. Click **🤖 Auto-Organize All Unassigned Notes**
2. Wait for the AI to classify and assign all notes
3. Review the results in the status box

## Visual Indicators

### Notes Panel
- **Green border**: Note is already assigned
- **Orange border**: Note is unassigned
- **⋮⋮ icon**: Drag handle (grab here to drag)

### Outline Panel
- **📚 Icon**: Part
- **📖 Icon**: Chapter
- **📄 Icon**: Subtopic (drop target)
- **Badge**: Shows number of notes assigned to each subtopic

## Tips

1. **Start with Auto-Organize**: Let the AI do the heavy lifting first
2. **Review and Adjust**: Manually reassign notes that were misclassified
3. **Use Search**: Filter notes by keywords to organize related content together
4. **Check Statistics**: Monitor your progress regularly

## Technical Details

### How It Works
- Notes are stored in ChromaDB with metadata
- Assignment updates the `chapter_id` and `subtopic_id` metadata fields
- The UI uses HTML/CSS/JavaScript for drag-and-drop
- Gradio handles the backend communication

### Browser Compatibility
- Works best in modern browsers (Chrome, Firefox, Edge)
- Drag-and-drop requires JavaScript enabled
- Fallback manual assignment always available

## Troubleshooting

### Drag-and-drop not working?
- Use the **Manual Assignment** section instead
- Check if JavaScript is enabled in your browser
- Try refreshing the page

### Notes not showing up?
- Click the **🔄 Refresh** button
- Make sure you've loaded a project
- Check that you've added notes in the Writer's Desk tab

### Auto-organize not working?
- Ensure you have an outline loaded
- Check that your notes have content (not empty)
- Review the status message for specific errors

## Integration with Existing Features

The Notes Organizer works seamlessly with:
- **Writer's Desk**: Add notes there, organize them here
- **Content Expansion**: Organized notes can be expanded into full content
- **Progress Tracking**: See how organization affects your writing progress
- **Book Assembly**: Well-organized notes lead to better book structure

## Next Steps

After organizing your notes:
1. Go to **✍️ Writer's Desk** to expand notes into full content
2. Check **📊 Progress & Gamification** to see your progress
3. Use **📚 Outline & Assembly** to build and export your book

---

**Need Help?** The organizer is designed to be intuitive, but if you have questions, refer to the main README or check the inline help text in the UI.
