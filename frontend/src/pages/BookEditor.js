import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import { API } from "../App";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Plus, Sparkles, BookOpen, Lightbulb, CheckCircle, AlertCircle } from "lucide-react";
import { toast } from "sonner";

const BookEditor = () => {
  const { bookId } = useParams();
  const navigate = useNavigate();
  const [book, setBook] = useState(null);
  const [notes, setNotes] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expandingNote, setExpandingNote] = useState(null);
  
  const [newNote, setNewNote] = useState("");
  const [targetWords, setTargetWords] = useState([500]);
  const [writingStyle, setWritingStyle] = useState("creative");

  useEffect(() => {
    fetchData();
  }, [bookId]);

  const fetchData = async () => {
    try {
      const [bookRes, notesRes, suggestionsRes] = await Promise.all([
        axios.get(`${API}/books/${bookId}`),
        axios.get(`${API}/notes/${bookId}`),
        axios.get(`${API}/suggestions/${bookId}`)
      ]);
      setBook(bookRes.data);
      setNotes(notesRes.data);
      setSuggestions(suggestionsRes.data);
    } catch (error) {
      console.error("Error fetching data:", error);
      toast.error("Failed to load book data");
    } finally {
      setLoading(false);
    }
  };

  const createNote = async () => {
    if (!newNote.trim()) {
      toast.error("Please enter some content");
      return;
    }

    try {
      const response = await axios.post(`${API}/notes`, {
        book_id: bookId,
        content: newNote,
        target_words: targetWords[0]
      });
      setNotes([response.data, ...notes]);
      setNewNote("");
      toast.success("Note created!");
    } catch (error) {
      console.error("Error creating note:", error);
      toast.error("Failed to create note");
    }
  };

  const expandNote = async (noteId) => {
    setExpandingNote(noteId);
    
    try {
      const response = await fetch(`${API}/expand`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          note_id: noteId,
          book_id: bookId,
          style: writingStyle,
          target_words: targetWords[0]
        })
      });

      if (!response.ok) {
        throw new Error('Failed to expand note');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let expandedContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            try {
              const parsed = JSON.parse(data);
              if (parsed.content) {
                expandedContent += parsed.content;
                // Update UI in real-time
                setNotes(prevNotes => 
                  prevNotes.map(note => 
                    note.id === noteId 
                      ? { ...note, expanded_content: expandedContent }
                      : note
                  )
                );
              }
              if (parsed.done) {
                toast.success("Note expanded successfully!");
                fetchData(); // Refresh to get updated stats
              }
              if (parsed.error) {
                toast.error(parsed.error);
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } catch (error) {
      console.error("Error expanding note:", error);
      toast.error(error.message || "Failed to expand note. Check API settings.");
    } finally {
      setExpandingNote(null);
    }
  };

  const generateSuggestions = async () => {
    try {
      toast.info("Generating suggestions...");
      const response = await axios.post(`${API}/suggestions/generate/${bookId}`);
      setSuggestions(response.data.suggestions || []);
      toast.success("Suggestions generated!");
      fetchData(); // Refresh suggestions
    } catch (error) {
      console.error("Error generating suggestions:", error);
      toast.error("Failed to generate suggestions");
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button variant="ghost" size="sm" onClick={() => navigate('/')} data-testid="back-button">
                <ArrowLeft className="w-4 h-4" />
              </Button>
              <div>
                <h1 className="text-xl font-bold text-gray-900">{book?.title}</h1>
                <p className="text-sm text-gray-500">{book?.book_type} • {book?.total_words?.toLocaleString() || 0} words</p>
              </div>
            </div>
            <Badge variant="outline" className="text-sm" data-testid="book-status-badge">
              {book?.status}
            </Badge>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Notes Input */}
          <div className="lg:col-span-2 space-y-6">
            {/* New Note Card */}
            <Card className="bg-white/70 backdrop-blur-sm" data-testid="new-note-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Plus className="w-5 h-5 text-indigo-500" />
                  Add New Note
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>Your Ideas & Notes</Label>
                  <Textarea
                    data-testid="note-content-input"
                    placeholder="Write your raw ideas, plot points, character notes, or any content you want to expand..."
                    value={newNote}
                    onChange={(e) => setNewNote(e.target.value)}
                    rows={6}
                    className="mt-2"
                  />
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Target Words: {targetWords[0]}</Label>
                    <Slider
                      data-testid="target-words-slider"
                      value={targetWords}
                      onValueChange={setTargetWords}
                      min={100}
                      max={5000}
                      step={100}
                      className="mt-2"
                    />
                  </div>
                  <div>
                    <Label>Writing Style</Label>
                    <Select value={writingStyle} onValueChange={setWritingStyle}>
                      <SelectTrigger className="mt-2" data-testid="writing-style-select">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="professional">Professional</SelectItem>
                        <SelectItem value="creative">Creative</SelectItem>
                        <SelectItem value="conversational">Conversational</SelectItem>
                        <SelectItem value="academic">Academic</SelectItem>
                        <SelectItem value="poetic">Poetic</SelectItem>
                        <SelectItem value="technical">Technical</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <Button 
                  onClick={createNote} 
                  className="w-full bg-indigo-600 hover:bg-indigo-700"
                  data-testid="create-note-button"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Create Note
                </Button>
              </CardContent>
            </Card>

            {/* Notes List */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold text-gray-900">Your Notes</h2>
                <Badge variant="outline" data-testid="notes-count-badge">{notes.length} notes</Badge>
              </div>

              {notes.length === 0 ? (
                <Card className="bg-white/60 backdrop-blur-sm" data-testid="empty-notes-state">
                  <CardContent className="flex flex-col items-center justify-center py-12">
                    <BookOpen className="w-12 h-12 text-gray-300 mb-3" />
                    <p className="text-gray-500">No notes yet. Create your first note above!</p>
                  </CardContent>
                </Card>
              ) : (
                <ScrollArea className="h-[calc(100vh-500px)]">
                  <div className="space-y-4 pr-4">
                    {notes.map((note) => (
                      <Card key={note.id} className="bg-white/70 backdrop-blur-sm" data-testid={`note-${note.id}`}>
                        <CardContent className="pt-6 space-y-4">
                          {/* Original Note */}
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <Badge variant="outline" className="text-xs">Original Note</Badge>
                              <span className="text-xs text-gray-500">{note.word_count} words</span>
                            </div>
                            <p className="text-sm text-gray-700 whitespace-pre-wrap">{note.content}</p>
                          </div>

                          {/* Expanded Content */}
                          {note.expanded_content && (
                            <div className="border-t pt-4">
                              <div className="flex items-center gap-2 mb-2">
                                <Badge className="text-xs bg-indigo-600">Expanded Content</Badge>
                                <CheckCircle className="w-4 h-4 text-green-500" />
                              </div>
                              <p className="text-sm text-gray-900 whitespace-pre-wrap leading-relaxed">
                                {note.expanded_content}
                              </p>
                            </div>
                          )}

                          {/* Expand Button */}
                          {!note.expanded_content && (
                            <Button
                              onClick={() => expandNote(note.id)}
                              disabled={expandingNote === note.id}
                              className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
                              data-testid={`expand-note-${note.id}`}
                            >
                              {expandingNote === note.id ? (
                                <>
                                  <div className="spinner mr-2"></div>
                                  Expanding...
                                </>
                              ) : (
                                <>
                                  <Sparkles className="w-4 h-4 mr-2" />
                                  Expand with AI
                                </>
                              )}
                            </Button>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </div>
          </div>

          {/* Right Panel - Suggestions */}
          <div className="space-y-6">
            <Card className="bg-white/70 backdrop-blur-sm" data-testid="suggestions-panel">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Lightbulb className="w-5 h-5 text-yellow-500" />
                  AI Suggestions
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button
                  onClick={generateSuggestions}
                  variant="outline"
                  className="w-full"
                  data-testid="generate-suggestions-button"
                >
                  <Sparkles className="w-4 h-4 mr-2" />
                  Generate Suggestions
                </Button>

                <ScrollArea className="h-[calc(100vh-300px)]">
                  {suggestions.length === 0 ? (
                    <div className="text-center py-8 text-gray-500 text-sm" data-testid="no-suggestions">
                      No suggestions yet. Generate some to get AI insights!
                    </div>
                  ) : (
                    <div className="space-y-3 pr-4">
                      {suggestions.map((suggestion) => (
                        <div
                          key={suggestion.id}
                          className="p-3 rounded-lg border bg-white/50"
                          data-testid={`suggestion-${suggestion.id}`}
                        >
                          <div className="flex items-start gap-2 mb-2">
                            <AlertCircle 
                              className={`w-4 h-4 mt-0.5 ${
                                suggestion.severity === 'high' ? 'text-red-500' :
                                suggestion.severity === 'medium' ? 'text-yellow-500' :
                                'text-blue-500'
                              }`}
                            />
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-1">
                                <Badge 
                                  variant="outline" 
                                  className="text-xs"
                                >
                                  {suggestion.type.replace('_', ' ')}
                                </Badge>
                                <Badge 
                                  variant={suggestion.severity === 'high' ? 'destructive' : 'secondary'}
                                  className="text-xs"
                                >
                                  {suggestion.severity}
                                </Badge>
                              </div>
                              <p className="text-sm text-gray-700">{suggestion.suggestion}</p>
                              {suggestion.action_items && suggestion.action_items.length > 0 && (
                                <ul className="mt-2 space-y-1">
                                  {suggestion.action_items.map((item, idx) => (
                                    <li key={idx} className="text-xs text-gray-600 flex items-start gap-1">
                                      <span className="text-indigo-600">•</span>
                                      <span>{item}</span>
                                    </li>
                                  ))}
                                </ul>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BookEditor;