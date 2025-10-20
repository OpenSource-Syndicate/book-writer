import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { API } from "../App";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BookOpen, Plus, Trophy, Flame, Target, Award, Settings, Zap } from "lucide-react";
import { toast } from "sonner";
import SettingsModal from "../components/SettingsModal";

const Dashboard = () => {
  const navigate = useNavigate();
  const [books, setBooks] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showNewBook, setShowNewBook] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  
  const [newBook, setNewBook] = useState({
    title: "",
    book_type: "fiction",
    concept: "",
    target_pages: 200
  });

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [booksRes, statsRes] = await Promise.all([
        axios.get(`${API}/books`),
        axios.get(`${API}/stats`)
      ]);
      setBooks(booksRes.data);
      setStats(statsRes.data);
    } catch (error) {
      console.error("Error fetching data:", error);
      toast.error("Failed to load data");
    } finally {
      setLoading(false);
    }
  };

  const createBook = async () => {
    if (!newBook.title || !newBook.concept) {
      toast.error("Please fill in all required fields");
      return;
    }

    try {
      const response = await axios.post(`${API}/books`, newBook);
      toast.success("Book created successfully!");
      setShowNewBook(false);
      setNewBook({ title: "", book_type: "fiction", concept: "", target_pages: 200 });
      navigate(`/book/${response.data.id}`);
    } catch (error) {
      console.error("Error creating book:", error);
      toast.error("Failed to create book");
    }
  };

  const achievements = [
    { id: "FIRST_DRAFT", name: "First Draft", description: "Write your first 1,000 words", icon: "📝", unlocked: stats?.achievements?.includes("FIRST_DRAFT") },
    { id: "DEDICATED_WRITER", name: "Dedicated Writer", description: "Complete 10 chapters", icon: "✍️", unlocked: stats?.achievements?.includes("DEDICATED_WRITER") },
    { id: "WEEK_STREAK", name: "Week Warrior", description: "7-day writing streak", icon: "🔥", unlocked: stats?.achievements?.includes("WEEK_STREAK") },
    { id: "NOVELIST", name: "Novelist", description: "Write 50,000 words", icon: "📚", unlocked: stats?.achievements?.includes("NOVELIST") }
  ];

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
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <BookOpen className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">NarrativeForge</h1>
                <p className="text-sm text-gray-500">AI-Powered Book Creation</p>
              </div>
            </div>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => setShowSettings(true)}
              data-testid="settings-button"
            >
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </Button>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card className="bg-white/60 backdrop-blur-sm border-blue-200" data-testid="xp-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600 flex items-center gap-2">
                <Zap className="w-4 h-4 text-yellow-500" />
                Total XP
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-indigo-600">{stats?.total_xp || 0}</div>
              <p className="text-xs text-gray-500 mt-1">Level {stats?.level || 1}</p>
            </CardContent>
          </Card>

          <Card className="bg-white/60 backdrop-blur-sm border-orange-200" data-testid="streak-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600 flex items-center gap-2">
                <Flame className="w-4 h-4 text-orange-500" />
                Current Streak
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-orange-600">{stats?.current_streak || 0}</div>
              <p className="text-xs text-gray-500 mt-1">Longest: {stats?.longest_streak || 0} days</p>
            </CardContent>
          </Card>

          <Card className="bg-white/60 backdrop-blur-sm border-green-200" data-testid="chapters-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600 flex items-center gap-2">
                <Target className="w-4 h-4 text-green-500" />
                Chapters Done
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-600">{stats?.chapters_completed || 0}</div>
              <p className="text-xs text-gray-500 mt-1">Keep writing!</p>
            </CardContent>
          </Card>

          <Card className="bg-white/60 backdrop-blur-sm border-purple-200" data-testid="words-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600 flex items-center gap-2">
                <Trophy className="w-4 h-4 text-purple-500" />
                Total Words
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-purple-600">{stats?.total_words_written?.toLocaleString() || 0}</div>
              <p className="text-xs text-gray-500 mt-1">Written so far</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs defaultValue="books" className="space-y-6">
          <TabsList className="bg-white/80 backdrop-blur-sm" data-testid="tabs-list">
            <TabsTrigger value="books" data-testid="books-tab">My Books</TabsTrigger>
            <TabsTrigger value="achievements" data-testid="achievements-tab">Achievements</TabsTrigger>
          </TabsList>

          <TabsContent value="books" className="space-y-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-900">Your Books</h2>
              <Dialog open={showNewBook} onOpenChange={setShowNewBook}>
                <DialogTrigger asChild>
                  <Button className="bg-indigo-600 hover:bg-indigo-700" data-testid="new-book-button">
                    <Plus className="w-4 h-4 mr-2" />
                    New Book
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-[500px]" data-testid="new-book-dialog">
                  <DialogHeader>
                    <DialogTitle>Create New Book</DialogTitle>
                    <DialogDescription>
                      Start your writing journey with AI assistance
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 py-4">
                    <div className="space-y-2">
                      <Label htmlFor="title">Book Title *</Label>
                      <Input
                        id="title"
                        data-testid="book-title-input"
                        placeholder="Enter your book title"
                        value={newBook.title}
                        onChange={(e) => setNewBook({ ...newBook, title: e.target.value })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="type">Book Type</Label>
                      <Select
                        value={newBook.book_type}
                        onValueChange={(value) => setNewBook({ ...newBook, book_type: value })}
                      >
                        <SelectTrigger data-testid="book-type-select">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="fiction">Fiction</SelectItem>
                          <SelectItem value="biography">Biography</SelectItem>
                          <SelectItem value="motivation">Motivation</SelectItem>
                          <SelectItem value="technical">Technical</SelectItem>
                          <SelectItem value="business">Business</SelectItem>
                          <SelectItem value="self-help">Self-Help</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="concept">Book Concept *</Label>
                      <Textarea
                        id="concept"
                        data-testid="book-concept-input"
                        placeholder="Describe your book idea, themes, and key concepts..."
                        value={newBook.concept}
                        onChange={(e) => setNewBook({ ...newBook, concept: e.target.value })}
                        rows={4}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="pages">Target Pages</Label>
                      <Input
                        id="pages"
                        data-testid="target-pages-input"
                        type="number"
                        value={newBook.target_pages}
                        onChange={(e) => setNewBook({ ...newBook, target_pages: parseInt(e.target.value) })}
                      />
                    </div>
                  </div>
                  <div className="flex justify-end gap-3">
                    <Button variant="outline" onClick={() => setShowNewBook(false)} data-testid="cancel-book-button">
                      Cancel
                    </Button>
                    <Button onClick={createBook} className="bg-indigo-600 hover:bg-indigo-700" data-testid="create-book-button">
                      Create Book
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
            </div>

            {books.length === 0 ? (
              <Card className="bg-white/60 backdrop-blur-sm" data-testid="empty-books-state">
                <CardContent className="flex flex-col items-center justify-center py-16">
                  <BookOpen className="w-16 h-16 text-gray-300 mb-4" />
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">No books yet</h3>
                  <p className="text-gray-500 text-center mb-6">Start your writing journey by creating your first book</p>
                  <Button onClick={() => setShowNewBook(true)} className="bg-indigo-600 hover:bg-indigo-700" data-testid="empty-new-book-button">
                    <Plus className="w-4 h-4 mr-2" />
                    Create Your First Book
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {books.map((book) => (
                  <Card
                    key={book.id}
                    className="card-hover cursor-pointer bg-white/70 backdrop-blur-sm"
                    onClick={() => navigate(`/book/${book.id}`)}
                    data-testid={`book-card-${book.id}`}
                  >
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <CardTitle className="text-lg line-clamp-1">{book.title}</CardTitle>
                          <CardDescription className="mt-1">
                            <Badge variant="outline" className="text-xs">
                              {book.book_type}
                            </Badge>
                          </CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-gray-600 line-clamp-2 mb-4">{book.concept}</p>
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs text-gray-500">
                          <span>Progress</span>
                          <span>{book.total_words?.toLocaleString() || 0} words</span>
                        </div>
                        <Progress value={(book.total_words / (book.target_pages * 250)) * 100} className="h-2" />
                      </div>
                      <div className="flex items-center justify-between mt-4 text-xs text-gray-500">
                        <span>{book.chapters?.length || 0} chapters</span>
                        <Badge variant={book.status === "completed" ? "success" : "secondary"}>
                          {book.status}
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="achievements">
            <Card className="bg-white/60 backdrop-blur-sm" data-testid="achievements-container">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Award className="w-5 h-5 text-yellow-500" />
                  Your Achievements
                </CardTitle>
                <CardDescription>
                  Unlock badges by reaching milestones in your writing journey
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {achievements.map((achievement) => (
                    <div
                      key={achievement.id}
                      className={`p-4 rounded-lg border-2 ${
                        achievement.unlocked
                          ? "bg-gradient-to-br from-yellow-50 to-orange-50 border-yellow-300"
                          : "bg-gray-50 border-gray-200 opacity-50"
                      }`}
                      data-testid={`achievement-${achievement.id}`}
                    >
                      <div className="flex items-start gap-3">
                        <div className="text-3xl">{achievement.icon}</div>
                        <div className="flex-1">
                          <h4 className="font-semibold text-gray-900">{achievement.name}</h4>
                          <p className="text-sm text-gray-600 mt-1">{achievement.description}</p>
                          {achievement.unlocked && (
                            <Badge className="mt-2 bg-yellow-500">Unlocked!</Badge>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Settings Modal */}
      <SettingsModal open={showSettings} onOpenChange={setShowSettings} />
    </div>
  );
};

export default Dashboard;