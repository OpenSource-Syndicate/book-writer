from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import json
import httpx
import asyncio
from emergentintegrations import OpenAI

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# ==================== MODELS ====================

class Note(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    book_id: str
    content: str
    expanded_content: Optional[str] = None
    word_count: int = 0
    target_words: int = 500
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Chapter(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    book_id: str
    title: str
    order: int
    notes: List[str] = []  # List of note IDs
    estimated_pages: int = 0
    status: str = "draft"  # draft, in_progress, completed
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Book(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    book_type: str  # fiction, biography, motivation, technical, etc.
    concept: str
    target_pages: int = 200
    total_words: int = 0
    status: str = "planning"  # planning, writing, completed
    chapters: List[str] = []  # List of chapter IDs
    book_bible: Dict[str, Any] = {}  # Character profiles, world-building, etc.
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserStats(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    total_xp: int = 0
    current_streak: int = 0
    longest_streak: int = 0
    chapters_completed: int = 0
    total_words_written: int = 0
    level: int = 1
    achievements: List[str] = []
    last_activity_date: Optional[str] = None

class ApiSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    use_custom_api: bool = False
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2000

class Suggestion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    book_id: str
    chapter_id: Optional[str] = None
    type: str  # plot_hole, pacing, character_consistency, expansion, tone_drift
    severity: str  # low, medium, high
    suggestion: str
    action_items: List[str] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Request Models
class BookCreate(BaseModel):
    title: str
    book_type: str
    concept: str
    target_pages: int = 200

class ChapterCreate(BaseModel):
    book_id: str
    title: str
    order: int

class NoteCreate(BaseModel):
    book_id: str
    content: str
    target_words: int = 500

class ExpansionRequest(BaseModel):
    note_id: str
    book_id: str
    style: str = "creative"  # professional, creative, conversational, academic, poetic, technical
    target_words: int = 500

class ApiSettingsUpdate(BaseModel):
    use_custom_api: bool = False
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2000

# ==================== HELPER FUNCTIONS ====================

async def get_api_settings():
    """Get API settings from database or return default with Emergent key"""
    settings = await db.api_settings.find_one({}, {"_id": 0})
    if not settings:
        # Return default with Emergent LLM key
        emergent_key = os.environ.get('EMERGENT_LLM_KEY', '')
        default_settings = {
            "use_custom_api": False,
            "api_endpoint": "",
            "api_key": emergent_key,
            "model_name": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 2000
        }
        return default_settings
    return settings

async def stream_ai_completion(prompt: str, settings: dict):
    """Stream AI completion using emergentintegrations or fallback to non-streaming"""
    api_key = settings.get('api_key')
    model_name = settings.get('model_name', 'gpt-4o')
    temperature = settings.get('temperature', 0.7)
    max_tokens = settings.get('max_tokens', 2000)
    use_custom_api = settings.get('use_custom_api', False)
    
    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")
    
    try:
        # Use emergentintegrations for Emergent key
        if not use_custom_api:
            client = OpenAI(api_key=api_key)
            
            # Try streaming first
            try:
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            yield delta.content
            except Exception as stream_error:
                # Fallback to non-streaming
                logging.info(f"Streaming failed, using non-streaming: {stream_error}")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
                if response.choices and len(response.choices) > 0:
                    yield response.choices[0].message.content
        else:
            # Custom API endpoint - use httpx
            api_endpoint = settings.get('api_endpoint', 'https://api.openai.com/v1')
            
            if not api_endpoint.endswith('/chat/completions'):
                if api_endpoint.endswith('/'):
                    api_endpoint = api_endpoint + 'chat/completions'
                else:
                    api_endpoint = api_endpoint + '/chat/completions'
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }
            
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                async with http_client.stream('POST', api_endpoint, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise HTTPException(status_code=response.status_code, detail=f"API Error: {error_text.decode()}")
                    
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            data = line[6:]
                            if data.strip() == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
    except Exception as e:
        logging.error(f"AI completion error: {e}")
        raise HTTPException(status_code=500, detail=f"AI completion failed: {str(e)}")

def generate_expansion_prompt(notes: str, book_type: str, style: str, target_words: int, 
                              book_context: dict, previous_content: str = "") -> str:
    """Generate advanced prompt for note expansion"""
    
    style_guides = {
        "professional": "formal, clear, authoritative tone",
        "creative": "vivid, imaginative, engaging narrative voice",
        "conversational": "friendly, approachable, direct tone",
        "academic": "scholarly, analytical, well-researched tone",
        "poetic": "lyrical, metaphorical, evocative language",
        "technical": "precise, detailed, instructional tone"
    }
    
    book_type_guides = {
        "fiction": "Create compelling narrative with rich character development and vivid scenes",
        "biography": "Present factual events with engaging storytelling and human insight",
        "motivation": "Inspire and empower readers with actionable insights and relatable examples",
        "technical": "Explain complex concepts clearly with practical examples",
        "business": "Provide strategic insights with real-world applications",
        "self-help": "Offer practical guidance with empathy and encouragement"
    }
    
    prompt = f"""You are a professional ghostwriter specializing in {book_type} content.

WRITING CONTEXT:
- Book Type: {book_type}
- Writing Style: {style_guides.get(style, 'engaging and clear')}
- Target Length: approximately {target_words} words
- Genre Approach: {book_type_guides.get(book_type, 'Clear and engaging')}

BOOK BIBLE (Consistency Reference):
{json.dumps(book_context.get('book_bible', {}), indent=2) if book_context.get('book_bible') else 'No prior context established'}

PREVIOUS CONTENT (for continuity):
{previous_content[-1500:] if previous_content else 'This is the opening of the book'}

YOUR TASK:
Expand the following notes into a compelling {book_type} section. Maintain consistency with previous content and match the writing style precisely.

NOTES TO EXPAND:
{notes}

REQUIREMENTS:
1. Target approximately {target_words} words
2. Write in {style} style
3. Maintain narrative continuity with previous content
4. Use vivid, concrete examples (show, don't tell)
5. Break into natural paragraphs with good flow
6. End with a compelling transition or cliffhanger
7. Ensure pacing matches the book's rhythm
8. For fiction: focus on scene-setting, dialogue, and character emotions
9. For non-fiction: use anecdotes, data, and actionable insights

EXPANDED CONTENT:"""
    
    return prompt

async def calculate_xp(action: str, metadata: dict) -> int:
    """Calculate XP for various actions"""
    xp_table = {
        "NOTE_CREATED": 10,
        "EXPANSION_COMPLETED": 50,
        "SUGGESTION_ACCEPTED": 25,
        "CHAPTER_COMPLETED": 200,
        "BOOK_COMPLETED": 1000
    }
    
    base_xp = xp_table.get(action, 0)
    
    # Bonus for word count
    if action == "EXPANSION_COMPLETED" and "word_count" in metadata:
        base_xp += metadata["word_count"] // 100
    
    return base_xp

async def update_user_stats(action: str, metadata: dict):
    """Update user statistics and handle achievements"""
    stats = await db.user_stats.find_one({}, {"_id": 0})
    
    if not stats:
        stats = UserStats().model_dump()
        stats['id'] = str(uuid.uuid4())
    
    # Calculate XP
    xp_earned = await calculate_xp(action, metadata)
    stats['total_xp'] += xp_earned
    
    # Update level (every 1000 XP = 1 level)
    stats['level'] = (stats['total_xp'] // 1000) + 1
    
    # Update counters
    if action == "CHAPTER_COMPLETED":
        stats['chapters_completed'] += 1
    
    if action == "EXPANSION_COMPLETED" and "word_count" in metadata:
        stats['total_words_written'] += metadata['word_count']
    
    # Update streak
    today = datetime.now(timezone.utc).date().isoformat()
    if stats.get('last_activity_date') != today:
        yesterday = (datetime.now(timezone.utc).date().day) - 1
        last_date = stats.get('last_activity_date')
        
        if last_date and (datetime.now(timezone.utc).date().day - 1) == int(last_date.split('-')[-1]):
            stats['current_streak'] += 1
        else:
            stats['current_streak'] = 1
        
        stats['last_activity_date'] = today
        
        if stats['current_streak'] > stats['longest_streak']:
            stats['longest_streak'] = stats['current_streak']
    
    # Check for achievements
    achievements = stats.get('achievements', [])
    
    if stats['total_words_written'] >= 1000 and 'FIRST_DRAFT' not in achievements:
        achievements.append('FIRST_DRAFT')
    
    if stats['chapters_completed'] >= 10 and 'DEDICATED_WRITER' not in achievements:
        achievements.append('DEDICATED_WRITER')
    
    if stats['current_streak'] >= 7 and 'WEEK_STREAK' not in achievements:
        achievements.append('WEEK_STREAK')
    
    if stats['total_words_written'] >= 50000 and 'NOVELIST' not in achievements:
        achievements.append('NOVELIST')
    
    stats['achievements'] = achievements
    
    # Save to database
    await db.user_stats.replace_one({}, stats, upsert=True)
    
    return stats

# ==================== API ROUTES ====================

@api_router.get("/")
async def root():
    return {"message": "Book Creation Platform API"}

# Books
@api_router.post("/books", response_model=Book)
async def create_book(book_data: BookCreate):
    book = Book(**book_data.model_dump())
    doc = book.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    doc['updated_at'] = doc['updated_at'].isoformat()
    await db.books.insert_one(doc)
    return book

@api_router.get("/books", response_model=List[Book])
async def get_books():
    books = await db.books.find({}, {"_id": 0}).to_list(1000)
    for book in books:
        if isinstance(book['created_at'], str):
            book['created_at'] = datetime.fromisoformat(book['created_at'])
        if isinstance(book['updated_at'], str):
            book['updated_at'] = datetime.fromisoformat(book['updated_at'])
    return books

@api_router.get("/books/{book_id}", response_model=Book)
async def get_book(book_id: str):
    book = await db.books.find_one({"id": book_id}, {"_id": 0})
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    if isinstance(book['created_at'], str):
        book['created_at'] = datetime.fromisoformat(book['created_at'])
    if isinstance(book['updated_at'], str):
        book['updated_at'] = datetime.fromisoformat(book['updated_at'])
    return book

@api_router.put("/books/{book_id}", response_model=Book)
async def update_book(book_id: str, updates: dict):
    updates['updated_at'] = datetime.now(timezone.utc).isoformat()
    result = await db.books.update_one({"id": book_id}, {"$set": updates})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Book not found")
    return await get_book(book_id)

# Chapters
@api_router.post("/chapters", response_model=Chapter)
async def create_chapter(chapter_data: ChapterCreate):
    chapter = Chapter(**chapter_data.model_dump())
    doc = chapter.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.chapters.insert_one(doc)
    
    # Add chapter to book
    await db.books.update_one(
        {"id": chapter_data.book_id},
        {"$push": {"chapters": chapter.id}}
    )
    
    return chapter

@api_router.get("/chapters/{book_id}", response_model=List[Chapter])
async def get_chapters(book_id: str):
    chapters = await db.chapters.find({"book_id": book_id}, {"_id": 0}).sort("order", 1).to_list(1000)
    for chapter in chapters:
        if isinstance(chapter['created_at'], str):
            chapter['created_at'] = datetime.fromisoformat(chapter['created_at'])
    return chapters

@api_router.put("/chapters/{chapter_id}/status")
async def update_chapter_status(chapter_id: str, status: str):
    result = await db.chapters.update_one({"id": chapter_id}, {"$set": {"status": status}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    if status == "completed":
        await update_user_stats("CHAPTER_COMPLETED", {})
    
    return {"success": True}

# Notes
@api_router.post("/notes", response_model=Note)
async def create_note(note_data: NoteCreate):
    note = Note(**note_data.model_dump(), word_count=len(note_data.content.split()))
    doc = note.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    doc['updated_at'] = doc['updated_at'].isoformat()
    await db.notes.insert_one(doc)
    
    await update_user_stats("NOTE_CREATED", {})
    
    return note

@api_router.get("/notes/{book_id}", response_model=List[Note])
async def get_notes(book_id: str):
    notes = await db.notes.find({"book_id": book_id}, {"_id": 0}).to_list(1000)
    for note in notes:
        if isinstance(note['created_at'], str):
            note['created_at'] = datetime.fromisoformat(note['created_at'])
        if isinstance(note['updated_at'], str):
            note['updated_at'] = datetime.fromisoformat(note['updated_at'])
    return notes

# Note Expansion with Streaming
@api_router.post("/expand")
async def expand_note(request: ExpansionRequest):
    # Get note
    note = await db.notes.find_one({"id": request.note_id}, {"_id": 0})
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    # Get book context
    book = await db.books.find_one({"id": request.book_id}, {"_id": 0})
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    # Get previous notes for context
    previous_notes = await db.notes.find(
        {"book_id": request.book_id, "expanded_content": {"$ne": None}},
        {"_id": 0}
    ).limit(3).to_list(3)
    
    previous_content = "\n\n".join([n.get('expanded_content', '') for n in previous_notes if n.get('expanded_content')])
    
    # Generate prompt
    prompt = generate_expansion_prompt(
        note['content'],
        book['book_type'],
        request.style,
        request.target_words,
        book,
        previous_content
    )
    
    # Get API settings
    settings = await get_api_settings()
    
    # Stream response
    async def generate():
        full_text = ""
        try:
            async for chunk in stream_ai_completion(prompt, settings):
                full_text += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            # Save expanded content
            word_count = len(full_text.split())
            await db.notes.update_one(
                {"id": request.note_id},
                {"$set": {
                    "expanded_content": full_text,
                    "word_count": word_count,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }}
            )
            
            # Update book word count
            await db.books.update_one(
                {"id": request.book_id},
                {"$inc": {"total_words": word_count}}
            )
            
            # Update stats
            await update_user_stats("EXPANSION_COMPLETED", {"word_count": word_count})
            
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# API Settings
@api_router.get("/settings/api")
async def get_api_config():
    settings = await get_api_settings()
    # Don't expose full API key
    if settings.get('api_key'):
        settings['api_key'] = settings['api_key'][:8] + '...' if len(settings['api_key']) > 8 else '***'
    return settings

@api_router.post("/settings/api")
async def update_api_settings(settings: ApiSettingsUpdate):
    settings_dict = settings.model_dump()
    settings_dict['id'] = str(uuid.uuid4())
    
    # If using Emergent key, load it
    if not settings.use_custom_api:
        settings_dict['api_key'] = os.environ.get('EMERGENT_LLM_KEY', '')
        settings_dict['api_endpoint'] = ""
    
    await db.api_settings.delete_many({})
    await db.api_settings.insert_one(settings_dict)
    
    return {"success": True}

# User Stats
@api_router.get("/stats", response_model=UserStats)
async def get_stats():
    stats = await db.user_stats.find_one({}, {"_id": 0})
    if not stats:
        stats = UserStats().model_dump()
    return stats

# Suggestions
@api_router.get("/suggestions/{book_id}", response_model=List[Suggestion])
async def get_suggestions(book_id: str):
    suggestions = await db.suggestions.find({"book_id": book_id}, {"_id": 0}).to_list(100)
    for suggestion in suggestions:
        if isinstance(suggestion['created_at'], str):
            suggestion['created_at'] = datetime.fromisoformat(suggestion['created_at'])
    return suggestions

@api_router.post("/suggestions/generate/{book_id}")
async def generate_suggestions(book_id: str):
    """Generate AI suggestions for a book"""
    book = await db.books.find_one({"id": book_id}, {"_id": 0})
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    # Get all notes with content
    notes = await db.notes.find(
        {"book_id": book_id, "expanded_content": {"$ne": None}},
        {"_id": 0}
    ).to_list(1000)
    
    if not notes:
        return {"suggestions": []}
    
    # Combine content
    all_content = "\n\n".join([n.get('expanded_content', '') for n in notes if n.get('expanded_content')])
    
    prompt = f"""Analyze this {book['book_type']} book content and provide suggestions:

{all_content[:3000]}

Identify:
1. Areas that could be expanded
2. Pacing issues (too fast/slow)
3. Tone consistency
4. Next chapter direction suggestions
5. Character/concept development opportunities

Return ONLY a JSON array with format:
[{{"type": "expansion|pacing|tone_drift|next_chapter", "severity": "low|medium|high", "suggestion": "description", "action_items": ["item1", "item2"]}}]"""
    
    settings = await get_api_settings()
    
    try:
        full_response = ""
        async for chunk in stream_ai_completion(prompt, settings):
            full_response += chunk
        
        # Parse JSON response
        suggestions_data = json.loads(full_response.strip())
        
        # Save suggestions
        for sug_data in suggestions_data:
            suggestion = Suggestion(
                book_id=book_id,
                type=sug_data.get('type', 'general'),
                severity=sug_data.get('severity', 'medium'),
                suggestion=sug_data.get('suggestion', ''),
                action_items=sug_data.get('action_items', [])
            )
            doc = suggestion.model_dump()
            doc['created_at'] = doc['created_at'].isoformat()
            await db.suggestions.insert_one(doc)
        
        return {"suggestions": suggestions_data}
    except Exception as e:
        logging.error(f"Error generating suggestions: {e}")
        return {"error": str(e)}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()