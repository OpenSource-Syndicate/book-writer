"""
Book Writer System - NLP-based Content Classifier
Reliable classification of notes using traditional NLP algorithms, Sentence Transformers, and hybrid search
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import difflib  # For fuzzy string matching

# Import Sentence Transformers for enhanced semantic matching
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

# Import FAISS for optimized similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")

# Import Weaviate for vector database integration
try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    print("Warning: Weaviate not available. Install with: pip install weaviate-client")

# Import NLTK for advanced text preprocessing
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Install with: pip install nltk")


class NLPContentClassifier:
    """NLP-based classifier for content using TF-IDF, Sentence Transformers, and similarity metrics."""
    
    def __init__(self):
        """Initialize the NLP classifier."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            stop_words='english',
            max_features=10000,
            lowercase=True
        )
        self.chapters_vector_cache = {}
        self.subtopics_vector_cache = {}
        self.content_vector_cache = {}  # Cache for content vectors
        self.cache_stats = {"hits": 0, "misses": 0}  # Cache hit/miss statistics
        
        # Enhanced classification cache for improved performance
        self.classification_cache = {}  # Cache for complete classification results
        self.cache_max_size = 1000  # Maximum number of entries in cache
        self.cache_access_order = []  # Track access order for LRU eviction
        
        # Topic extraction cache
        self.topic_extraction_cache = {}
        
        # Define common synonyms and misspellings for better matching
        self.synonyms = {
            "ai": ["artificial intelligence", "machine intelligence"],
            "artificial intelligence": ["ai", "machine intelligence"],
            "intellegence": ["intelligence"],  # Common misspelling
            "intellegent": ["intelligent"],   # Common misspelling
            "how": ["how does", "how do", "ways"],
            "changed": ["transformed", "evolved", "shifted", "impacted"],
            "world": ["society", "humanity", "civilization"],
            "future": ["upcoming", "next", "forthcoming"],
            "trends": ["developments", "advancements", "progress"],
        }
        
        # Initialize Sentence Transformer model for semantic matching if available
        self.semantic_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use a lightweight model for good performance
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Sentence Transformer model initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize Sentence Transformer model: {e}")
        
        # FAISS index for optimized similarity search
        self.faiss_index = None
        self.index_dimension = 384  # Default dimension for all-MiniLM-L6-v2
        if FAISS_AVAILABLE and self.semantic_model:
            try:
                # Initialize FAISS index for L2 similarity search
                self.faiss_index = faiss.IndexFlatL2(self.index_dimension)
                print("FAISS index initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize FAISS index: {e}")
        
        # FAISS vector cache for optimized similarity search
        self.faiss_vector_cache = {}  # Cache for FAISS vectors
        self.faiss_id_mapping = {}    # Mapping from FAISS IDs to content IDs
        self.next_faiss_id = 0        # Counter for FAISS IDs
        
        # NLTK components for advanced preprocessing
        self.lemmatizer = None
        self.stop_words = set()
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data if not already present
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                
                try:
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    nltk.download('wordnet', quiet=True)
                
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                
                # Initialize lemmatizer and stop words
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
                print("NLTK components initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize NLTK components: {e}")
                # Don't modify the global NLTK_AVAILABLE; just set instance variables to None/empty
                self.lemmatizer = None
                self.stop_words = set()
        else:
            # Ensure that NLTK variables are set to None/empty if not available
            self.lemmatizer = None
            self.stop_words = set()
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Use advanced preprocessing if available
        if self.lemmatizer and self.stop_words:
            return self._preprocess_text_advanced(text)
        
        # Fallback to basic preprocessing
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        return text
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords from text.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        # Simple keyword extraction based on word frequency
        words = text.lower().split()
        word_freq = Counter(words)
        return [word for word, freq in word_freq.most_common(top_n)]
    
    def calculate_similarity(self, content: str, outline_texts: List[str]) -> List[float]:
        """Calculate similarity between content and outline texts.
        
        Args:
            content: Content to classify
            outline_texts: List of outline section texts to compare against
            
        Returns:
            List of similarity scores
        """
        if not content or not outline_texts:
            return [0.0] * len(outline_texts)
        
        # Preprocess content
        processed_content = self.preprocess_text(content)
        
        # Check cache for content vector
        content_hash = hash(processed_content)
        if content_hash in self.content_vector_cache:
            content_vector = self.content_vector_cache[content_hash]
            self.cache_stats["hits"] += 1
        else:
            # Vectorize content and outline texts together
            all_texts = [processed_content] + [self.preprocess_text(t) for t in outline_texts]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            content_vector = tfidf_matrix[0]
            # Cache the content vector
            self.content_vector_cache[content_hash] = content_vector
            self.cache_stats["misses"] += 1
        
        # Vectorize outline texts (or use cached vectors)
        processed_outline_texts = [self.preprocess_text(t) for t in outline_texts]
        tfidf_matrix = self.vectorizer.fit_transform([processed_content] + processed_outline_texts)
        
        # Calculate cosine similarity between content and each outline section
        similarities = []
        for i in range(1, len(processed_outline_texts) + 1):
            outline_vector = tfidf_matrix[i]
            similarity = cosine_similarity(tfidf_matrix[0], outline_vector)[0][0]
            similarities.append(similarity)
        
        return similarities
    
    def classify_content_batch(self, contents: List[str], outline_data: Dict) -> List[Dict]:
        """Classify multiple contents in batch for better performance.
        
        Args:
            contents: List of contents to classify
            outline_data: The book outline data
            
        Returns:
            List of classification results
        """
        print(f"Batch classifying {len(contents)} contents...")
        
        results = []
        for i, content in enumerate(contents):
            print(f"Processing content {i+1}/{len(contents)}...")
            result = self.classify_content(content, outline_data)
            results.append(result)
        
        print(f"Batch classification completed. Processed {len(results)} contents.")
        return results
    
    def _get_classification_cache_key(self, content: str, outline_data: Dict) -> str:
        """
        Generate a cache key for a classification request.
        
        Args:
            content: Content to classify
            outline_data: Outline data used for classification
            
        Returns:
            A unique cache key
        """
        import hashlib
        # Create a unique key based on content and outline structure
        outline_structure = str(sorted([part.get('id', '') for part in outline_data.get('parts', [])]))
        cache_input = f"{content[:100]}_{outline_structure[:200]}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _get_topic_cache_key(self, content: str) -> str:
        """
        Generate a cache key for topic extraction.
        
        Args:
            content: Content to extract topics from
            
        Returns:
            A unique cache key
        """
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()

    def _check_classification_cache(self, key: str) -> Optional[Dict]:
        """
        Check if a classification result exists in the cache.
        
        Args:
            key: The cache key
            
        Returns:
            Cached classification result if found, else None
        """
        if key in self.classification_cache:
            # Update access order for LRU
            if key in self.cache_access_order:
                self.cache_access_order.remove(key)
            self.cache_access_order.append(key)
            
            self.cache_stats["hits"] += 1
            return self.classification_cache[key]
        
        self.cache_stats["misses"] += 1
        return None

    def _store_in_classification_cache(self, key: str, result: Dict):
        """
        Store a classification result in the cache.
        
        Args:
            key: The cache key
            result: The classification result to store
        """
        # Check if cache is at max size
        if len(self.classification_cache) >= self.cache_max_size:
            # Remove least recently used item
            if self.cache_access_order:
                lru_key = self.cache_access_order.pop(0)
                del self.classification_cache[lru_key]
        
        self.classification_cache[key] = result
        self.cache_access_order.append(key)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache hit/miss statistics
        """
        all_cache_stats = self.cache_stats.copy()
        all_cache_stats["total_classification_cache_size"] = len(self.classification_cache)
        all_cache_stats["total_topic_cache_size"] = len(self.topic_extraction_cache)
        all_cache_stats["max_classification_cache_size"] = self.cache_max_size
        return all_cache_stats
    
    def clear_cache(self):
        """Clear all caches."""
        self.chapters_vector_cache.clear()
        self.subtopics_vector_cache.clear()
        self.content_vector_cache.clear()
        self.classification_cache.clear()
        self.topic_extraction_cache.clear()
        self.cache_access_order.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def _apply_domain_boost(self, content: str, outline_section: Dict, base_score: float) -> float:
        """Apply domain-specific boosting for better matching.
        
        Args:
            content: Content to classify
            outline_section: Outline section to match against
            base_score: Base similarity score
            
        Returns:
            Score with domain boosting applied
        """
        # Domain-specific terms that should boost matching
        domain_terms = [
            "ai", "artificial intelligence", "machine learning", "neural network",
            "algorithm", "data", "model", "training", "prediction"
        ]
        
        content_lower = content.lower()
        section_text = f"{outline_section.get('title', '')} {outline_section.get('description', '')}".lower()
        
        # Count domain term matches
        domain_matches = sum(1 for term in domain_terms if term in content_lower or term in section_text)
        
        if domain_matches > 0:
            # Boost score based on domain term matches (up to 20% boost)
            boost = min(domain_matches * 0.05, 0.2)
            return min(base_score + boost, 1.0)  # Cap at 1.0
        else:
            return base_score
    
    def classify_content(self, content: str, outline_data: Dict) -> Dict:
        """Classify content into appropriate chapters and subtopics using NLP.
        
        Args:
            content: The content to classify
            outline_data: The book outline data
            
        Returns:
            A dictionary with classification results
        """
        # Check if result is already in cache
        cache_key = self._get_classification_cache_key(content, outline_data)
        cached_result = self._check_classification_cache(cache_key)
        if cached_result is not None:
            print(f"Cache hit for content: {content[:50]}...")
            return cached_result
        
        try:
            print(f"NLP Classifying content: {content[:100]}...")
            
            if not content or not outline_data:
                print("Empty content or outline data, returning empty classification")
                result = {
                    "chapter": None,
                    "subtopic": None,
                    "chapter_score": 0.0,
                    "subtopic_score": 0.0
                }
                self._store_in_classification_cache(cache_key, result)
                return result
            
            # Extract all chapters and subtopics with their texts
            chapters = []
            chapter_texts = []
            subtopics = []
            subtopic_texts = []
            
            for part in outline_data.get("parts", []):
                for chapter in part.get("chapters", []):
                    chapter_info = {
                        "id": chapter["id"],
                        "title": chapter["title"],
                        "description": chapter.get("description", ""),
                        "part_id": part["id"]
                    }
                    
                    chapters.append(chapter_info)
                    # Combine title and description for matching
                    chapter_text = f"{chapter['title']} {chapter.get('description', '')}"
                    chapter_texts.append(chapter_text)
                    
                    for subtopic in chapter.get("subtopics", []):
                        subtopic_info = {
                            "id": subtopic["id"],
                            "title": subtopic["title"],
                            "description": subtopic.get("description", ""),
                            "chapter_id": chapter["id"]
                        }
                        
                        subtopics.append(subtopic_info)
                        # Combine title and description for matching
                        subtopic_text = f"{subtopic['title']} {subtopic.get('description', '')}"
                        subtopic_texts.append(subtopic_text)
            
            if not chapters or not subtopics:
                print("No chapters or subtopics found in outline data")
                return {
                    "chapter": None,
                    "subtopic": None,
                    "chapter_score": 0.0,
                    "subtopic_score": 0.0
                }
            
            # Calculate chapter scores using hybrid approach
            print("Calculating chapter scores...")
            chapter_scores = []
            chapter_details = []
            
            for i, chapter in enumerate(chapters):
                chapter_text = chapter_texts[i]
                score = self._calculate_hybrid_score(content, chapter_text, chapter["title"])
                chapter_scores.append(score)
                chapter_details.append({
                    "title": chapter["title"],
                    "score": score,
                    "text_preview": chapter_text[:50] + "..." if len(chapter_text) > 50 else chapter_text
                })
            
            # Find best matching chapter
            if chapter_scores:
                best_chapter_idx = np.argmax(chapter_scores)
                best_chapter_score = chapter_scores[best_chapter_idx]
                best_chapter = chapters[best_chapter_idx]
                
                # Log top 3 chapter matches for debugging
                sorted_chapters = sorted(enumerate(chapter_scores), key=lambda x: x[1], reverse=True)
                print("Top 3 chapter matches:")
                for i, (idx, score) in enumerate(sorted_chapters[:3]):
                    chapter_title = chapters[idx]["title"]
                    print(f"  {i+1}. {chapter_title}: {score:.4f}")
            else:
                best_chapter_score = 0.0
                best_chapter = None
            
            print(f"Best chapter match: {best_chapter['title'] if best_chapter else 'None'} (score: {best_chapter_score:.4f})")
            
            # Calculate comprehensive scores for subtopics (only those in the best matching chapter)
            best_subtopic = None
            best_subtopic_score = 0.0
            
            if best_chapter:
                # Filter subtopics to only those in the best matching chapter
                chapter_subtopics = []
                chapter_subtopic_texts = []
                chapter_subtopic_titles = []
                
                for i, subtopic in enumerate(subtopics):
                    if subtopic["chapter_id"] == best_chapter["id"]:
                        chapter_subtopics.append(subtopic)
                        chapter_subtopic_texts.append(subtopic_texts[i])
                        chapter_subtopic_titles.append(subtopic["title"])
                
                if chapter_subtopics:
                    print("Calculating subtopic scores...")
                    subtopic_scores = []
                    subtopic_details = []
                    
                    for i, subtopic_text in enumerate(chapter_subtopic_texts):
                        subtopic_title = chapter_subtopic_titles[i]
                        score = self._calculate_hybrid_score(content, subtopic_text, subtopic_title)
                        subtopic_scores.append(score)
                        subtopic_details.append({
                            "title": subtopic_title,
                            "score": score,
                            "text_preview": subtopic_text[:50] + "..." if len(subtopic_text) > 50 else subtopic_text
                        })
                    
                    # Find best matching subtopic
                    if subtopic_scores:
                        best_subtopic_idx = np.argmax(subtopic_scores)
                        best_subtopic_score = subtopic_scores[best_subtopic_idx]
                        best_subtopic = chapter_subtopics[best_subtopic_idx]
                        
                        # Log top 3 subtopic matches for debugging
                        sorted_subtopics = sorted(enumerate(subtopic_scores), key=lambda x: x[1], reverse=True)
                        print("Top 3 subtopic matches:")
                        for i, (idx, score) in enumerate(sorted_subtopics[:3]):
                            subtopic_title = chapter_subtopics[idx]["title"]
                            print(f"  {i+1}. {subtopic_title}: {score:.4f}")
                        
                        print(f"Best subtopic match: {best_subtopic['title'] if best_subtopic else 'None'} (score: {best_subtopic_score:.4f})")
            
            # Normalize scores to be more interpretable
            normalized_chapter_score = self._normalize_confidence_score(float(best_chapter_score) if best_chapter_score else 0.0)
            normalized_subtopic_score = self._normalize_confidence_score(float(best_subtopic_score) if best_subtopic_score else 0.0)
            
            # Apply more permissive confidence thresholds
            chapter_threshold = 0.05  # Lowered from 0.1
            subtopic_threshold = 0.05  # Lowered from 0.1
            
            # Log detailed scoring information for debugging
            print(f"Scoring details - Chapter: {normalized_chapter_score:.4f} (threshold: {chapter_threshold}), "
                  f"Subtopic: {normalized_subtopic_score:.4f} (threshold: {subtopic_threshold})")
            
            final_chapter = best_chapter if normalized_chapter_score >= chapter_threshold else None
            final_subtopic = best_subtopic if normalized_subtopic_score >= subtopic_threshold else None
            
            result = {
                "chapter": final_chapter,
                "subtopic": final_subtopic,
                "chapter_score": normalized_chapter_score,
                "subtopic_score": normalized_subtopic_score
            }
            
            print(f"Final NLP classification result: {result}")
            self._store_in_classification_cache(cache_key, result)
            return result
            
        except Exception as e:
            error_msg = f"Critical error in NLP content classification: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            # Return default classification on error
            error_result = {
                "chapter": None,
                "subtopic": None,
                "chapter_score": 0.0,
                "subtopic_score": 0.0,
                "error": error_msg
            }
            self._store_in_classification_cache(cache_key, error_result)
            return error_result
    
    def _normalize_confidence_score(self, raw_score: float) -> float:
        """Normalize raw similarity score to a more interpretable confidence score.
        
        Args:
            raw_score: Raw similarity score from cosine similarity
            
        Returns:
            Normalized confidence score between 0.0 and 1.0
        """
        # Cosine similarity ranges from -1 to 1, but typically 0 to 1 for TF-IDF
        # We'll map this to a more interpretable range and apply domain knowledge
        
        # Clamp to 0-1 range
        clamped_score = max(0.0, min(1.0, raw_score))
        
        # Apply less aggressive transformation to preserve score differentiation
        # Using square root makes the curve less steep, preserving more of the original score differences
        # This prevents high scores from being overly diminished
        if clamped_score > 0:
            transformed_score = min(clamped_score ** 0.7, 1.0)  # Less aggressive than square
        else:
            transformed_score = 0.0
        
        return transformed_score
    
    def _calculate_comprehensive_score(self, content: str, section_text: str, section_title: str) -> float:
        """Calculate a comprehensive matching score combining multiple factors.
        
        Args:
            content: Content to classify
            section_text: Combined text of section title and description
            section_title: Title of the section
            
        Returns:
            Comprehensive matching score
        """
        # Expand content with synonyms for better matching
        expanded_content = self._expand_with_synonyms(content)
        
        # Calculate TF-IDF similarity
        tfidf_score = self._calculate_tfidf_similarity(expanded_content, section_text)
        
        # Calculate keyword matching score
        keyword_score = self._calculate_keyword_match_score(expanded_content, section_text)
        
        # Calculate title matching score (more important)
        title_score = self._calculate_title_match_score(expanded_content, section_title)
        
        # Calculate fuzzy matching score for handling typos and close matches
        fuzzy_score = self._calculate_fuzzy_match_score(content, section_title)
        
        # Calculate semantic similarity using Sentence Transformers if available
        semantic_score = self._calculate_semantic_similarity(content, section_text)
        
        # Weighted combination with added semantic similarity
        comprehensive_score = (
            0.3 * tfidf_score +      # 30% TF-IDF similarity
            0.15 * keyword_score +   # 15% keyword matching
            0.15 * title_score +    # 15% title matching
            0.15 * fuzzy_score +    # 15% fuzzy matching (for typo handling)
            0.25 * semantic_score   # 25% semantic similarity (added weight)
        )
        
        return comprehensive_score
    
    def _calculate_hybrid_score(self, content: str, section_text: str, section_title: str) -> float:
        """Calculate hybrid similarity score combining lexical and semantic approaches.
        
        Args:
            content: Content to classify
            section_text: Combined text of section title and description
            section_title: Title of the section
            
        Returns:
            Hybrid similarity score between 0.0 and 1.0
        """
        # Calculate lexical score using existing comprehensive scoring
        lexical_score = self._calculate_comprehensive_score(content, section_text, section_title)
        
        # Calculate semantic score using Sentence Transformers if available
        semantic_score = self._calculate_semantic_similarity(content, section_text)
        
        # Combine lexical and semantic scores with adaptive weighting
        if semantic_score > 0:  # Semantic model is available and working
            # Use a balanced combination
            hybrid_score = (0.6 * lexical_score) + (0.4 * semantic_score)
        else:
            # Fall back to purely lexical scoring
            hybrid_score = lexical_score
        
        return float(hybrid_score)
    
    def _calculate_semantic_similarity(self, content: str, section_text: str) -> float:
        """Calculate semantic similarity using Sentence Transformers.
        
        Args:
            content: Content to classify
            section_text: Text to compare against
            
        Returns:
            Semantic similarity score between 0.0 and 1.0
        """
        # Return 0.0 if semantic model is not available
        if not self.semantic_model:
            return 0.0
        
        try:
            # Encode both texts to get embeddings
            embeddings = self.semantic_model.encode([content, section_text])
            
            # Calculate cosine similarity between embeddings
            # Normalize embeddings to unit vectors for cosine similarity
            norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Calculate cosine similarity (dot product of normalized vectors)
            similarity = np.dot(norm_embeddings[0], norm_embeddings[1])
            
            # Convert to 0-1 range (cosine similarity ranges from -1 to 1)
            normalized_similarity = (similarity + 1) / 2
            
            return float(normalized_similarity)
        except Exception as e:
            print(f"Warning: Failed to calculate semantic similarity: {e}")
            return 0.0
    
    def _calculate_semantic_similarity_faiss(self, content: str, section_texts: List[str]) -> List[float]:
        """Calculate semantic similarities using FAISS for optimized search.
        
        Args:
            content: Content to classify
            section_texts: List of texts to compare against
            
        Returns:
            List of semantic similarity scores
        """
        # Return empty list if FAISS is not available or semantic model is not available
        if not self.faiss_index or not self.semantic_model:
            return [0.0] * len(section_texts)
        
        try:
            # Encode content and section texts to get embeddings
            content_embedding = self.semantic_model.encode([content])[0]
            section_embeddings = self.semantic_model.encode(section_texts)
            
            # Normalize embeddings for cosine similarity using FAISS
            content_norm = np.linalg.norm(content_embedding)
            if content_norm > 0:
                content_embedding = content_embedding / content_norm
            
            section_norms = np.linalg.norm(section_embeddings, axis=1, keepdims=True)
            section_norms[section_norms == 0] = 1  # Avoid division by zero
            normalized_section_embeddings = section_embeddings / section_norms
            
            # Add section embeddings to FAISS index if not already cached
            faiss_ids = []
            for i, section_embedding in enumerate(normalized_section_embeddings):
                section_hash = hash(section_texts[i])
                if section_hash not in self.faiss_vector_cache:
                    # Add to FAISS index
                    self.faiss_index.add(section_embedding.reshape(1, -1))
                    faiss_id = self.next_faiss_id
                    self.faiss_vector_cache[section_hash] = faiss_id
                    self.faiss_id_mapping[faiss_id] = section_texts[i]
                    self.next_faiss_id += 1
                    faiss_ids.append(faiss_id)
                else:
                    faiss_ids.append(self.faiss_vector_cache[section_hash])
            
            # Search for similar vectors using FAISS
            content_embedding = content_embedding.reshape(1, -1)
            k = min(len(section_texts), 10)  # Search for top 10 matches or fewer if less sections
            distances, indices = self.faiss_index.search(content_embedding, k)
            
            # Convert FAISS distances to similarity scores
            # FAISS returns squared L2 distances, convert to cosine similarity
            similarities = []
            for i in range(len(section_texts)):
                # Find if this section was in the top k results
                similarity = 0.0
                for j, idx in enumerate(indices[0]):
                    if idx < len(self.faiss_id_mapping) and self.faiss_id_mapping[idx] == section_texts[i]:
                        # Convert L2 distance to cosine similarity
                        # similarity = 1 - (distance^2 / 2) for normalized vectors
                        similarity = max(0, 1 - (distances[0][j] / 2))
                        break
                similarities.append(similarity)
            
            return similarities
        except Exception as e:
            print(f"Warning: Failed to calculate FAISS semantic similarities: {e}")
            # Fall back to regular semantic similarity calculation
            regular_similarities = []
            for section_text in section_texts:
                similarity = self._calculate_semantic_similarity(content, section_text)
                regular_similarities.append(similarity)
            return regular_similarities
    
    def _calculate_tfidf_similarity(self, content: str, section_text: str) -> float:
        """Calculate TF-IDF based similarity.
        
        Args:
            content: Content to classify
            section_text: Text to compare against
            
        Returns:
            Similarity score
        """
        if not content or not section_text:
            return 0.0
            
        # Vectorize content and section text together
        all_texts = [self.preprocess_text(content), self.preprocess_text(section_text)]
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return float(similarity)
    
    def _calculate_keyword_match_score(self, content: str, section_text: str) -> float:
        """Calculate keyword matching score.
        
        Args:
            content: Content to classify
            section_text: Text to compare against
            
        Returns:
            Keyword matching score
        """
        content_words = set(self.preprocess_text(content).split())
        section_words = set(self.preprocess_text(section_text).split())
        
        if not content_words or not section_words:
            return 0.0
            
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(content_words.intersection(section_words))
        union = len(content_words.union(section_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_title_match_score(self, content: str, section_title: str) -> float:
        """Calculate title matching score with emphasis.
        
        Args:
            content: Content to classify
            section_title: Title to compare against
            
        Returns:
            Title matching score (weighted more heavily)
        """
        content_words = set(self.preprocess_text(content).split())
        title_words = set(self.preprocess_text(section_title).split())
        
        if not content_words or not title_words:
            return 0.0
            
        # Count exact matches and partial matches
        exact_matches = len(content_words.intersection(title_words))
        partial_matches = sum(1 for cw in content_words for tw in title_words if cw in tw or tw in cw)
        
        # Weight exact matches more heavily
        weighted_matches = exact_matches * 2 + partial_matches * 0.5
        max_possible = len(title_words) * 2  # Max possible weighted score
        
        return min(weighted_matches / max_possible, 1.0) if max_possible > 0 else 0.0
    
    def _enhance_matching_with_keywords(self, content: str, outline_section: Dict, base_score: float) -> float:
        """Enhance matching score with keyword analysis.
        
        Args:
            content: Content to classify
            outline_section: Outline section to match against
            base_score: Base similarity score
            
        Returns:
            Enhanced score
        """
        # Extract keywords from both content and outline section
        content_keywords = set(self.extract_keywords(content))
        section_text = f"{outline_section.get('title', '')} {outline_section.get('description', '')}"
        section_keywords = set(self.extract_keywords(section_text))
        
        # Calculate keyword overlap
        overlap = len(content_keywords.intersection(section_keywords))
        total_keywords = len(content_keywords.union(section_keywords))
        
        if total_keywords > 0:
            keyword_score = overlap / total_keywords
            # Boost the base score with keyword matching (up to 30% boost)
            enhanced_score = base_score + (keyword_score * 0.3)
            return min(enhanced_score, 1.0)  # Cap at 1.0
        else:
            return base_score
    
    def classify_content_batch(self, contents: List[str], outline_data: Dict) -> List[Dict]:
        """Classify multiple contents in batch for better performance.
        
        Args:
            contents: List of contents to classify
            outline_data: The book outline data
            
        Returns:
            List of classification results
        """
        print(f"Batch classifying {len(contents)} contents...")
        
        results = []
        for i, content in enumerate(contents):
            print(f"Processing content {i+1}/{len(contents)}...")
            result = self.classify_content(content, outline_data)
            results.append(result)
        
        print(f"Batch classification completed. Processed {len(results)} contents.")
        return results
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache hit/miss statistics
        """
        return self.cache_stats.copy()
    
    def clear_cache(self):
        """Clear all caches."""
        self.chapters_vector_cache.clear()
        self.subtopics_vector_cache.clear()
        self.content_vector_cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}


    def _expand_with_synonyms(self, text: str) -> str:
        """Expand text with synonyms and common variations for better matching.
        
        Args:
            text: Input text to expand
            
        Returns:
            Expanded text with synonyms
        """
        expanded_text = text.lower()
        
        # Apply synonyms and common variations
        for term, variations in self.synonyms.items():
            if term in expanded_text:
                # Add variations to the text for better matching
                expanded_text += " " + " ".join(variations)
        
        return expanded_text
    
    def _calculate_fuzzy_match_score(self, content: str, section_title: str) -> float:
        """Calculate fuzzy string matching score for close matches.
        
        Args:
            content: Content to classify
            section_title: Section title to match against
            
        Returns:
            Fuzzy matching score between 0.0 and 1.0
        """
        # Calculate fuzzy matching ratio
        fuzzy_ratio = difflib.SequenceMatcher(None, content.lower(), section_title.lower()).ratio()
        
        # Boost score for partial matches of important terms
        content_words = set(content.lower().split())
        title_words = set(section_title.lower().split())
        
        # Check for partial word matches (substring matching)
        partial_matches = 0
        total_checks = 0
        
        for cw in content_words:
            for tw in title_words:
                total_checks += 1
                if cw in tw or tw in cw:  # Substring matching
                    partial_matches += 1
        
        partial_ratio = partial_matches / total_checks if total_checks > 0 else 0
        
        # Combine fuzzy ratio with partial matching
        combined_score = (fuzzy_ratio * 0.7) + (partial_ratio * 0.3)
        
        return min(combined_score, 1.0)
    
    def multi_level_classify(self, content: str, outline_data: Dict) -> Dict:
        """
        Perform multi-level classification (part -> chapter -> subtopic) with confidence propagation.
        
        Args:
            content: The content to classify
            outline_data: The book outline data
            
        Returns:
            A dictionary with multi-level classification results
        """
        print(f"Multi-level classifying content: {content[:100]}...")
        
        if not content or not outline_data:
            print("Empty content or outline data, returning empty classification")
            return {
                "part": None,
                "chapter": None,
                "subtopic": None,
                "part_score": 0.0,
                "chapter_score": 0.0,
                "subtopic_score": 0.0,
                "confidence_propagation": 0.0
            }
        
        # Extract all parts, chapters, and subtopics
        parts = []
        part_texts = []
        
        for part in outline_data.get("parts", []):
            parts.append({
                "id": part["id"],
                "title": part["title"],
                "description": part.get("description", "")
            })
            part_texts.append(f"{part['title']} {part.get('description', '')}")
        
        if not parts:
            return {
                "part": None, "chapter": None, "subtopic": None,
                "part_score": 0.0, "chapter_score": 0.0, "subtopic_score": 0.0,
                "confidence_propagation": 0.0
            }
        
        # Calculate part similarity scores
        print("Calculating part scores...")
        part_similarities = self._calculate_semantic_similarity_faiss(content, part_texts)
        
        # Find best matching part
        best_part_idx = np.argmax(part_similarities)
        best_part_score = part_similarities[best_part_idx]
        best_part = parts[best_part_idx]
        
        print(f"Best part match: {best_part['title']} (score: {best_part_score:.4f})")
        
        # Now find the best chapter within the best part
        best_chapter = None
        best_chapter_score = 0.0
        best_subtopic = None
        best_subtopic_score = 0.0
        
        # Get chapters from the best matching part
        for part in outline_data.get("parts", []):
            if part["id"] == best_part["id"]:
                chapters = []
                chapter_texts = []
                
                for chapter in part.get("chapters", []):
                    chapters.append({
                        "id": chapter["id"],
                        "title": chapter["title"],
                        "description": chapter.get("description", ""),
                        "part_id": part["id"]
                    })
                    chapter_texts.append(f"{chapter['title']} {chapter.get('description', '')}")
                
                if chapters:
                    print("Calculating chapter scores within best part...")
                    chapter_similarities = self._calculate_semantic_similarity_faiss(content, chapter_texts)
                    
                    # Find best matching chapter
                    best_chapter_idx = np.argmax(chapter_similarities)
                    best_chapter_score = chapter_similarities[best_chapter_idx]
                    best_chapter = chapters[best_chapter_idx]
                    
                    print(f"Best chapter match: {best_chapter['title']} (score: {best_chapter_score:.4f})")
                    
                    # Now find the best subtopic within the best chapter
                    for chapter in part.get("chapters", []):
                        if chapter["id"] == best_chapter["id"]:
                            subtopics = []
                            subtopic_texts = []
                            
                            for subtopic in chapter.get("subtopics", []):
                                subtopics.append({
                                    "id": subtopic["id"],
                                    "title": subtopic["title"],
                                    "description": subtopic.get("description", ""),
                                    "chapter_id": chapter["id"]
                                })
                                subtopic_texts.append(f"{subtopic['title']} {subtopic.get('description', '')}")
                            
                            if subtopics:
                                print("Calculating subtopic scores within best chapter...")
                                subtopic_similarities = self._calculate_semantic_similarity_faiss(content, subtopic_texts)
                                
                                # Find best matching subtopic
                                best_subtopic_idx = np.argmax(subtopic_similarities)
                                best_subtopic_score = subtopic_similarities[best_subtopic_idx]
                                best_subtopic = subtopics[best_subtopic_idx]
                                
                                print(f"Best subtopic match: {best_subtopic['title']} (score: {best_subtopic_score:.4f})")
                                break
                break
        
        # Calculate confidence propagation score
        # This represents how confident we are in the entire hierarchical classification
        confidence_propagation = (best_part_score + best_chapter_score + best_subtopic_score) / 3
        
        result = {
            "part": best_part,
            "chapter": best_chapter,
            "subtopic": best_subtopic,
            "part_score": float(best_part_score),
            "chapter_score": float(best_chapter_score),
            "subtopic_score": float(best_subtopic_score),
            "confidence_propagation": confidence_propagation
        }
        
        print(f"Final multi-level classification result: {result}")
        return result

    def fuzzy_classify(self, content: str, outline_data: Dict, threshold: float = 0.3) -> List[Dict]:
        """
        Perform fuzzy classification that returns multiple possible classifications
        above a certain threshold.
        
        Args:
            content: The content to classify
            outline_data: The book outline data
            threshold: Minimum score threshold for inclusion
            
        Returns:
            A list of possible classifications above the threshold
        """
        print(f"Fuzzy classifying content: {content[:100]}... (threshold: {threshold})")
        
        if not content or not outline_data:
            print("Empty content or outline data, returning empty classification")
            return []
        
        # Extract all chapters and subtopics to compare against
        chapters = []
        chapter_texts = []
        subtopics = []
        subtopic_texts = []
        
        for part in outline_data.get("parts", []):
            for chapter in part.get("chapters", []):
                chapters.append({
                    "id": chapter["id"],
                    "title": chapter["title"],
                    "description": chapter.get("description", ""),
                    "part_id": part["id"]
                })
                chapter_texts.append(f"{chapter['title']} {chapter.get('description', '')}")
                
                for subtopic in chapter.get("subtopics", []):
                    subtopics.append({
                        "id": subtopic["id"],
                        "title": subtopic["title"],
                        "description": subtopic.get("description", ""),
                        "chapter_id": chapter["id"]
                    })
                    subtopic_texts.append(f"{subtopic['title']} {subtopic.get('description', '')}")
        
        results = []
        
        # Calculate chapter similarities
        if chapter_texts:
            print(f"Calculating similarities to {len(chapter_texts)} chapters...")
            chapter_similarities = self._calculate_semantic_similarity_faiss(content, chapter_texts)
            
            for i, similarity in enumerate(chapter_similarities):
                if similarity >= threshold:
                    results.append({
                        "type": "chapter",
                        "element": chapters[i],
                        "score": float(similarity),
                        "rank": i
                    })
        
        # Calculate subtopic similarities
        if subtopic_texts:
            print(f"Calculating similarities to {len(subtopic_texts)} subtopics...")
            subtopic_similarities = self._calculate_semantic_similarity_faiss(content, subtopic_texts)
            
            for i, similarity in enumerate(subtopic_similarities):
                if similarity >= threshold:
                    results.append({
                        "type": "subtopic", 
                        "element": subtopics[i],
                        "score": float(similarity),
                        "rank": i
                    })
        
        # Sort results by score in descending order
        results.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"Found {len(results)} fuzzy matches above threshold {threshold}")
        return results

    def classify_by_topic_modeling(self, content: str, outline_data: Dict) -> Dict:
        """
        Classify content using topic modeling approach alongside traditional methods.
        This method uses a hybrid approach combining topic modeling with traditional similarity.
        
        Args:
            content: The content to classify
            outline_data: The book outline data
            
        Returns:
            A dictionary with classification results
        """
        print(f"Topic modeling classification for content: {content[:100]}...")
        
        if not content or not outline_data:
            print("Empty content or outline data, returning empty classification")
            return {
                "chapter": None,
                "subtopic": None,
                "chapter_score": 0.0,
                "subtopic_score": 0.0,
                "method_used": "none"
            }
        
        # Extract all possible elements
        chapters = []
        chapter_texts = []
        subtopics = []
        subtopic_texts = []
        
        for part in outline_data.get("parts", []):
            for chapter in part.get("chapters", []):
                chapters.append({
                    "id": chapter["id"],
                    "title": chapter["title"],
                    "description": chapter.get("description", ""),
                    "part_id": part["id"]
                })
                chapter_texts.append(f"{chapter['title']} {chapter.get('description', '')}")
                
                for subtopic in chapter.get("subtopics", []):
                    subtopics.append({
                        "id": subtopic["id"],
                        "title": subtopic["title"],
                        "description": subtopic.get("description", ""),
                        "chapter_id": chapter["id"]
                    })
                    subtopic_texts.append(f"{subtopic['title']} {subtopic.get('description', '')}")
        
        # Get the traditional classification result
        traditional_result = self.classify_content(content, outline_data)
        
        # Use topic modeling approach by finding related topics
        topic_modeling_result = self._find_topics_in_content(content)
        
        # Look for matches in outline based on identified topics
        best_topic_chapter = None
        best_topic_subtopic = None
        best_topic_chapter_score = 0.0
        best_topic_subtopic_score = 0.0
        
        if topic_modeling_result:
            # Create a synthetic text from extracted topics
            topics_text = " ".join(topic_modeling_result)
            
            # Find chapters that match the topics
            for i, chapter_text in enumerate(chapter_texts):
                # Calculate similarity between topic text and chapter
                topic_sim = self._calculate_semantic_similarity(topics_text, chapter_text)
                if topic_sim > best_topic_chapter_score:
                    best_topic_chapter_score = topic_sim
                    best_topic_chapter = chapters[i]
            
            # Find subtopics that match the topics
            for i, subtopic_text in enumerate(subtopic_texts):
                topic_sim = self._calculate_semantic_similarity(topics_text, subtopic_text)
                if topic_sim > best_topic_subtopic_score:
                    best_topic_subtopic_score = topic_sim
                    best_topic_subtopic = subtopics[i]
        
        # Combine traditional and topic modeling results
        if best_topic_chapter_score > traditional_result["chapter_score"]:
            final_chapter = best_topic_chapter
            final_chapter_score = best_topic_chapter_score
            method_used = "topic_modeling"
        else:
            final_chapter = traditional_result["chapter"]
            final_chapter_score = traditional_result["chapter_score"]
            method_used = "traditional"
        
        if best_topic_subtopic_score > traditional_result["subtopic_score"]:
            final_subtopic = best_topic_subtopic
            final_subtopic_score = best_topic_subtopic_score
            method_used = "topic_modeling"  # Overall method, so update if subtopic is more confident
        else:
            final_subtopic = traditional_result["subtopic"]
            final_subtopic_score = traditional_result["subtopic_score"]
        
        result = {
            "chapter": final_chapter,
            "subtopic": final_subtopic,
            "chapter_score": float(final_chapter_score),
            "subtopic_score": float(final_subtopic_score),
            "method_used": method_used
        }
        
        print(f"Topic modeling classification result: {result}")
        return result

    def find_topics_in_content(self, content: str) -> List[str]:
        """
        Perform topic modeling on content by extracting key topics/phrases.
        
        Args:
            content: Content to analyze for topics
            
        Returns:
            List of identified topics/phrases
        """
        # Check if result is already in topic cache
        cache_key = self._get_topic_cache_key(content)
        if cache_key in self.topic_extraction_cache:
            self.cache_stats["hits"] += 1
            print(f"Topic cache hit for content: {content[:50]}...")
            return self.topic_extraction_cache[cache_key]
        
        print(f"Finding topics in content: {content[:50]}...")
        
        # Use more sophisticated approach if available
        if self.lemmatizer:
            topics = self._advanced_topic_extraction(content)
        else:
            # Use simpler approach
            processed_content = self.preprocess_text(content)
            topics = self._extract_noun_phrases(processed_content)
            
            # If NLTK approach didn't work, fall back to keyword extraction
            if not topics:
                topics = self.extract_keywords(content, top_n=5)
        
        # Limit cache size to prevent memory issues
        if len(self.topic_extraction_cache) >= self.cache_max_size:
            # Remove a random key to maintain size limit
            import random
            keys = list(self.topic_extraction_cache.keys())
            if keys:
                del self.topic_extraction_cache[random.choice(keys)]
        
        # Store in cache
        self.topic_extraction_cache[cache_key] = topics
        self.cache_stats["misses"] += 1
        
        print(f"Extracted topics: {topics}")
        return topics

    def _advanced_topic_extraction(self, content: str) -> List[str]:
        """
        Advanced topic extraction using multiple techniques.
        
        Args:
            content: Content to analyze for topics
            
        Returns:
            List of extracted topics
        """
        # Try noun phrase extraction first
        topics = self._extract_noun_phrases(content)
        
        # If not sufficient topics found, enhance with keyword extraction
        if len(topics) < 3:
            keywords = self.extract_keywords(content, top_n=5)
            topics = list(set(topics + keywords))
        
        # If we have access to more sophisticated topic modeling libraries,
        # we could use them here (e.g., BERTopic, LDA)
        return topics

    def cluster_similar_content(self, contents: List[str], n_clusters: Optional[int] = None) -> List[List[str]]:
        """
        Cluster similar content items to identify common topics/themes.
        
        Args:
            contents: List of content strings to cluster
            n_clusters: Number of clusters (if None, auto-determined)
            
        Returns:
            List of clusters, each containing similar content items
        """
        if len(contents) < 2:
            return [contents] if contents else []
        
        # Use the TF-IDF vectorizer to convert content to vectors
        tfidf_matrix = self.vectorizer.fit_transform(contents)
        
        # Determine number of clusters if not provided
        if n_clusters is None:
            # Use a heuristic based on content count
            n_clusters = min(max(2, len(contents) // 3), len(contents))
        
        # Perform clustering using scikit-learn
        try:
            from sklearn.cluster import KMeans
            clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clustering_model.fit_predict(tfidf_matrix.toarray())
            
            # Group content by cluster
            clusters = [[] for _ in range(n_clusters)]
            for i, label in enumerate(cluster_labels):
                clusters[label].append(contents[i])
            
            return clusters
        except ImportError:
            print("Scikit-learn KMeans not available, using fallback clustering")
            # Simple fallback clustering using hierarchical clustering
            from sklearn.cluster import AgglomerativeClustering
            clustering_model = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
            cluster_labels = clustering_model.fit_predict(tfidf_matrix.toarray())
            
            # Group content by cluster
            clusters = [[] for _ in range(n_clusters)]
            for i, label in enumerate(cluster_labels):
                clusters[label].append(contents[i])
            
            return clusters

    def _find_topics_in_content(self, content: str) -> List[str]:
        """
        Simulate topic modeling on content by extracting key topics/phrases.
        This would ideally use libraries like Gensim's LDA or BERTopic in a production system.
        
        Args:
            content: Content to analyze for topics
            
        Returns:
            List of identified topics/phrases
        """
        print(f"Finding topics in content: {content[:50]}...")
        
        # This is a simplified implementation of topic modeling
        # In a real system, you'd use proper topic modeling algorithms
        processed_content = self.preprocess_text(content)
        
        # Extract noun phrases as potential topics
        topics = self._extract_noun_phrases(processed_content)
        
        # If NLTK approach didn't work, fall back to keyword extraction
        if not topics:
            topics = self.extract_keywords(content, top_n=5)
        
        print(f"Extracted topics: {topics}")
        return topics

    def _extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases as potential topics.
        Uses NLTK if available, otherwise a simple keyword-based approach.
        """
        # This is a simplified implementation
        # A proper implementation would use NLTK's POS tagging and chunking
        try:
            import nltk
            # Check if required NLTK data is available
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                print("Required NLTK data not available, using fallback")
                return []
            
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            
            # Simple noun phrase pattern: DT (optional) + JJ* + NN* + NNS/NNP/NNPS
            noun_phrases = []
            current_phrase = []
            
            for word, pos in pos_tags:
                if pos.startswith('NN') or pos in ['JJ', 'DT']:
                    current_phrase.append(word)
                else:
                    if len(current_phrase) > 1:
                        noun_phrases.append(' '.join(current_phrase))
                    current_phrase = []
            
            # Add the last phrase if it exists
            if len(current_phrase) > 1:
                noun_phrases.append(' '.join(current_phrase))
            
            return noun_phrases
        except ImportError:
            print("NLTK not available, using simple keyword extraction as topic modeling fallback")
            # Fallback to keyword extraction
            return self.extract_keywords(text, top_n=5)

    def _preprocess_text_advanced(self, text: str) -> str:
        """Advanced text preprocessing with lemmatization and stop word removal.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Apply NLTK preprocessing if available
        if self.lemmatizer and self.stop_words:
            try:
                # Tokenize the text
                tokens = word_tokenize(text)
                
                # Remove stop words and lemmatize
                filtered_tokens = [
                    self.lemmatizer.lemmatize(token) 
                    for token in tokens 
                    if token not in self.stop_words and len(token) > 1
                ]
                
                # Join tokens back to text
                text = ' '.join(filtered_tokens)
            except Exception as e:
                print(f"Warning: Failed to apply advanced preprocessing: {e}")
        
        return text


_NLP_CLASSIFIER_SINGLETON: Optional[NLPContentClassifier] = None


def create_nlp_classifier() -> NLPContentClassifier:
    """Factory function to create or return a cached NLP content classifier.
    
    This ensures the underlying SentenceTransformer model is loaded only once
    per process, preventing repeated downloads and heavy re-initialization.
    
    Returns:
        A singleton instance of NLPContentClassifier
    """
    global _NLP_CLASSIFIER_SINGLETON
    if _NLP_CLASSIFIER_SINGLETON is None:
        _NLP_CLASSIFIER_SINGLETON = NLPContentClassifier()
    return _NLP_CLASSIFIER_SINGLETON