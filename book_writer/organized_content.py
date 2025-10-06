"""
Book Writer System - Enhanced Content Organization Module
Provides advanced algorithms for organizing book chapters, content structure, 
and improved selection note categorization
"""
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import graphviz

from book_writer.outline import BookOutline
from book_writer.note_processor import NoteProcessor, ContentManager
from book_writer.nlp_classifier import NLPContentClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContentRelationship:
    """Represents a relationship between content items."""
    source_id: str
    target_id: str
    relationship_type: str  # 'sequential', 'parallel', 'prerequisite', 'complementary', 'conflict'
    strength: float = 1.0  # 0.0 to 1.0, indicating strength of relationship
    metadata: Dict = field(default_factory=dict)


@dataclass
class ContentCluster:
    """Represents a cluster of related content items."""
    id: str
    title: str
    content_ids: List[str]
    description: str = ""
    topic_keywords: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class OrganizationMetrics:
    """Metrics for evaluating content organization quality."""
    completeness_score: float = 0.0
    flow_score: float = 0.0
    cohesion_score: float = 0.0
    gap_count: int = 0
    suggestions_count: int = 0


class ContentOrganizer:
    """
    An advanced content organization system that provides algorithms for clustering,
    flow optimization, gap detection, and relationship mapping.

    This class is designed for developers who need fine-grained control over the
    content organization process. For a simpler, high-level interface, use the
    `ContentOrganization` class.

    The `ContentOrganizer` uses TF-IDF for content vectorization and Agglomerative
    Clustering for creating content clusters. It also provides methods for
    detecting content gaps, suggesting reorganization, and managing content
    relationships.
    """
    
    def __init__(self, project_path: Union[str, Path], 
                 note_processor: NoteProcessor, 
                 content_manager: ContentManager):
        """
        Initializes the ContentOrganizer.

        Args:
            project_path: The path to the project directory.
            note_processor: A NoteProcessor instance for note embeddings.
            content_manager: A ContentManager instance for content operations.
        """
        self.project_path = Path(project_path)
        self.note_processor = note_processor
        self.content_manager = content_manager
        
        # Initialize NLP classifier for enhanced categorization
        self.nlp_classifier = NLPContentClassifier()
        
        # Storage for relationships and clusters
        self.relationships: List[ContentRelationship] = []
        self.clusters: List[ContentCluster] = []
        
        # Vectorizer for content similarity calculations
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=10000,
            lowercase=True
        )
        
        # Load from persistent storage if it exists
        self._load_organization_data()
    
    def _load_organization_data(self):
        """Load organization data from persistent storage."""
        org_dir = self.project_path / "org_data"
        org_dir.mkdir(exist_ok=True, parents=True)
        
        # Load relationships
        relationships_file = org_dir / "relationships.json"
        if relationships_file.exists():
            try:
                with open(relationships_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.relationships = [
                    ContentRelationship(
                        source_id=rel["source_id"],
                        target_id=rel["target_id"],
                        relationship_type=rel["relationship_type"],
                        strength=rel["strength"],
                        metadata=rel.get("metadata", {})
                    ) for rel in data
                ]
            except Exception as e:
                logger.warning(f"Could not load relationships: {e}")
        
        # Load clusters
        clusters_file = org_dir / "clusters.json"
        if clusters_file.exists():
            try:
                with open(clusters_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.clusters = [
                    ContentCluster(
                        id=cluster["id"],
                        title=cluster["title"],
                        content_ids=cluster["content_ids"],
                        description=cluster["description"],
                        topic_keywords=cluster.get("topic_keywords", []),
                        confidence=cluster.get("confidence", 1.0)
                    ) for cluster in data
                ]
            except Exception as e:
                logger.warning(f"Could not load clusters: {e}")
    
    def _save_organization_data(self):
        """Save organization data to persistent storage."""
        org_dir = self.project_path / "org_data"
        org_dir.mkdir(exist_ok=True, parents=True)
        
        # Save relationships
        relationships_data = [
            {
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "relationship_type": rel.relationship_type,
                "strength": rel.strength,
                "metadata": rel.metadata
            } for rel in self.relationships
        ]
        with open(org_dir / "relationships.json", "w", encoding="utf-8") as f:
            json.dump(relationships_data, f, indent=2)
        
        # Save clusters
        clusters_data = [
            {
                "id": cluster.id,
                "title": cluster.title,
                "content_ids": cluster.content_ids,
                "description": cluster.description,
                "topic_keywords": cluster.topic_keywords,
                "confidence": cluster.confidence
            } for cluster in self.clusters
        ]
        with open(org_dir / "clusters.json", "w", encoding="utf-8") as f:
            json.dump(clusters_data, f, indent=2)
    
    def create_content_clusters(self, 
                                content_ids: List[str], 
                                n_clusters: Optional[int] = None,
                                min_topic_similarity: float = 0.3) -> List[ContentCluster]:
        """
        Creates content clusters using topic modeling and similarity analysis.

        This method uses a TF-IDF vectorizer to convert the content into a matrix
        of TF-IDF features. Then, it uses an Agglomerative Clustering model to
        group the content into clusters.

        Args:
            content_ids: A list of content IDs to cluster.
            n_clusters: The number of clusters to create. If None, the number of
                clusters is automatically determined based on the number of content
                items.
            min_topic_similarity: The minimum similarity threshold for clustering.

        Returns:
            A list of ContentCluster objects.
        """
        if len(content_ids) < 2:
            logger.warning("Need at least 2 content items to create clusters")
            return []
        
        # Retrieve content texts
        contents = []
        content_texts = []
        for content_id in content_ids:
            try:
                content_text = self.content_manager.get_content(content_id)
                content_texts.append(content_text)
                contents.append((content_id, content_text))
            except FileNotFoundError:
                logger.warning(f"Content {content_id} not found, skipping")
                continue
        
        if len(content_texts) < 2:
            logger.warning("Not enough content to create clusters")
            return []
        
        # Vectorize content
        tfidf_matrix = self.vectorizer.fit_transform(content_texts)
        
        # Determine number of clusters if not provided
        if n_clusters is None:
            # Use a heuristic based on content count
            n_clusters = min(max(2, len(content_texts) // 3), len(content_texts))
        
        # Perform clustering
        clustering_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        
        cluster_labels = clustering_model.fit_predict(tfidf_matrix.toarray())
        
        # Group content by cluster
        clusters_by_label = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters_by_label[label].append(contents[i])
        
        # Create cluster objects
        clusters = []
        for label, content_list in clusters_by_label.items():
            cluster_id = str(uuid.uuid4())
            
            # Extract topic keywords using TF-IDF for the cluster
            cluster_texts = [text for _, text in content_list]
            cluster_tfidf = self.vectorizer.fit_transform(cluster_texts)
            # Get top terms for the cluster
            feature_names = self.vectorizer.get_feature_names_out()
            cluster_topic_keywords = self._get_top_terms(cluster_tfidf, feature_names, top_k=5)
            
            # Create cluster content IDs list
            cluster_content_ids = [content_id for content_id, _ in content_list]
            
            # Create cluster title based on top keywords
            cluster_title = " ".join(cluster_topic_keywords[:3])
            
            cluster = ContentCluster(
                id=cluster_id,
                title=cluster_title,
                content_ids=cluster_content_ids,
                topic_keywords=cluster_topic_keywords,
                confidence=self._calculate_cluster_confidence(cluster_tfidf)
            )
            
            clusters.append(cluster)
        
        # Store clusters
        self.clusters = clusters
        self._save_organization_data()
        
        return clusters
    
    def _get_top_terms(self, tfidf_matrix, feature_names, top_k: int = 5) -> List[str]:
        """
        Get top terms for a cluster based on TF-IDF scores.
        
        Args:
            tfidf_matrix: TF-IDF matrix for the cluster
            feature_names: Feature names from the vectorizer
            top_k: Number of top terms to return
            
        Returns:
            List of top terms
        """
        # Average TF-IDF scores across all documents in the cluster
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        
        # Get indices of top terms
        top_indices = mean_scores.argsort()[-top_k:][::-1]
        
        # Get corresponding feature names
        top_terms = [feature_names[i] for i in top_indices if i < len(feature_names)]
        
        return top_terms
    
    def _calculate_cluster_confidence(self, tfidf_matrix) -> float:
        """
        Calculate confidence score for a cluster based on internal similarity.
        
        Args:
            tfidf_matrix: TF-IDF matrix for the cluster
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if tfidf_matrix.shape[0] < 2:
            return 1.0  # Single item cluster gets maximum confidence
        
        # Calculate similarities between all pairs in the cluster
        similarities = cosine_similarity(tfidf_matrix)
        
        # Exclude diagonal (self-similarity) and calculate average
        mask = np.ones(similarities.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        avg_similarity = similarities[mask].mean()
        
        # Normalize to 0-1 range
        return min(max(avg_similarity, 0.0), 1.0)
    
    def detect_content_gaps(self, outline: BookOutline) -> List[Dict]:
        """
        Detects potential content gaps in the book structure.

        This method analyzes the book outline to identify chapters and subtopics
        that are missing content. It also checks for gaps in the content flow
        within chapters.

        Args:
            outline: The book outline to analyze.

        Returns:
            A list of detected gaps, with suggestions for how to fill them.
        """
        gaps = []
        
        # Analyze each part and chapter for potential gaps
        for part_idx, part in enumerate(outline.parts):
            for chapter_idx, chapter in enumerate(part["chapters"]):
                # Get all content assigned to this chapter
                chapter_content = []
                for subtopic in chapter["subtopics"]:
                    subtopic_content = self.content_manager.retrieve_content_by_subtopic(subtopic["id"])
                    chapter_content.extend(subtopic_content)
                
                # Calculate content distribution metrics
                if not chapter_content:
                    gaps.append({
                        "type": "missing_content",
                        "location": f"Part {part_idx+1}: {part['title']} -> Chapter {chapter_idx+1}: {chapter['title']}",
                        "severity": "high",
                        "suggestion": f"Add content to chapter '{chapter['title']}' to fulfill its purpose: {chapter.get('description', 'No description provided')}"
                    })
                else:
                    # Check for possible gaps in content flow within chapter
                    content_texts = [item["content"] for item in chapter_content]
                    
                    # Use similarity to detect potential missing content areas
                    if len(content_texts) > 1:
                        similarities = cosine_similarity(self.vectorizer.fit_transform(content_texts))
                        # Find low similarity pairs that might indicate missing connecting content
                        for i in range(len(similarities)):
                            for j in range(i+1, len(similarities)):
                                if similarities[i][j] < 0.2:  # Low similarity
                                    gaps.append({
                                        "type": "content_flow_gap",
                                        "location": f"Chapter: {chapter['title']}",
                                        "severity": "medium",
                                        "suggestion": f"Consider adding connecting content between content items {i} and {j} in chapter '{chapter['title']}' for better flow"
                                    })
        
        return gaps
    
    def suggest_content_reorganization(self, outline: BookOutline) -> List[Dict]:
        """
        Suggests content reorganization based on content similarity and flow.

        This method analyzes the content across the entire book and suggests
        moving content items to more suitable locations based on their similarity
        to other content.

        Args:
            outline: The current book outline.

        Returns:
            A list of reorganization suggestions.
        """
        suggestions = []
        
        # Analyze content across the entire book
        all_content = []
        content_metadata = {}
        
        for part in outline.parts:
            for chapter in part["chapters"]:
                for subtopic in chapter["subtopics"]:
                    subtopic_content = self.content_manager.retrieve_content_by_subtopic(subtopic["id"])
                    for item in subtopic_content:
                        all_content.append(item["content"])
                        content_metadata[item["id"]] = {
                            "content_id": item["id"],
                            "chapter_id": chapter["id"],
                            "subtopic_id": subtopic["id"],
                            "chapter_title": chapter["title"],
                            "subtopic_title": subtopic["title"]
                        }
        
        if len(all_content) < 2:
            return []
        
        # Create TF-IDF matrix for all content
        tfidf_matrix = self.vectorizer.fit_transform(all_content)
        
        # Calculate content similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find content items that might be better placed elsewhere
        for i, (content_id, metadata) in enumerate(content_metadata.items()):
            current_similarities = similarity_matrix[i]
            
            # Find most similar content that's not in the same subtopic
            most_similar_idx = -1
            max_similarity = 0
            for j, (other_content_id, other_metadata) in enumerate(content_metadata.items()):
                if i != j and current_similarities[j] > max_similarity:
                    # Only suggest if content is in a different subtopic
                    if metadata["subtopic_id"] != other_metadata["subtopic_id"]:
                        max_similarity = current_similarities[j]
                        most_similar_idx = j
            
            if most_similar_idx != -1 and max_similarity > 0.7:
                suggestions.append({
                    "type": "content_relocation",
                    "content_id": content_id,
                    "current_location": f"{metadata['chapter_title']} -> {metadata['subtopic_title']}",
                    "suggested_location": f"{content_metadata[all_content[most_similar_idx]['id']]['chapter_title']} -> {content_metadata[all_content[most_similar_idx]['id']]['subtopic_title']}",
                    "similarity_score": max_similarity,
                    "reason": f"Content is highly similar to content in the suggested location (similarity: {max_similarity:.2f})"
                })
        
        return suggestions
    
    def create_content_relationships(self, 
                                   source_content_id: str, 
                                   target_content_id: str, 
                                   relationship_type: str,
                                   strength: float = 1.0) -> bool:
        """
        Creates a relationship between two content items.

        This method allows you to define a relationship between two content items,
        such as 'sequential', 'parallel', 'prerequisite', 'complementary', or
        'conflict'.

        Args:
            source_content_id: The ID of the source content item.
            target_content_id: The ID of the target content item.
            relationship_type: The type of relationship.
            strength: The strength of the relationship (from 0.0 to 1.0).

        Returns:
            True if the relationship was created successfully, False otherwise.
        """
        # Validate content IDs exist
        try:
            self.content_manager.get_content(source_content_id)
            self.content_manager.get_content(target_content_id)
        except FileNotFoundError:
            logger.error(f"One or both content IDs do not exist: {source_content_id}, {target_content_id}")
            return False
        
        # Remove existing relationship if it exists
        self.relationships = [
            rel for rel in self.relationships 
            if not (rel.source_id == source_content_id and rel.target_id == target_content_id)
        ]
        
        # Create new relationship
        relationship = ContentRelationship(
            source_id=source_content_id,
            target_id=target_content_id,
            relationship_type=relationship_type,
            strength=strength
        )
        
        self.relationships.append(relationship)
        self._save_organization_data()
        
        return True
    
    def get_related_content(self, content_id: str, relationship_type: Optional[str] = None) -> List[Dict]:
        """
        Gets content related to the specified content ID.

        This method retrieves all content items that have a relationship with the
        specified content item.

        Args:
            content_id: The ID of the content item to find relations for.
            relationship_type: An optional filter for a specific relationship type.

        Returns:
            A list of related content items, with relationship information.
        """
        related_content = []
        
        # Find relationships where this content is the source
        for rel in self.relationships:
            if rel.source_id == content_id:
                if relationship_type is None or rel.relationship_type == relationship_type:
                    try:
                        content = self.content_manager.get_content(rel.target_id)
                        related_content.append({
                            "content_id": rel.target_id,
                            "content": content,
                            "relationship_type": rel.relationship_type,
                            "strength": rel.strength
                        })
                    except FileNotFoundError:
                        logger.warning(f"Related content {rel.target_id} not found")
        
        # Find relationships where this content is the target
        for rel in self.relationships:
            if rel.target_id == content_id:
                if relationship_type is None or rel.relationship_type == relationship_type:
                    try:
                        content = self.content_manager.get_content(rel.source_id)
                        related_content.append({
                            "content_id": rel.source_id,
                            "content": content,
                            "relationship_type": rel.relationship_type,
                            "strength": rel.strength
                        })
                    except FileNotFoundError:
                        logger.warning(f"Related content {rel.source_id} not found")
        
        return related_content
    
    def improve_note_classification(self, 
                                  note_id: str, 
                                  current_outline: BookOutline,
                                  additional_context: Optional[Dict] = None) -> Dict:
        """
        Improves note classification with additional context and confidence scoring.

        This method uses an NLP classifier to classify a note and then enhances the
        classification with additional information, such as alternative
        classifications and contextual relevance.

        Args:
            note_id: The ID of the note to classify.
            current_outline: The current book outline for reference.
            additional_context: Additional context to consider during classification.

        Returns:
            A dictionary with the enhanced classification result.
        """
        # Get the note from ChromaDB
        results = self.note_processor.notes_collection.get(
            ids=[note_id],
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            raise ValueError(f"Note with ID {note_id} not found")
        
        note_text = results["documents"][0]
        
        # Use the enhanced NLP classifier
        classification = self.nlp_classifier.classify_content(note_text, current_outline.to_dict())
        
        # Enhance classification with additional metrics
        enhanced_result = {
            "original_classification": classification,
            "note_id": note_id,
            "content_preview": note_text[:200] + "..." if len(note_text) > 200 else note_text,
            "confidence_boost": 0.0,
            "alternative_classifications": [],
            "contextual_relevance": 0.0
        }
        
        # Calculate alternative classifications based on context
        if additional_context:
            # Use additional context to refine classification
            if "previous_section" in additional_context:
                # Check similarity between note and previous section
                prev_content = additional_context["previous_section"]
                try:
                    similarity = cosine_similarity(
                        self.vectorizer.fit_transform([note_text, prev_content])
                    )[0][1]
                    enhanced_result["contextual_relevance"] = similarity
                except:
                    enhanced_result["contextual_relevance"] = 0.0
        
        # Generate alternative classifications if confidence is low
        if classification["chapter_score"] < 0.5 or classification["subtopic_score"] < 0.5:
            alternatives = self._find_alternative_classifications(note_text, current_outline)
            enhanced_result["alternative_classifications"] = alternatives
        
        return enhanced_result
    
    def _find_alternative_classifications(self, note_text: str, current_outline: BookOutline) -> List[Dict]:
        """
        Find alternative classifications for a note based on different matching strategies.
        
        Args:
            note_text: Text to classify
            current_outline: Current book outline
            
        Returns:
            List of alternative classifications with scores
        """
        alternatives = []
        
        # Extract all possible chapter and subtopic targets
        all_targets = []
        
        for part_idx, part in enumerate(current_outline.parts):
            for chapter_idx, chapter in enumerate(part["chapters"]):
                all_targets.append({
                    "type": "chapter",
                    "id": chapter["id"],
                    "title": chapter["title"],
                    "description": chapter.get("description", ""),
                    "path": f"{part['title']} -> {chapter['title']}"
                })
                
                for subtopic_idx, subtopic in enumerate(chapter["subtopics"]):
                    all_targets.append({
                        "type": "subtopic",
                        "id": subtopic["id"],
                        "title": subtopic["title"],
                        "description": subtopic.get("description", ""),
                        "path": f"{part['title']} -> {chapter['title']} -> {subtopic['title']}",
                        "chapter_id": chapter["id"]
                    })
        
        # Calculate similarity to each target
        for target in all_targets:
            target_text = f"{target['title']} {target['description']}"
            try:
                similarity = cosine_similarity(
                    self.vectorizer.fit_transform([note_text, target_text])
                )[0][1]
                
                if similarity > 0.2:  # Only include if similarity is above threshold
                    alternatives.append({
                        "target": target,
                        "similarity": float(similarity),
                        "type": target["type"]
                    })
            except:
                continue
        
        # Sort by similarity and return top 5
        alternatives.sort(key=lambda x: x["similarity"], reverse=True)
        return alternatives[:5]
    
    def calculate_organization_metrics(self, outline: BookOutline) -> OrganizationMetrics:
        """
        Calculates metrics for evaluating the quality of the content organization.

        This method calculates the following metrics:
        - Completeness score: The percentage of subtopics that have content.
        - Flow score: The average similarity between consecutive content items.
        - Cohesion score: The average similarity between all content items in a chapter.

        Args:
            outline: The book outline to analyze.

        Returns:
            An OrganizationMetrics object with the calculated metrics.
        """
        # Calculate completeness score based on content coverage
        total_subtopics = 0
        covered_subtopics = 0
        
        for part in outline.parts:
            for chapter in part["chapters"]:
                for subtopic in chapter["subtopics"]:
                    total_subtopics += 1
                    subtopic_content = self.content_manager.retrieve_content_by_subtopic(subtopic["id"])
                    if subtopic_content:
                        covered_subtopics += 1
        
        completeness_score = covered_subtopics / total_subtopics if total_subtopics > 0 else 0.0
        
        # Calculate flow score based on content sequencing
        flow_score = self._calculate_content_flow_score(outline)
        
        # Calculate cohesion score based on internal content similarity
        cohesion_score = self._calculate_cohesion_score(outline)
        
        # Detect gaps
        gaps = self.detect_content_gaps(outline)
        
        return OrganizationMetrics(
            completeness_score=completeness_score,
            flow_score=flow_score,
            cohesion_score=cohesion_score,
            gap_count=len(gaps),
            suggestions_count=len(self.suggest_content_reorganization(outline))
        )
    
    def _calculate_content_flow_score(self, outline: BookOutline) -> float:
        """
        Calculate content flow score based on sequential relationships and topic progression.
        
        Args:
            outline: The book outline to analyze
            
        Returns:
            Flow score (0.0 to 1.0)
        """
        # This is a simplified implementation - would need more sophisticated analysis
        # in a real implementation
        total_score = 0.0
        score_count = 0
        
        for part in outline.parts:
            for chapter in part["chapters"]:
                # Get content for this chapter
                chapter_content = []
                for subtopic in chapter["subtopics"]:
                    subtopic_content = self.content_manager.retrieve_content_by_subtopic(subtopic["id"])
                    for item in subtopic_content:
                        chapter_content.append(item["content"])
                
                if len(chapter_content) > 1:
                    # Calculate similarity between consecutive content items
                    similarities = cosine_similarity(self.vectorizer.fit_transform(chapter_content))
                    # Average similarity of consecutive items
                    consecutive_similarities = []
                    for i in range(len(similarities) - 1):
                        consecutive_similarities.append(similarities[i][i+1])
                    
                    if consecutive_similarities:
                        avg_consecutive_similarity = sum(consecutive_similarities) / len(consecutive_similarities)
                        total_score += avg_consecutive_similarity
                        score_count += 1
        
        if score_count > 0:
            return total_score / score_count
        else:
            # If no consecutive content to analyze, return a neutral score
            return 0.5
    
    def _calculate_cohesion_score(self, outline: BookOutline) -> float:
        """
        Calculate cohesion score based on internal content similarity.
        
        Args:
            outline: The book outline to analyze
            
        Returns:
            Cohesion score (0.0 to 1.0)
        """
        total_score = 0.0
        score_count = 0
        
        for part in outline.parts:
            for chapter in part["chapters"]:
                # Get content for this chapter
                chapter_content = []
                for subtopic in chapter["subtopics"]:
                    subtopic_content = self.content_manager.retrieve_content_by_subtopic(subtopic["id"])
                    for item in subtopic_content:
                        chapter_content.append(item["content"])
                
                if len(chapter_content) > 1:
                    # Calculate average similarity between all content in the chapter
                    tfidf_matrix = self.vectorizer.fit_transform(chapter_content)
                    similarities = cosine_similarity(tfidf_matrix)
                    
                    # Calculate average similarity (excluding self-similarity)
                    mask = np.ones(similarities.shape, dtype=bool)
                    np.fill_diagonal(mask, 0)
                    avg_similarity = similarities[mask].mean()
                    
                    total_score += avg_similarity
                    score_count += 1
        
        if score_count > 0:
            return total_score / score_count
        else:
            # If no multi-content chapters, return a neutral score
            return 0.5
    
    def get_content_summary(self, 
                           outline: BookOutline, 
                           include_clusters: bool = True) -> Dict:
        """
        Generates a comprehensive summary of the content organization.

        This method provides a summary of the content organization, including:
        - Outline information (number of parts, chapters, and subtopics).
        - Content statistics (total number of content items, content per chapter).
        - Organization metrics (completeness, flow, and cohesion scores).
        - Detected gaps and reorganization suggestions.

        Args:
            outline: The book outline to summarize.
            include_clusters: Whether to include cluster information in the summary.

        Returns:
            A dictionary with the organization summary.
        """
        summary = {
            "outline_info": {
                "title": outline.title,
                "parts_count": len(outline.parts),
                "chapters_count": sum(len(part["chapters"]) for part in outline.parts),
                "subtopics_count": sum(
                    len(chapter["subtopics"]) 
                    for part in outline.parts 
                    for chapter in part["chapters"]
                )
            },
            "content_stats": {
                "total_content_items": 0,
                "content_per_chapter": {},
                "content_per_subtopic": {}
            },
            "organization_metrics": self.calculate_organization_metrics(outline).__dict__,
            "gaps_detected": self.detect_content_gaps(outline),
            "reorganization_suggestions": self.suggest_content_reorganization(outline)
        }
        
        # Add content statistics
        for part in outline.parts:
            for chapter in part["chapters"]:
                chapter_content_count = 0
                for subtopic in chapter["subtopics"]:
                    subtopic_content = self.content_manager.retrieve_content_by_subtopic(subtopic["id"])
                    content_count = len(subtopic_content)
                    summary["content_stats"]["total_content_items"] += content_count
                    summary["content_stats"]["content_per_subtopic"][subtopic["id"]] = content_count
                    chapter_content_count += content_count
                
                summary["content_stats"]["content_per_chapter"][chapter["id"]] = chapter_content_count
        
        # Add cluster information if requested
        if include_clusters:
            summary["clusters"] = [
                {
                    "id": cluster.id,
                    "title": cluster.title,
                    "content_count": len(cluster.content_ids),
                    "topic_keywords": cluster.topic_keywords,
                    "confidence": cluster.confidence
                } for cluster in self.clusters
            ]
        
        return summary


class ContentOrganization:
    """
    A user-friendly interface for organizing book content.

    This class simplifies content organization by providing high-level methods
    to cluster content, get organization suggestions, and visualize the content structure.

    Args:
        project_path (Union[str, Path]): The path to the project directory.
        note_processor (NoteProcessor): A NoteProcessor instance for note embeddings.
        content_manager (ContentManager): A ContentManager instance for content operations.
    """

    def __init__(self, project_path: Union[str, Path],
                 note_processor: NoteProcessor,
                 content_manager: ContentManager):
        self.organizer = ContentOrganizer(project_path, note_processor, content_manager)

    def organize_content(self, outline: BookOutline, content_ids: List[str]) -> Dict:
        """
        Organize the book content by clustering it and providing a summary.

        This method performs the following steps:
        1. Creates content clusters from the provided content IDs.
        2. Generates a summary of the content organization.

        Args:
            outline (BookOutline): The book outline.
            content_ids (List[str]): A list of content IDs to organize.

        Returns:
            A dictionary containing the organization summary.
        """
        self.organizer.create_content_clusters(content_ids)
        return self.organizer.get_content_summary(outline)

    def get_organization_suggestions(self, outline: BookOutline) -> Dict:
        """
        Get suggestions for improving the content organization.

        This method provides a list of suggestions, including:
        - Detected content gaps.
        - Content reorganization suggestions.

        Args:
            outline (BookOutline): The book outline.

        Returns:
            A dictionary containing the organization suggestions.
        """
        return {
            "gaps_detected": self.organizer.detect_content_gaps(outline),
            "reorganization_suggestions": self.organizer.suggest_content_reorganization(outline)
        }

    def visualize_content_structure(self, outline: BookOutline, output_path: Union[str, Path]) -> None:
        """
        Generate a visual representation of the content structure.

        This method creates a graph of the content structure, including:
        - Chapters and subtopics from the outline.
        - Content clusters.
        - Relationships between content items.

        Args:
            outline (BookOutline): The book outline.
            output_path (Union[str, Path]): The path to save the visualization.
        """
        dot = graphviz.Digraph('content_structure', comment='Content Structure')
        dot.attr(rankdir='TB', splines='ortho')

        # Add outline structure
        with dot.subgraph(name='cluster_outline') as c:
            c.attr(label='Outline', style='filled', color='lightgrey')
            for part in outline.parts:
                for chapter in part["chapters"]:
                    c.node(chapter["id"], chapter["title"], shape='box')
                    for subtopic in chapter["subtopics"]:
                        c.node(subtopic["id"], subtopic["title"], shape='ellipse')
                        c.edge(chapter["id"], subtopic["id"])

        # Add content clusters
        with dot.subgraph(name='cluster_clusters') as c:
            c.attr(label='Content Clusters', style='filled', color='lightblue')
            for cluster in self.organizer.clusters:
                c.node(cluster.id, cluster.title, shape='box')
                for content_id in cluster.content_ids:
                    c.node(content_id, f'Content {content_id[:8]}', shape='ellipse')
                    c.edge(cluster.id, content_id)

        # Add relationships
        for rel in self.organizer.relationships:
            dot.edge(rel.source_id, rel.target_id, label=rel.relationship_type, style='dashed')

        output_path = Path(output_path)
        dot.render(output_path, format='png', view=False)
        logger.info(f"Content structure visualization saved to {output_path}.png")
