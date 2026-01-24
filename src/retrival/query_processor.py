"""
Query Processor Module

Handles query cleaning, expansion, and analysis for improved retrieval.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Set

from .config import QueryConfig


logger = logging.getLogger(__name__)


# Common veterinary abbreviations and their expansions
VETERINARY_ABBREVIATIONS = {
    "gi": "gastrointestinal",
    "uti": "urinary tract infection",
    "uri": "upper respiratory infection",
    "ccl": "cranial cruciate ligament",
    "acl": "anterior cruciate ligament",
    "ivdd": "intervertebral disc disease",
    "chf": "congestive heart failure",
    "ckd": "chronic kidney disease",
    "dm": "diabetes mellitus",
    "hbc": "hit by car",
    "doa": "dead on arrival",
    "bw": "body weight",
    "bid": "twice daily",
    "tid": "three times daily",
    "sid": "once daily",
    "prn": "as needed",
    "po": "by mouth",
    "iv": "intravenous",
    "im": "intramuscular",
    "sq": "subcutaneous",
    "cpr": "cardiopulmonary resuscitation",
}

# Synonyms for common veterinary terms
VETERINARY_SYNONYMS = {
    "vomit": ["vomiting", "throwing up", "puking", "emesis"],
    "diarrhea": ["loose stool", "runny stool", "watery stool"],
    "poison": ["toxic", "poisoning", "toxicity", "toxin"],
    "chocolate": ["cocoa", "cacao"],
    "breathing": ["respiration", "respiratory", "breath"],
    "bleeding": ["hemorrhage", "blood loss", "hemorrhaging"],
    "seizure": ["convulsion", "fit", "epileptic"],
    "limp": ["limping", "lameness", "lame"],
    "swollen": ["swelling", "edema", "inflammation"],
    "pain": ["painful", "hurting", "discomfort"],
    "eat": ["eating", "ate", "ingested", "swallowed", "consumed"],
    "not eating": ["anorexia", "loss of appetite", "won't eat", "refusing food"],
    "tired": ["lethargy", "lethargic", "weakness", "weak"],
}

# Emergency keywords that indicate urgency
EMERGENCY_KEYWORDS = {
    "emergency", "urgent", "immediately", "dying", "unconscious",
    "not breathing", "choking", "seizure", "convulsion", "poisoned",
    "bleeding heavily", "hit by car", "trauma", "collapse", "collapsed",
    "unresponsive", "can't breathe", "blue gums", "pale gums",
    "severe", "sudden", "rapidly", "critical"
}

# Animal-related keywords
DOG_KEYWORDS = {"dog", "dogs", "puppy", "puppies", "canine", "pup"}
CAT_KEYWORDS = {"cat", "cats", "kitten", "kittens", "feline", "kitty"}


@dataclass
class ProcessedQuery:
    """Result of query processing."""

    original_query: str
    cleaned_query: str
    expanded_query: str
    detected_animal: Optional[str] = None  # "dog", "cat", or None
    is_emergency: bool = False
    keywords: List[str] = field(default_factory=list)


class QueryProcessor:
    """
    Processes user queries for improved retrieval.

    Handles:
    - Text cleaning and normalization
    - Abbreviation expansion
    - Synonym expansion
    - Animal type detection
    - Emergency/urgency detection
    """

    def __init__(self, config: Optional[QueryConfig] = None):
        """
        Initialize the query processor.

        Args:
            config: Query processing configuration. Uses defaults if not provided.
        """
        self.config = config or QueryConfig()
        logger.debug("QueryProcessor initialized")

    def process(self, query: str) -> ProcessedQuery:
        """
        Process a user query for retrieval.

        Args:
            query: Raw user query string.

        Returns:
            ProcessedQuery with cleaned, expanded query and metadata.
        """
        if not query or not query.strip():
            logger.warning("Empty query received")
            return ProcessedQuery(
                original_query=query,
                cleaned_query="",
                expanded_query="",
            )

        logger.debug(f"Processing query: '{query[:50]}...'")

        # Step 1: Clean the query
        cleaned = self._clean_query(query)

        # Step 2: Detect animal type
        detected_animal = None
        if self.config.detect_animal_type:
            detected_animal = self._detect_animal(cleaned)
            if detected_animal is None and self.config.default_animal_type:
                detected_animal = self.config.default_animal_type

        # Step 3: Detect emergency
        is_emergency = False
        if self.config.detect_urgency:
            is_emergency = self._detect_emergency(cleaned)

        # Step 4: Expand the query
        expanded = self._expand_query(cleaned)

        # Step 5: Extract keywords
        keywords = self._extract_keywords(cleaned)

        result = ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned,
            expanded_query=expanded,
            detected_animal=detected_animal,
            is_emergency=is_emergency,
            keywords=keywords,
        )

        logger.info(
            f"Query processed: animal={detected_animal}, "
            f"emergency={is_emergency}, keywords={len(keywords)}"
        )

        return result

    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query text."""
        cleaned = query.strip()

        if self.config.normalize_whitespace:
            # Replace multiple spaces/newlines with single space
            cleaned = re.sub(r"\s+", " ", cleaned)

        if self.config.lowercase:
            cleaned = cleaned.lower()

        if self.config.remove_punctuation:
            # Keep only alphanumeric and spaces
            cleaned = re.sub(r"[^\w\s]", "", cleaned)

        return cleaned

    def _detect_animal(self, query: str) -> Optional[str]:
        """Detect if the query mentions a specific animal type."""
        query_lower = query.lower()
        words = set(query_lower.split())

        has_dog = bool(words & DOG_KEYWORDS)
        has_cat = bool(words & CAT_KEYWORDS)

        if has_dog and not has_cat:
            return "dog"
        elif has_cat and not has_dog:
            return "cat"
        elif has_dog and has_cat:
            return None  # Both mentioned
        else:
            return None  # Neither mentioned

    def _detect_emergency(self, query: str) -> bool:
        """Detect if the query indicates an emergency."""
        query_lower = query.lower()

        # Check for emergency keywords
        for keyword in EMERGENCY_KEYWORDS:
            if keyword in query_lower:
                logger.debug(f"Emergency keyword detected: '{keyword}'")
                return True

        return False

    def _expand_query(self, query: str) -> str:
        """Expand the query with synonyms and abbreviation expansions."""
        expanded_parts = [query]

        if self.config.expand_abbreviations:
            expanded_parts.extend(self._expand_abbreviations(query))

        if self.config.expand_synonyms:
            expanded_parts.extend(self._expand_synonyms(query))

        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique_parts = []
        for part in expanded_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        return " ".join(unique_parts)

    def _expand_abbreviations(self, query: str) -> List[str]:
        """Expand medical abbreviations in the query."""
        expansions = []
        query_lower = query.lower()

        for abbrev, expansion in VETERINARY_ABBREVIATIONS.items():
            # Match whole word only
            pattern = rf"\b{re.escape(abbrev)}\b"
            if re.search(pattern, query_lower, re.IGNORECASE):
                expansions.append(expansion)

        return expansions

    def _expand_synonyms(self, query: str) -> List[str]:
        """Add synonyms for terms in the query."""
        expansions = []
        query_lower = query.lower()

        for term, synonyms in VETERINARY_SYNONYMS.items():
            if term in query_lower:
                expansions.extend(synonyms)

        return expansions

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract significant keywords from the query."""
        # Remove common stop words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "and", "but", "if", "or", "because", "until", "while",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "am", "it", "its", "i", "me", "my", "myself", "we", "our",
            "ours", "ourselves", "you", "your", "yours", "yourself", "he",
            "him", "his", "himself", "she", "her", "hers", "herself",
            "they", "them", "their", "theirs", "themselves"
        }

        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords


def create_metadata_filter(
    processed_query: ProcessedQuery,
    filter_by_animal: bool = True
) -> Optional[dict]:
    """
    Create a ChromaDB metadata filter based on processed query.

    Args:
        processed_query: The processed query object.
        filter_by_animal: Whether to filter by detected animal type.

    Returns:
        ChromaDB filter dictionary or None if no filter needed.
    """
    conditions = []

    if filter_by_animal and processed_query.detected_animal:
        animal = processed_query.detected_animal
        # Match exact animal or "both" (documents relevant to both species)
        conditions.append({
            "$or": [
                {"animal_types": {"$eq": animal}},
                {"animal_types": {"$eq": f"{animal}, dog" if animal == "cat" else f"dog, {animal}"}},
                {"animal_types": {"$eq": "dog, cat"}},
                {"animal_types": {"$eq": "cat, dog"}},
            ]
        })

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}
