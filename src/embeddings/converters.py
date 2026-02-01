"""
Document Converters Module

Handles conversion between PawAid Chunk objects and LangChain Document objects.
"""

from typing import List, Any, Dict

from langchain_core.documents import Document as LCDocument

from src.ingestion.splitter import Chunk


def sanitize_metadata_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize metadata values for ChromaDB compatibility.

    ChromaDB only supports str, int, float, and bool metadata values.
    None values are dropped, lists are converted to comma-separated strings,
    and other complex types are converted to their string representation.

    Args:
        metadata: Original metadata dictionary.

    Returns:
        Sanitized metadata dictionary.
    """
    sanitized = {}

    for key, value in metadata.items():
        if value is None:
            continue
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            # Convert lists to comma-separated strings
            sanitized[key] = ",".join(str(v) for v in value) if value else ""
        else:
            # Convert other types to string
            sanitized[key] = str(value)

    return sanitized


def chunk_to_langchain_document(chunk: Chunk, sanitize: bool = True) -> LCDocument:
    """
    Convert a PawAid Chunk to a LangChain Document.

    Args:
        chunk: PawAid Chunk object.
        sanitize: Whether to sanitize metadata for ChromaDB compatibility.

    Returns:
        LangChain Document with content and metadata.
    """
    metadata = chunk.metadata.copy()

    if sanitize:
        metadata = sanitize_metadata_for_chroma(metadata)

    return LCDocument(
        page_content=chunk.content,
        metadata=metadata
    )


def chunks_to_langchain_documents(
    chunks: List[Chunk],
    sanitize: bool = True
) -> List[LCDocument]:
    """
    Convert a list of PawAid Chunks to LangChain Documents.

    Args:
        chunks: List of PawAid Chunk objects.
        sanitize: Whether to sanitize metadata for ChromaDB compatibility.

    Returns:
        List of LangChain Documents.
    """
    return [chunk_to_langchain_document(chunk, sanitize=sanitize) for chunk in chunks]


def langchain_document_to_chunk(document: LCDocument) -> Chunk:
    """
    Convert a LangChain Document to a PawAid Chunk.

    Args:
        document: LangChain Document object.

    Returns:
        PawAid Chunk with content and metadata.
    """
    return Chunk(
        content=document.page_content,
        metadata=document.metadata.copy() if document.metadata else {}
    )


def langchain_documents_to_chunks(documents: List[LCDocument]) -> List[Chunk]:
    """
    Convert a list of LangChain Documents to PawAid Chunks.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of PawAid Chunks.
    """
    return [langchain_document_to_chunk(doc) for doc in documents]


def generate_document_id(chunk: Chunk, index: int = 0) -> str:
    """
    Generate a deterministic document ID for a chunk.

    The ID is based on the source file and chunk index to ensure
    consistency across rebuilds.

    Args:
        chunk: PawAid Chunk object.
        index: Optional fallback index if chunk_index not in metadata.

    Returns:
        Unique document ID string.
    """
    source = chunk.metadata.get("source", "unknown")
    chunk_index = chunk.metadata.get("chunk_index", index)

    # Sanitize source for use in ID (replace path separators)
    source_safe = source.replace("/", "_").replace("\\", "_")

    return f"{source_safe}_{chunk_index}"


def generate_document_ids(chunks: List[Chunk]) -> List[str]:
    """
    Generate deterministic document IDs for a list of chunks.

    Args:
        chunks: List of PawAid Chunk objects.

    Returns:
        List of unique document ID strings.
    """
    ids = []
    seen = set()

    for i, chunk in enumerate(chunks):
        doc_id = generate_document_id(chunk, i)

        # Handle duplicates by appending a counter
        if doc_id in seen:
            counter = 1
            while f"{doc_id}_{counter}" in seen:
                counter += 1
            doc_id = f"{doc_id}_{counter}"

        seen.add(doc_id)
        ids.append(doc_id)

    return ids
