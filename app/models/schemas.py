from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class EmbeddingType(str, Enum):
    DOCUMENT = "document"
    QUERY = "query"


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    embedding_type: EmbeddingType = EmbeddingType.DOCUMENT
    normalize: bool = False

    @validator("texts")
    def validate_texts(cls, v):
        if not all(isinstance(text, str) and text.strip() for text in v):
            raise ValueError("All texts must be non-empty strings")
        return [text.strip() for text in v]


class EmbedResponse(BaseModel):
    embeddings: List[List[List[float]]]  # List of multivectors
    model_name: str
    embedding_type: str
    shapes: List[List[int]]  # Shape of each multivector [num_tokens, dim]


class IndexRequest(BaseModel):
    collection_name: str = Field(..., min_length=1)
    documents: List[str] = Field(..., min_items=1, max_items=1000)
    metadata: Optional[List[Dict[str, Any]]] = None
    ids: Optional[List[str]] = None


class IndexResponse(BaseModel):
    collection_name: str
    indexed_count: int
    operation_id: Optional[str] = None


class SearchResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any]
    text: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    search_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    qdrant_connected: bool
    version: str
    uptime_seconds: float


class FilterCondition(BaseModel):
    key: str
    match: Optional[Dict[str, Any]] = None
    range: Optional[Dict[str, Any]] = None
    geo_radius: Optional[Dict[str, Any]] = None


class SearchFilter(BaseModel):
    must: Optional[List[Union[FilterCondition, Dict[str, Any]]]] = None
    should: Optional[List[FilterCondition]] = None
    must_not: Optional[List[FilterCondition]] = None


class SearchRequest(BaseModel):
    collection_name: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    filter: Optional[Dict[str, Any]] = None
