import asyncio
import logging
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from qdrant_client.models import FilterSelector, Filter
from app.models.schemas import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

router = APIRouter()

# Global services (will be initialized in main.py)
embedding_service: EmbeddingService = None
qdrant_service: QdrantService = None


def get_embedding_service() -> EmbeddingService:
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")
    return embedding_service


def get_qdrant_service() -> QdrantService:
    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Qdrant service not initialized")
    return qdrant_service


@router.post("/embed", response_model=EmbedResponse)
async def embed_texts(
    request: EmbedRequest,
    embedding_svc: EmbeddingService = Depends(get_embedding_service),
):
    """Generate ColBERT multivector embeddings for texts."""
    try:
        start_time = time.time()

        if request.embedding_type == "document":
            embeddings, shapes = await embedding_svc.embed_documents(request.texts)
        else:  # query
            embeddings, shapes = await embedding_svc.embed_queries(request.texts)

        # Convert numpy arrays to lists
        embedding_lists = [emb.tolist() for emb in embeddings]

        processing_time = time.time() - start_time
        logger.info(f"Embedded {len(request.texts)} texts in {processing_time:.3f}s")

        return EmbedResponse(
            embeddings=embedding_lists,
            model_name=embedding_svc.model_name,
            embedding_type=request.embedding_type.value,
            shapes=[list(shape) for shape in shapes],
        )

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Embedding failed: {str(e)}"
        ) from e


@router.post("/index", response_model=IndexResponse)
async def index_documents(
    request: IndexRequest,
    embedding_svc: EmbeddingService = Depends(get_embedding_service),
    qdrant_svc: QdrantService = Depends(get_qdrant_service),
    background_tasks: BackgroundTasks = None,
):
    """Index documents into Qdrant with ColBERT embeddings."""
    try:
        # Generate embeddings
        embeddings, _ = await embedding_svc.embed_documents(request.documents)

        # Index into Qdrant
        success, indexed_count = await qdrant_svc.index_documents(
            collection_name=request.collection_name,
            embeddings=embeddings,
            texts=request.documents,
            metadata=request.metadata,
            ids=request.ids,
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to index documents")

        return IndexResponse(
            collection_name=request.collection_name, indexed_count=indexed_count
        )

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}") from e


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    embedding_svc: EmbeddingService = Depends(get_embedding_service),
    qdrant_svc: QdrantService = Depends(get_qdrant_service),
):
    """Search for similar documents using ColBERT query embedding."""
    try:
        # Generate query embedding
        query_embeddings, _ = await embedding_svc.embed_queries([request.query])
        query_embedding = query_embeddings[0]

        # Search in Qdrant
        results, search_time = await qdrant_svc.search(
            collection_name=request.collection_name,
            query_embedding=query_embedding,
            limit=request.limit,
            score_threshold=request.score_threshold,
            filter_conditions=request.filter,
        )

        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            search_time_ms=search_time,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


@router.get("/health", response_model=HealthResponse)
async def health_check(
    embedding_svc: EmbeddingService = Depends(get_embedding_service),
    qdrant_svc: QdrantService = Depends(get_qdrant_service),
):
    """Health check endpoint."""
    start_time = time.time()

    model_loaded = embedding_svc.is_ready
    qdrant_connected = await qdrant_svc.health_check()

    status = "healthy" if (model_loaded and qdrant_connected) else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        qdrant_connected=qdrant_connected,
        version="1.0.0",
        uptime_seconds=time.time() - start_time,
    )


@router.post("/create-index")
async def create_payload_index(
    collection_name: str,
    field_name: str,
    field_type: str = "keyword",
    qdrant_svc: QdrantService = Depends(get_qdrant_service),
):
    """Create a payload index for filtering."""
    success = await qdrant_svc.create_custom_index(
        collection_name, field_name, field_type
    )

    if success:
        return {
            "message": f"Index created for field '{field_name}' in collection '{collection_name}'"
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to create index")


@router.delete("/collections/{collection_name}")
async def delete_collection(
    collection_name: str, qdrant_svc: QdrantService = Depends(get_qdrant_service)
):
    """Delete an entire collection."""
    if not qdrant_svc._client:
        await qdrant_svc.initialize()

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: qdrant_svc._client.delete_collection(
                collection_name=collection_name
            ),
        )
        return {"message": f"Collection '{collection_name}' deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete collection '{collection_name}': {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete collection: {str(e)}"
        ) from e


@router.delete("/collections/{collection_name}/points")
async def clear_collection_points(
    collection_name: str, qdrant_svc: QdrantService = Depends(get_qdrant_service)
):
    """Clear all points from a collection (keep collection structure)."""
    if not qdrant_svc._client:
        await qdrant_svc.initialize()

    try:

        loop = asyncio.get_event_loop()

        # Method 1: Use FilterSelector with empty filter (deletes all)
        result = await loop.run_in_executor(
            None,
            lambda: qdrant_svc._client.delete(
                collection_name=collection_name,
                points_selector=FilterSelector(
                    filter=Filter()  # Empty filter matches all points
                ),
            ),
        )

        return {
            "message": f"All points cleared from collection '{collection_name}'",
            "operation_id": result.operation_id
            if hasattr(result, "operation_id")
            else None,
            "status": result.status if hasattr(result, "status") else "completed",
        }

    except Exception as e:
        logger.error(f"Failed to clear points from collection '{collection_name}': {e}")

        # Fallback method: Get all point IDs and delete them
        try:
            logger.info("Trying fallback method: scroll and delete by IDs")

            # Get all point IDs
            scroll_result = await loop.run_in_executor(
                None,
                lambda: qdrant_svc._client.scroll(
                    collection_name=collection_name,
                    limit=10000,  # Adjust based on your collection size
                    with_payload=False,
                    with_vectors=False,
                ),
            )

            points, next_page_offset = scroll_result
            point_ids = [point.id for point in points]

            if point_ids:
                # Delete by IDs
                from qdrant_client.models import PointIdsList

                result = await loop.run_in_executor(
                    None,
                    lambda: qdrant_svc._client.delete(
                        collection_name=collection_name,
                        points_selector=PointIdsList(points=point_ids),
                    ),
                )

                return {
                    "message": f"Deleted {len(point_ids)} points from collection '{collection_name}'",
                    "deleted_count": len(point_ids),
                    "operation_id": result.operation_id
                    if hasattr(result, "operation_id")
                    else None,
                }
            else:
                return {"message": f"Collection '{collection_name}' is already empty"}

        except Exception as fallback_error:
            logger.error(f"Fallback method also failed: {fallback_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to clear points: {str(e)}. Fallback also failed: {str(fallback_error)}",
            ) from e


@router.get("/collections")
async def list_collections(qdrant_svc: QdrantService = Depends(get_qdrant_service)):
    """List all collections."""
    if not qdrant_svc._client:
        await qdrant_svc.initialize()

    try:
        import asyncio

        loop = asyncio.get_event_loop()
        collections = await loop.run_in_executor(
            None, qdrant_svc._client.get_collections
        )

        # Get collection info with proper attributes
        collection_info = []
        for col in collections.collections:
            # Get detailed collection info
            try:
                collection_details = await loop.run_in_executor(
                    None, lambda c=col.name: qdrant_svc._client.get_collection(c)
                )

                collection_info.append(
                    {
                        "name": col.name,
                        "vectors_count": getattr(
                            collection_details, "vectors_count", 0
                        ),
                        "points_count": getattr(collection_details, "points_count", 0),
                        "status": getattr(collection_details, "status", "unknown"),
                        "config": {
                            "vector_size": collection_details.config.params.vectors.size
                            if hasattr(collection_details.config.params, "vectors")
                            else "unknown"
                        },
                    }
                )
            except Exception as e:
                # Fallback with basic info
                collection_info.append(
                    {
                        "name": col.name,
                        "vectors_count": 0,
                        "points_count": 0,
                        "status": "unknown",
                        "error": str(e),
                    }
                )

        return {"collections": collection_info}

    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list collections: {str(e)}"
        ) from e
