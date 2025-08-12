import asyncio
import logging
import time
import uuid
from typing import Any, Optional, Union

import numpy as np
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


class QdrantService:
    def __init__(self, url: str, api_key: Optional[str] = None, vector_size: int = 128):
        self.url = url
        self.api_key = api_key
        self.vector_size = vector_size
        self._client: Optional[QdrantClient] = None
        self._client_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize Qdrant client."""
        async with self._client_lock:
            if self._client is None:
                logger.info(f"Connecting to Qdrant at {self.url}")

                loop = asyncio.get_event_loop()
                self._client = await loop.run_in_executor(
                    None, lambda: QdrantClient(url=self.url, api_key=self.api_key)
                )

                try:
                    await loop.run_in_executor(None, self._client.get_collections)
                    logger.info("Successfully connected to Qdrant")
                except Exception as e:
                    logger.error(f"Failed to connect to Qdrant: {e}")
                    raise

    async def create_collection(
        self, collection_name: str, create_indexes: bool = True
    ) -> bool:
        """Create a new collection for ColBERT multivectors with payload indexes."""
        if not self._client:
            await self.initialize()

        try:
            loop = asyncio.get_event_loop()

            # Check if collection exists
            collections = await loop.run_in_executor(None, self._client.get_collections)
            existing_names = [col.name for col in collections.collections]

            if collection_name in existing_names:
                logger.info(f"Collection '{collection_name}' already exists")
                if create_indexes:
                    await self._create_payload_indexes(collection_name)
                return True

            # Create collection with multivector configuration
            await loop.run_in_executor(
                None,
                lambda: self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                        quantization_config=models.BinaryQuantization(
                            binary=models.BinaryQuantizationConfig(always_ram=True),
                        ),
                    ),
                ),
            )

            logger.info(f"Created collection '{collection_name}'")

            if create_indexes:
                await self._create_payload_indexes(collection_name)

            return True

        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            return False

    async def _create_payload_indexes(self, collection_name: str):
        """Create payload indexes for common filterable fields."""
        loop = asyncio.get_event_loop()

        # Common fields that users might want to filter on
        index_fields = [
            ("source", models.PayloadSchemaType.KEYWORD),
            ("year", models.PayloadSchemaType.INTEGER),
            ("category", models.PayloadSchemaType.KEYWORD),
            ("venue", models.PayloadSchemaType.KEYWORD),
            ("citation_count", models.PayloadSchemaType.INTEGER),
            ("institution", models.PayloadSchemaType.KEYWORD),
            ("authors", models.PayloadSchemaType.KEYWORD),
        ]

        for field_name, field_type in index_fields:
            try:
                await loop.run_in_executor(
                    None,
                    lambda fn=field_name, ft=field_type: self._client.create_payload_index(
                        collection_name=collection_name, field_name=fn, field_schema=ft
                    ),
                )
                logger.info(
                    f"Created index for field '{field_name}' in collection '{collection_name}'"
                )
            except Exception as e:
                logger.debug(
                    f"Index creation for '{field_name}' failed (might already exist): {e}"
                )

    def _prepare_multivector(self, embedding: np.ndarray) -> list[list[float]]:
        """Convert numpy multivector to the format expected by Qdrant."""
        if embedding.ndim == 1:
            return [embedding.astype(float).tolist()]
        elif embedding.ndim == 2:
            return embedding.astype(float).tolist()
        else:
            raise ValueError(f"Unsupported embedding dimension: {embedding.ndim}")

    def _convert_filter_dict_to_models(
        self, filter_input: Union[dict[str, Any], Any]
    ) -> models.Filter:
        """Convert dictionary or Pydantic filter to Qdrant Filter models."""

        # Convert Pydantic objects to dictionary first
        if hasattr(filter_input, "dict"):
            # It's a Pydantic model
            filter_dict = filter_input.dict(exclude_unset=True)
            logger.info(f"Converted Pydantic to dict: {filter_dict}")
        elif hasattr(filter_input, "__dict__"):
            # It's some other object with attributes
            filter_dict = {}
            for attr in ["must", "should", "must_not"]:
                value = getattr(filter_input, attr, None)
                if value is not None:
                    if hasattr(value, "__iter__") and not isinstance(value, str):
                        # Convert list of Pydantic objects to dicts
                        filter_dict[attr] = [
                            item.dict(exclude_unset=True)
                            if hasattr(item, "dict")
                            else item
                            for item in value
                        ]
                    else:
                        filter_dict[attr] = value
            logger.info(f"Converted object to dict: {filter_dict}")
        else:
            # It's already a dictionary
            filter_dict = filter_input
            logger.info(f"Using dict as-is: {filter_dict}")

        def convert_condition(condition: dict[str, Any]) -> models.FieldCondition:
            key = condition["key"]
            logger.info(f"Converting condition for key: {key}")

            if "match" in condition:
                value = condition["match"]["value"]
                logger.info(f"Creating match condition: {key} = {value}")
                return models.FieldCondition(
                    key=key, match=models.MatchValue(value=value)
                )
            elif "range" in condition:
                range_params = condition["range"]
                range_kwargs = {}

                for param in ["gte", "gt", "lte", "lt"]:
                    if param in range_params:
                        range_kwargs[param] = range_params[param]

                return models.FieldCondition(
                    key=key, range=models.Range(**range_kwargs)
                )
            else:
                raise ValueError(f"Unsupported condition: {condition}")

        # Convert conditions
        must_conditions = []
        should_conditions = []
        must_not_conditions = []

        if "must" in filter_dict and filter_dict["must"]:
            logger.info(f"Processing 'must' conditions: {filter_dict['must']}")
            for i, condition in enumerate(filter_dict["must"]):
                logger.info(f"Processing must condition {i}: {condition}")

                if "should" in condition:
                    # Handle nested should inside must
                    logger.info("Found nested 'should' in 'must'")
                    nested_should = [convert_condition(c) for c in condition["should"]]
                    nested_filter = models.Filter(should=nested_should)
                    must_conditions.append(nested_filter)
                else:
                    # Regular condition
                    converted = convert_condition(condition)
                    logger.info(f"Converted condition {i}: {converted}")
                    must_conditions.append(converted)

        if "should" in filter_dict and filter_dict["should"]:
            logger.info(f"Processing 'should' conditions: {filter_dict['should']}")
            should_conditions = [convert_condition(c) for c in filter_dict["should"]]

        if "must_not" in filter_dict and filter_dict["must_not"]:
            logger.info(f"Processing 'must_not' conditions: {filter_dict['must_not']}")
            must_not_conditions = [
                convert_condition(c) for c in filter_dict["must_not"]
            ]

        # Create the final filter
        final_filter = models.Filter(
            must=must_conditions if must_conditions else None,
            should=should_conditions if should_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None,
        )

        logger.info(f"Final filter object: {final_filter}")
        return final_filter

    async def index_documents(
        self,
        collection_name: str,
        embeddings: list[np.ndarray],
        texts: list[str],
        metadata: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ) -> tuple[bool, int]:
        """Index documents with their multivector embeddings."""
        if not self._client:
            await self.initialize()

        try:
            # Ensure collection exists
            await self.create_collection(collection_name)

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]

            # Prepare points
            points = []
            for i, (embedding, text, doc_id) in enumerate(zip(embeddings, texts, ids)):
                payload = {"text": text}
                if metadata and i < len(metadata):
                    payload.update(metadata[i])

                try:
                    vector_data = self._prepare_multivector(embedding)

                    points.append(
                        models.PointStruct(
                            id=doc_id, vector=vector_data, payload=payload
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to prepare vector for document {i}: {e}")
                    continue

            if not points:
                logger.error("No valid points to index")
                return False, 0

            # Upload in batches
            batch_size = 100
            total_indexed = 0

            loop = asyncio.get_event_loop()

            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                try:
                    await loop.run_in_executor(
                        None,
                        lambda b=batch: self._client.upsert(
                            collection_name=collection_name, points=b
                        ),
                    )
                    total_indexed += len(batch)
                    logger.info(
                        f"Indexed batch {i//batch_size + 1}, total: {total_indexed}"
                    )
                except Exception as e:
                    logger.error(f"Failed to index batch {i//batch_size + 1}: {e}")
                    continue

            logger.info(
                f"Successfully indexed {total_indexed} documents in '{collection_name}'"
            )
            return True, total_indexed

        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return False, 0

    async def search(
        self,
        collection_name: str,
        query_embedding: np.ndarray,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> tuple[list[dict[str, Any]], float]:
        """Search for similar documents using ColBERT query embedding with query_points."""
        if not self._client:
            await self.initialize()

        start_time = time.time()

        try:
            query_vector = query_embedding.tolist()

            logger.info(f"Query vector shape: {query_embedding.shape}")
            logger.info(f"Raw filter_conditions: {filter_conditions}")

            # Convert filter
            query_filter = None
            if filter_conditions:
                query_filter = self._convert_filter_dict_to_models(filter_conditions)
                logger.info(f"Converted filter object: {query_filter}")
                logger.info(f"Filter type: {type(query_filter)}")

                # Debug the filter components
                if hasattr(query_filter, "must") and query_filter.must:
                    for i, condition in enumerate(query_filter.must):
                        logger.info(f"Must condition {i}: {condition}")
                        logger.info(f"Condition type: {type(condition)}")

            loop = asyncio.get_event_loop()

            # Debug the actual call
            logger.info("Calling query_points with:")
            logger.info(f"  collection_name: {collection_name}")
            logger.info(f"  query vector length: {len(query_vector)}")
            logger.info(f"  limit: {limit}")
            logger.info(f"  score_threshold: {score_threshold}")
            logger.info(f"  query_filter: {query_filter}")

            search_result = await loop.run_in_executor(
                None,
                lambda: self._client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=query_filter,  # This is correct
                    with_payload=True,
                ),
            )

            search_time = (time.time() - start_time) * 1000

            # Format results
            results = []
            for hit in search_result.points:
                logger.info(
                    f"Result: ID={hit.id}, Score={hit.score}, Payload={hit.payload}"
                )
                results.append(
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "payload": hit.payload or {},
                        "text": hit.payload.get("text") if hit.payload else None,
                    }
                )

            logger.info(
                f"Search completed in {search_time:.2f}ms, found {len(results)} results"
            )
            return results, search_time

        except Exception as e:
            logger.error(f"Search failed: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            search_time = (time.time() - start_time) * 1000
            return [], search_time

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        if not self._client:
            try:
                await self.initialize()
            except Exception:
                return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._client.get_collections)
            return True
        except Exception:
            return False

    @property
    def is_ready(self) -> bool:
        """Check if client is ready."""
        return self._client is not None
