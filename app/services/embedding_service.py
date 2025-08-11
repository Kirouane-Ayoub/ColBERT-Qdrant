import asyncio
import logging
import time
from typing import List, Optional, Tuple

import numpy as np
from fastembed import LateInteractionTextEmbedding

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str, max_batch_size: int = 32):
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self._model: Optional[LateInteractionTextEmbedding] = None
        self._model_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the embedding model asynchronously."""
        async with self._model_lock:
            if self._model is None:
                logger.info(f"Loading ColBERT model: {self.model_name}")
                start_time = time.time()

                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None, lambda: LateInteractionTextEmbedding(self.model_name)
                )

                load_time = time.time() - start_time
                logger.info(f"Model loaded in {load_time:.2f}s")

    async def embed_documents(
        self, texts: List[str]
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Embed documents and return multivectors with shapes."""
        if not self._model:
            await self.initialize()

        # Process in batches to manage memory
        all_embeddings = []
        all_shapes = []

        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]

            # Run embedding in thread pool
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                None, lambda: list(self._model.embed(batch))
            )

            for embedding in batch_embeddings:
                embedding_array = np.array(embedding)
                all_embeddings.append(embedding_array)
                all_shapes.append(embedding_array.shape)

        return all_embeddings, all_shapes

    async def embed_queries(
        self, texts: List[str]
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Embed queries and return multivectors with shapes."""
        if not self._model:
            await self.initialize()

        all_embeddings = []
        all_shapes = []

        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]

            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                None, lambda: list(self._model.query_embed(batch))
            )

            for embedding in batch_embeddings:
                embedding_array = np.array(embedding)
                all_embeddings.append(embedding_array)
                all_shapes.append(embedding_array.shape)

        return all_embeddings, all_shapes

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model is not None

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "max_batch_size": self.max_batch_size,
            "is_ready": self.is_ready,
        }
