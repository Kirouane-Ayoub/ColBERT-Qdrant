# debug_filter.py
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv(override=True)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)


def debug_filter_issue():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    collection_name = "my_docs"

    print("=== 1. Check Collection Info ===")
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Collection exists: {collection_name}")
        print(f"Points count: {collection_info.points_count}")
        print(f"Vectors count: {collection_info.vectors_count}")
    except Exception as e:
        print(f"Error getting collection info: {e}")

    print("\n=== 2. Check All Points ===")
    try:
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )

        points, _ = scroll_result
        print(f"Found {len(points)} points:")
        for i, point in enumerate(points):
            print(f"  {i+1}. ID: {point.id}")
            print(f"     Payload: {point.payload}")
    except Exception as e:
        print(f"Error scrolling points: {e}")

    print("\n=== 3. Check Payload Indexes ===")
    try:
        collection_info = client.get_collection(collection_name)
        # Try to access payload indexes if available
        print("Collection config available")
        # The exact way to check indexes varies by client version
    except Exception as e:
        print(f"Error checking indexes: {e}")

    print("\n=== 4. Test Filter Directly ===")
    try:
        # Create a simple filter for source = "paper"
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="source", match=models.MatchValue(value="paper")
                )
            ]
        )

        # Test the filter with scroll (not search)
        scroll_result = client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_condition,
            limit=10,
            with_payload=True,
        )

        points, _ = scroll_result
        print(f"Filter test (source=paper): Found {len(points)} points")
        for point in points:
            print(f"  - ID: {point.id}, Source: {point.payload.get('source')}")

    except Exception as e:
        print(f"Direct filter test failed: {e}")

    print("\n=== 5. Test Filter for Non-existent Value ===")
    try:
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="source", match=models.MatchValue(value="nonexistent")
                )
            ]
        )

        scroll_result = client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_condition,
            limit=10,
            with_payload=True,
        )

        points, _ = scroll_result
        print(f"Filter test (source=nonexistent): Found {len(points)} points")

    except Exception as e:
        print(f"Nonexistent filter test failed: {e}")


if __name__ == "__main__":
    debug_filter_issue()
