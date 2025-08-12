from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    collection_name: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    filter: Optional[dict[str, Any]] = None


# Initialize the MCP server
mcp = FastMCP("Search Service")


@mcp.tool()
async def search(
    query: str,
    collection_name: str = "my_docs",
    limit: int = 10,
    score_threshold: Optional[float] = None,
    filter_json: Optional[str] = None,
    base_url: str = "http://localhost:8000",
) -> dict[str, Any]:
    """
    Search a collection with optional filtering.
    Args:
        query: Search query text
        collection_name: Name of the collection to search
        limit: Maximum number of results to return (1-100)
        score_threshold: Minimum score threshold for results (0.0-1.0)
        filter_json: JSON string for filtering with the following structure:
            {
                "must": [FilterCondition, ...],     // All conditions must match (AND logic)
                "should": [FilterCondition, ...],   // At least one condition should match (OR logic)
                "must_not": [FilterCondition, ...]  // None of these conditions should match (NOT logic)
            }

            FilterCondition structure:
            {
                "key": "field_name",                // Required: field to filter on
                "match": {"value": "exact_value"},  // Exact match filter
                "range": {                          // Range filter (for numbers, dates)
                    "gte": value,                   // Greater than or equal
                    "lte": value,                   // Less than or equal
                    "gt": value,                    // Greater than
                    "lt": value                     // Less than
                },
                "geo_radius": {                     // Geo-spatial radius filter
                    "center": {"lat": 0.0, "lon": 0.0},
                    "radius": 1000                  // in meters
                }
            }

            Examples:
            - Simple match: '{"must": [{"key": "source", "match": {"value": "blog"}}]}'
            - Multiple conditions: '{"must": [{"key": "status", "match": {"value": "published"}}, {"key": "category", "match": {"value": "ai"}}]}'
            - Complex logic: '{"must": [{"key": "status", "match": {"value": "active"}}], "should": [{"key": "priority", "match": {"value": "high"}}, {"key": "urgent", "match": {"value": "true"}}], "must_not": [{"key": "archived", "match": {"value": "true"}}]}'
            - Range filter: '{"must": [{"key": "date", "range": {"gte": "2023-01-01", "lte": "2024-12-31"}}]}'
            - Mixed filters: '{"must": [{"key": "status", "match": {"value": "published"}}], "should": [{"key": "score", "range": {"gte": 0.8}}]}'

        base_url: Base URL of the search service

    Returns:
        dictionary containing search results and metadata:
        {
            "status": "success|error",
            "results_count": int,
            "results": [
                {
                    "text": "document text",
                    "score": float,
                    "payload": {"metadata": "fields"}
                }
            ],
            "query_info": {
                "query": "search terms",
                "collection": "collection_name",
                "limit": int,
                "score_threshold": float,
                "filter_applied": boolean
            }
        }
    """
    try:
        import json

        # Parse filter if provided
        filter_dict = None
        if filter_json:
            filter_dict = json.loads(filter_json)

        # Create search request
        search_request = SearchRequest(
            collection_name=collection_name,
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            filter=filter_dict,
        )

        # Make the API request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/v1/search",
                json=search_request.model_dump(exclude_none=True),
                timeout=30.0,
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                formatted_results = []
                for result in results:
                    formatted_results.append(
                        {
                            "text": result.get("text", ""),
                            "score": result.get("score"),
                            "payload": result.get("payload", {}),
                        }
                    )

                return {
                    "status": "success",
                    "results_count": len(results),
                    "results": formatted_results,
                    "query_info": {
                        "query": query,
                        "collection": collection_name,
                        "limit": limit,
                        "score_threshold": score_threshold,
                        "filter_applied": filter_dict is not None,
                    },
                }
            else:
                return {
                    "status": "error",
                    "status_code": response.status_code,
                    "error_message": response.text,
                }
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error_type": "InvalidJSON",
            "error_message": "Invalid filter JSON format",
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
        }


if __name__ == "__main__":
    mcp.run(transport="sse")
