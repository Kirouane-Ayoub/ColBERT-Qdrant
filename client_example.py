import httpx

# Test the Indexing service
print("=== Testing Indexing Service ===")
response = httpx.post(
    "http://localhost:8000/api/v1/index",
    json={
        "collection_name": "my_docs",
        "documents": [
            "ColBERT uses late interaction for better retrieval accuracy",
            "FastEmbed provides easy access to state-of-the-art embedding models",
            "Qdrant vector database supports efficient similarity search",
            "BERT-based models revolutionized natural language processing",
        ],
        "metadata": [
            {
                "source": "blog",
                "year": 2021,
                "category": "ml_research",
                "venue": "Medium",
                "citation_count": 87,
                "authors": ["Liam Carter", "Sophie Wang"],
                "institution": "AI Research Lab",
            },
            {
                "source": "whitepaper",
                "year": 2019,
                "category": "vector_search",
                "venue": "TechConf",
                "citation_count": 312,
                "authors": ["Raj Patel"],
                "institution": "DeepSearch Inc.",
            },
            {
                "source": "report",
                "year": 2022,
                "category": "data_eng",
                "venue": "Internal",
                "citation_count": 14,
                "authors": ["Elena Petrova", "Tomás Silva"],
                "institution": "DataOps Solutions",
            },
            {
                "source": "thesis",
                "year": 2017,
                "category": "nlp",
                "venue": "MIT",
                "citation_count": 456,
                "authors": ["Maria Gonzales"],
                "institution": "MIT CSAIL",
            },
        ],
    },
)

if response.status_code == 200:
    print("Indexing successful")
else:
    print(f"Indexing failed: {response.status_code} - {response.text}")


# Testing the Search service
def test_service_with_debug():
    print("=== Testing Service with Debug Logging ===")
    # Test filter for 'library'
    response = httpx.post(
        "http://localhost:8000/api/v1/search",
        json={
            "collection_name": "my_docs",
            "query": "BERT-based models",
            "limit": 1,
            "score_threshold": 0.1,
            "filter": {"must": [{"key": "source", "match": {"value": "paper"}}]},
        },
    )
    if response.status_code == 200:
        results = response.json()["results"]
        print(f"Service filter for 'paper': {len(results)} results")
        for result in results:
            print(
                f"  - Source: {result['payload']['source']}, Text: {result['text'][:50]}..."
            )
    else:
        print(f"Failed: {response.status_code} - {response.text}")


if __name__ == "__main__":
    test_service_with_debug()
