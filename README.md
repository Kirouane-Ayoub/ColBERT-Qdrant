# ColBERT-Qdrant Embedding Service

A FastAPI-based microservice for generating [ColBERT](https://github.com/stanford-futuredata/ColBERT) multivector embeddings and indexing/searching them in [Qdrant](https://qdrant.tech/) vector database. Supports advanced filtering, payload indexing, and rich metadata for semantic search applications.

## Features

- REST API for embedding, indexing, and searching documents using ColBERT multivectors
- Qdrant integration with support for payload filtering and custom indexes
- Batch processing, async endpoints, and health checks
- Dockerized for easy deployment


## Quickstart

### 1. Clone and Configure

```sh
git clone git@github.com:Kirouane-Ayoub/ColBERT-Qdrant.git
cd colbert-qdrant
cp app/.env.example app/.env
# Edit app/.env with your Qdrant URL and API key
```

### 2. Build and Run with Docker

```sh
docker-compose up --build
```

- The API will be available at [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. API Usage

- **/api/v1/embed**: Generate embeddings for texts
- **/api/v1/index**: Index documents with metadata
- **/api/v1/search**: Search for similar documents with optional filters
- **/api/v1/health**: Health check endpoint

See [app/models/schemas.py](app/models/schemas.py) for request/response formats.

### Example: Index Documents

```python
import httpx

response = httpx.post("http://localhost:8000/api/v1/index", json={
    "collection_name": "my_docs",
    "documents": [
        "ColBERT uses late interaction for better retrieval accuracy",
        "FastEmbed provides easy access to state-of-the-art embedding models",
        "Qdrant vector database supports efficient similarity search",
        "BERT-based models revolutionized natural language processing"
    ],
    "metadata": [
        {
            "source": "blog",
            "year": 2021,
            "category": "ml_research",
            "venue": "Medium",
            "citation_count": 87,
            "authors": ["Liam Carter", "Sophie Wang"],
            "institution": "AI Research Lab"
        },
        {
            "source": "whitepaper",
            "year": 2019,
            "category": "vector_search",
            "venue": "TechConf",
            "citation_count": 312,
            "authors": ["Raj Patel"],
            "institution": "DeepSearch Inc."
        },
        {
            "source": "report",
            "year": 2022,
            "category": "data_eng",
            "venue": "Internal",
            "citation_count": 14,
            "authors": ["Elena Petrova", "Tomás Silva"],
            "institution": "DataOps Solutions"
        },
        {
            "source": "thesis",
            "year": 2017,
            "category": "nlp",
            "venue": "MIT",
            "citation_count": 456,
            "authors": ["Maria Gonzales"],
            "institution": "MIT CSAIL"
        }
    ]
})

if response.status_code == 200:
    print("Indexing successful")
else:
    print(f"Indexing failed: {response.status_code} - {response.text}")
```

### Example: Search with Filter

```python
def test_service_with_debug():
    print("=== Testing Service with Debug Logging ===")

    # Test filter for 'library'
    response = httpx.post("http://localhost:8000/api/v1/search", json={
        "collection_name": "my_docs",
        "query": "BERT-based models",
        "limit": 1,
        "score_threshold": 0.1,
        "filter": {
            "must": [
                {
                    "key": "source",
                    "match": {
                        "value": "paper"
                    }
                }
            ]
        }
    })

    if response.status_code == 200:
        results = response.json()["results"]
        print(f"Service filter for 'paper': {len(results)} results")
        for result in results:
            print(f"  - Source: {result['payload']['source']}, Text: {result['text'][:50]}...")
    else:
        print(f"Failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_service_with_debug()
```
### 4. Development

- Install dependencies: `pip install -r requirements.txt`
- Run locally: `uvicorn app.main:app --reload`

## Configuration

Edit [app/.env](app/.env) for Qdrant connection and model settings:

```
QDRANT_URL = "https://your-qdrant-url:6333"
QDRANT_API_KEY = "your-qdrant-api-key"
```

See [app/config.py](app/config.py) for all settings.

## Collection Management Endpoints

The API provides endpoints to manage your Qdrant collections:

- **List all collections**
  - `GET /collections`
  - Returns metadata about all collections in Qdrant.

- **Create a payload index**
  - `POST /create-index`
  - Body parameters: `collection_name`, `field_name`, `field_type` (default: `"keyword"`)
  - Example:
    ```json
    {
      "collection_name": "my_docs",
      "field_name": "source",
      "field_type": "keyword"
    }
    ```

- **Delete an entire collection**
  - `DELETE /collections/{collection_name}`
  - Removes the specified collection and all its data.

- **Clear all points from a collection (keep structure)**
  - `DELETE /collections/{collection_name}/points`
  - Deletes all points (documents) from the specified collection but keeps the collection and its configuration.
