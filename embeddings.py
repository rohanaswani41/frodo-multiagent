from typing import TypedDict, Annotated, List, Dict, Any, Tuple, Optional
from langgraph.graph.message import add_messages
import json
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import numpy as np

@dataclass
class SearchResult:
    text: str
    similarity: float
    metadata: Optional[Dict[str, Any]] = None

# [Previous type definitions remain the same...]
class ProductReview(TypedDict):
    reviewer: str
    comment: str
    rating: int

class Product(TypedDict):
    name: str
    price: float
    size: str
    type: str
    height: str
    warranty: str
    trial_period: str
    key_features: List[str]
    best_for: List[str]
    reviews: List[ProductReview]

class ProductCatalog(TypedDict):
    product_catalog: List[Product]

class EmbeddingData(TypedDict):
    chunk: str
    embedding: List[float]

class State(TypedDict):
    messages: Annotated[List[str], add_messages]
    intent: str
    context: Dict[str, Any]

class ProductProcessor:
    # [ProductProcessor implementation remains the same...]
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 512):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.model = SentenceTransformer(model_name)
        
    def load_json(self, file_path: str) -> ProductCatalog:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

    # [Rest of ProductProcessor methods remain the same...]
    # Helper function: Split text into smaller chunks
    def split_into_chunks(self, text, chunk_size):
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # Step 3: Generate embeddings for text chunks
    def generate_embeddings(self, chunks, model_name="all-MiniLM-L6-v2"):
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks, convert_to_numpy=True)
        return embeddings

    # Step 4: Save embeddings and their corresponding chunks
    def save_embeddings(self, embeddings, chunks, output_file):
        data = [{"chunk": chunk, "embedding": embedding.tolist()} for chunk, embedding in zip(chunks, embeddings)]
        with open(output_file, 'w') as f:
            json.dump(data, f)
    # Step 2: Extract and chunk text from the JSON structure
    def product_extract_and_chunk(self, json_data, chunk_size=512):
        chunks = []
        for product in json_data.get("product_catalog", []):
            # Combine product attributes into text chunks
            product_text = (
                f"Name: {product['name']}\n"
                f"Price: {product['price']}\n"
                f"Size: {product['size']}\n"
                f"Type: {product['type']}\n"
                f"Height: {product['height']}\n"
                f"Warranty: {product['warranty']}\n"
                f"Trial Period: {product['trial_period']}\n"
                f"Key Features: {', '.join(product['key_features'])}\n"
                f"Best For: {', '.join(product['best_for'])}\n"
            )
            chunks.extend(self.split_into_chunks(product_text, chunk_size))
        return chunks

    def reviews_extract_and_chunk(self, json_data, chunk_size=512):
        chunks = []
        for review in json_data.get("reviews", []):
            prod_name = review['name']
            review_text = f"Product Name: {prod_name}"
            for rev in review['reviews']:
                review_text += f"Review by {rev['reviewer']}: {rev['comment']} (Rating: {rev['rating']}/5)"
                chunks.extend(self.split_into_chunks(review_text, chunk_size))
        
        return chunks


class EmbeddingSearcher:
    def __init__(self, embedding_file: str, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding searcher."""
        self.model = SentenceTransformer(model_name)
        self.embeddings_data = self._load_embeddings(embedding_file)
        self.embeddings = np.array([item['embedding'] for item in self.embeddings_data])
        self.texts = [item['chunk'] for item in self.embeddings_data]

    def _load_embeddings(self, file_path: str) -> List[Dict[str, Any]]:
        """Load embeddings from file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Search for relevant chunks."""
        query_embedding = self.model.encode([query])[0]
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                text=self.texts[idx],
                similarity=float(similarities[idx]),
                metadata=self._extract_metadata(self.texts[idx])
            ))
        return results

    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text chunk."""
        metadata = {}
        lines = text.split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
        return metadata

    def get_product_details(self, product_name: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific product."""
        results = self.search(f"Name: {product_name}", top_k=1)
        if results:
            return results[0].metadata
        return None
