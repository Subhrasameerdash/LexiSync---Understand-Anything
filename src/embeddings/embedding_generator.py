import logging
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass

# Changed: replaced FastEmbed with sentence-transformers
# Model: sentence-transformers/all-MiniLM-L6-v2
# Embedding dimension: 384 — identical to the previous FastEmbed model,
# so the Milvus collection schema (embedding_dim=384) requires NO changes.
from sentence_transformers import SentenceTransformer

from src.document_processing.doc_processor import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """Document chunk with its embedding vector"""
    chunk: DocumentChunk
    embedding: np.ndarray
    embedding_model: str

    def to_vector_db_format(self) -> Dict[str, Any]:
        return {
            'id': self.chunk.chunk_id,
            'vector': self.embedding.tolist(),
            'content': self.chunk.content,
            'source_file': self.chunk.source_file,
            'source_type': self.chunk.source_type,
            'page_number': self.chunk.page_number,
            'chunk_index': self.chunk.chunk_index,
            'start_char': self.chunk.start_char,
            'end_char': self.chunk.end_char,
            'metadata': self.chunk.metadata,
            'embedding_model': self.embedding_model
        }


class EmbeddingGenerator:
    # Changed: default model is now all-MiniLM-L6-v2 (384-dim, same as before)
    # Alternative high-quality option: "BAAI/bge-base-en-v1.5" (768-dim) —
    # if used, update MilvusVectorDB(embedding_dim=768) accordingly.
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: SentenceTransformer = None
        self.embedding_dim: int = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            logger.info(f"Initializing sentence-transformers embedding model: {self.model_name}")
            # SentenceTransformer downloads and caches the model locally on first run.
            self.model = SentenceTransformer(self.model_name)

            # Probe the actual output dimension so it matches whatever model is chosen.
            sample = self.model.encode(["test"], normalize_embeddings=True)
            self.embedding_dim = sample.shape[1]

            logger.info(
                f"Embedding model ready. Dimension: {self.embedding_dim}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise

    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List["EmbeddedChunk"]:
        if not chunks:
            return []

        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        try:
            texts = [chunk.content for chunk in chunks]

            # encode() returns a numpy array of shape (N, dim)
            # normalize_embeddings=True → unit-length vectors (cosine-friendly)
            embeddings: np.ndarray = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            embedded_chunks = []
            for chunk, embedding in zip(chunks, embeddings):
                embedded_chunk = EmbeddedChunk(
                    chunk=chunk,
                    embedding=np.array(embedding, dtype=np.float32),
                    embedding_model=self.model_name,
                )
                embedded_chunks.append(embedded_chunk)

            logger.info(f"Successfully generated {len(embedded_chunks)} embeddings")
            return embedded_chunks

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def generate_query_embedding(self, query_text: str) -> np.ndarray:
        try:
            embedding = self.model.encode(
                [query_text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.array(embedding[0], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim

    def batch_generate_embeddings(
        self,
        chunks_batches: List[List[DocumentChunk]],
        batch_size: int = 32,
    ) -> List[List["EmbeddedChunk"]]:
        all_embedded_batches = []
        for i, chunk_batch in enumerate(chunks_batches):
            logger.info(f"Processing batch {i + 1}/{len(chunks_batches)}")

            embedded_batch = []
            for j in range(0, len(chunk_batch), batch_size):
                sub_batch = chunk_batch[j: j + batch_size]
                embedded_sub_batch = self.generate_embeddings(sub_batch)
                embedded_batch.extend(embedded_sub_batch)

            all_embedded_batches.append(embedded_batch)

        return all_embedded_batches


if __name__ == "__main__":
    from src.document_processing.doc_processor import DocumentProcessor

    doc_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator()

    try:
        chunks = doc_processor.process_document("data/raft.pdf")
        embedded_chunks = embedding_generator.generate_embeddings(chunks)

        if embedded_chunks:
            sample = embedded_chunks[0]
            print(f"Sample embedding shape: {sample.embedding.shape}")
            print(f"Sample content: {sample.chunk.content[:100]}...")
            print(f"Citation info: {sample.chunk.get_citation_info()}")

            query = "What is the main topic?"
            query_embedding = embedding_generator.generate_query_embedding(query)
            print(f"Query embedding shape: {query_embedding.shape}")

    except Exception as e:
        print(f"Error in example usage: {e}")