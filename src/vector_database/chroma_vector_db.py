import logging
from typing import List, Dict, Any, Optional
import os

import chromadb
from src.embeddings.embedding_generator import EmbeddedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaVectorDB:
    def __init__(
        self, 
        db_path: str = "./chroma_db",
        collection_name: str = "notebook_lm",
        embedding_dim: int = 384
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Ensure directory exists
        os.makedirs(self.db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "l2"}
            )
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {str(e)}")
            raise
            
    def create_index(self, **kwargs):
        # ChromaDB auto-indexes during insertion, so this is a no-op to satisfy the interface.
        logger.info("ChromaDB creates indexes automatically. Skipping manual index creation.")

    def insert_embeddings(self, embedded_chunks: List[EmbeddedChunk]) -> List[str]:
        if not embedded_chunks:
            return []
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in embedded_chunks:
            data = chunk.to_vector_db_format()
            ids.append(data["id"])
            embeddings.append(data["vector"])
            documents.append(data["content"])
            
            # Format metadata flattening for Chroma constraints
            meta = {
                "source_file": data.get("source_file", ""),
                "source_type": data.get("source_type", ""),
                "page_number": data.get("page_number") if data.get("page_number") is not None else -1,
                "chunk_index": data.get("chunk_index") if data.get("chunk_index") is not None else -1,
                "start_char": data.get("start_char") if data.get("start_char") is not None else -1,
                "end_char": data.get("end_char") if data.get("end_char") is not None else -1,
                "embedding_model": data.get("embedding_model", ""),
            }
            # Add nested metadata variables if valid types
            if isinstance(data.get("metadata"), dict):
                for k, v in data["metadata"].items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[f"meta_{k}"] = v
            metadatas.append(meta)
            
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Inserted {len(ids)} embeddings into Chroma database")
            return ids
        except Exception as e:
            logger.error(f"Error inserting embeddings into Chroma: {str(e)}")
            raise
            
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    meta = results["metadatas"][0][i] or {}
                    
                    # Reconstruct nested metadata map
                    nested_metadata = {}
                    for k, v in list(meta.items()):
                        if k.startswith("meta_"):
                            nested_metadata[k[5:]] = v
                            del meta[k]
                            
                    formatted_result = {
                        'id': results["ids"][0][i],
                        'score': results["distances"][0][i],
                        'content': results["documents"][0][i],
                        'citation': {
                            'source_file': meta.get('source_file'),
                            'source_type': meta.get('source_type'),
                            'page_number': meta.get('page_number') if meta.get('page_number') != -1 else None,
                            'chunk_index': meta.get('chunk_index'),
                            'start_char': meta.get('start_char') if meta.get('start_char') != -1 else None,
                            'end_char': meta.get('end_char') if meta.get('end_char') != -1 else None,
                        },
                        'metadata': nested_metadata,
                        'embedding_model': meta.get('embedding_model')
                    }
                    formatted_results.append(formatted_result)
                    
            logger.info(f"Search completed: {len(formatted_results)} results found in Chroma")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during Chroma search: {str(e)}")
            raise
            
    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted from Chroma")
            # Clear reference and recreate empty to mimic previous behavior
            self.collection = self.client.create_collection(
                name=self.collection_name, 
                metadata={"hnsw:space": "l2"}
            )
        except Exception as e:
            logger.error(f"Error deleting Chroma collection: {str(e)}")
            raise
            
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"] and len(results["ids"]) > 0:
                meta = results["metadatas"][0] or {}
                
                # Reconstruct nested metadata map
                nested_metadata = {}
                for k, v in list(meta.items()):
                    if k.startswith("meta_"):
                        nested_metadata[k[5:]] = v
                        
                return {
                    "id": results["ids"][0],
                    "content": results["documents"][0],
                    "metadata": nested_metadata,
                    "source_file": meta.get("source_file"),
                    "source_type": meta.get("source_type"),
                    "page_number": meta.get("page_number") if meta.get("page_number") != -1 else None,
                    "chunk_index": meta.get("chunk_index")
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving chunk from Chroma by ID {chunk_id}: {str(e)}")
            return None
            
    def close(self):
        # ChromaDB PersistentClient doesn't require explicit closing like Milvus
        pass
