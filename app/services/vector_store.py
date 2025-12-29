# app/services/vector_store.py
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional
import logging
import time

from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Service for managing vector storage in Pinecone
    """
    
    def __init__(self):
        """Initialize Pinecone client and index"""
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        logger.info("Initializing Pinecone client...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        
        # Create index if it doesn't exist
        self._ensure_index_exists()
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
        logger.info(f"Connected to Pinecone index: {self.index_name}")
    
    def _ensure_index_exists(self, dimension: int = 384):
        """
        Create index if it doesn't exist
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",  # cosine similarity for semantic search
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Change to your preferred region
                )
            )
            
            # Wait for index to be ready
            logger.info("Waiting for index to be ready...")
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            
            logger.info(f"Index {self.index_name} created successfully")
        else:
            logger.info(f"Index {self.index_name} already exists")
    
    def upsert_chunks(
        self,
        file_id: str,
        filename: str,
        chunks: List[str],
        embeddings: List[List[float]],
        additional_metadata: Optional[Dict] = None
    ) -> int:
        """
        Store chunks with their embeddings in Pinecone
        
        Args:
            file_id: Unique identifier for the file
            filename: Original filename
            chunks: List of text chunks
            embeddings: List of embedding vectors
            additional_metadata: Optional additional metadata to store
            
        Returns:
            Number of vectors upserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        logger.info(f"Upserting {len(chunks)} chunks for file {file_id}")
        
        # Prepare vectors for upsert
        vectors = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{file_id}_chunk_{idx}"
            
            metadata = {
                "file_id": file_id,
                "filename": filename,
                "chunk_index": idx,
                "chunk_text": chunk[:1000],  # Pinecone has metadata size limits (40KB)
                "text_length": len(chunk)
            }
            
            # Add any additional metadata
            if additional_metadata:
                metadata.update(additional_metadata)
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # Upsert in batches (Pinecone recommends batches of 100)
        batch_size = 100
        total_upserted = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            total_upserted += len(batch)
            logger.info(f"Upserted batch: {total_upserted}/{len(vectors)}")
        
        logger.info(f"Successfully upserted {total_upserted} vectors")
        return total_upserted
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Search for similar chunks using query embedding
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter (e.g., {"file_id": "abc123"})
            
        Returns:
            Search results with matches
        """
        logger.info(f"Searching for top {top_k} similar chunks")
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        logger.info(f"Found {len(results.matches)} matches")
        return results
    
    def delete_by_file_id(self, file_id: str) -> bool:
        """
        Delete all chunks associated with a file
        
        Args:
            file_id: File identifier
            
        Returns:
            True if deletion was successful
        """
        logger.info(f"Deleting all chunks for file {file_id}")
        
        try:
            # Delete by metadata filter
            self.index.delete(filter={"file_id": file_id})
            logger.info(f"Successfully deleted chunks for file {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunks: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the index
        
        Returns:
            Dictionary with index statistics
        """
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness
        }

# Create singleton instance
vector_store = VectorStore()