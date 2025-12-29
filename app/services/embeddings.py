from sentence_transformers import SentenceTransformer
from typing import List
import logging

# makes logger specify origin file  
logger = logging.getLogger(__name__)

class EmbeddingService:
  """
  Service for generating text embeddings using Sentence Transformers
  """

  def __init__ (self, model_name: str="all-MiniLM-L6-v2"):
    """
    Initialize the embedding model

    Args: 
    model_name: name of the sentence transofrmer used
    """

    logger.info(f"Loading embedding mode: {model_name}")
    self.model = SentenceTransformer(model_name)
    self.embedding_dim = self.model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")


  def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
    """
      Generate embeddings for a list of texts
      Args:
          texts: List of text strings to embed      
      Returns:
          List of embedding vectors
    """

    if not texts:
      return []

    logger.info(f"Generating embeddings for {len(texts)} texts")
    embeddings = self.model.encode(
      texts,
      show_progress_bar=True,
      convert_to_numpy=True
    )

    return embeddings.tolist()

  def generate_single_embedding(self, text: str) -> List[float]:
    """
    Generate embedding for  a single text
    Args:
        text: Text string to embed
    Returns:
        Embedding vector  
    """

    # generates list and picks first and only item of the list
    embedding = self.model.encode([text])[0]
    return embedding.tolist()

# singleton instance of Embedding Service class 
embedding_service = EmbeddingService()

