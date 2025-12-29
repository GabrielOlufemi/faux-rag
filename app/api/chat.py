from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store
from app.services.llm import llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    message: str
    file_ids: Optional[List[str]] = None  # Optional: search specific files only
    top_k: int = 5  # Number of relevant chunks to retrieve

class Source(BaseModel):
    filename: str
    chunk_text: str
    chunk_index: int
    similarity_score: float

class ChatResponse(BaseModel):
    reply: str
    sources: List[Source]
    num_sources: int

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with your documents using RAG
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
   
    logger.info(f"Received chat request: {request.message[:50]}...")
   
    # Generte embedding for user question
    logger.info("Generating query embedding...")
    query_embedding = embedding_service.generate_single_embedding(request.message)
   
    # Search Pinecone for relevant chunks
    logger.info(f"Searching for top {request.top_k} relevant chunks...")
   
    filter_dict = None
    if request.file_ids:
        # Search only in specific files
        filter_dict = {"file_id": {"$in": request.file_ids}}
   
    search_results = vector_store.search(
        query_embedding=query_embedding,
        top_k=request.top_k,
        filter_dict=filter_dict
    )
   
    if not search_results.matches:
        return ChatResponse(
            reply="I couldn't find any relevant information in the uploaded documents to answer your question.",
            sources=[],
            num_sources=0
        )
   
    # Extract context from search results
    context_chunks = []
    sources = []
   
    for match in search_results.matches:
        chunk_text = match.metadata.get("chunk_text", "")
        context_chunks.append(chunk_text)
       
        sources.append(Source(
            filename=match.metadata.get("filename", "Unknown"),
            chunk_text=chunk_text[:200] + "...",  # Preview only
            chunk_index=match.metadata.get("chunk_index", 0),
            similarity_score=round(match.score, 4)
        ))
   
    logger.info(f"Retrieved {len(context_chunks)} relevant chunks")
   
    # Generate response using LLM
    logger.info("Generating response with LLM...")
    context = "\n\n".join(context_chunks)
   
    llm_response = llm_service.generate_response(
        query=request.message,
        context=context
    )
   
    return ChatResponse(
        reply=llm_response,
        sources=sources,
        num_sources=len(sources)
    )