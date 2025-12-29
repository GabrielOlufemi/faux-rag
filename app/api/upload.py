from fastapi import APIRouter, File, UploadFile, HTTPException
from pathlib import Path
import uuid
from datetime import datetime
import logging

from app.services.document_processor import (
  process_document, 
  chunk_text
)

from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store  # Add this import

from app.config import (
  UPLOAD_DIR, MAX_FILE_SIZE_MB,
  ALLOWED_EXTENSIONS, CHUNK_SIZE, CHUNK_OVERLAP
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])

def validate_file(file: UploadFile) -> None:
  """
  Validate Uploaded file
  Raise Exception if file is invalid
  """

  # check file extension
  file_extension = Path(file.filename).suffix.lower()
  if file_extension not in ALLOWED_EXTENSIONS:
    raise HTTPException(
      status_code=400,
      detail=f"File type not allowed. Allowed types: {','.join(ALLOWED_EXTENSIONS)}"
    )

  # check if file has content
  if not file.filename:
    raise HTTPException(
      status_code=400,
      detail="No file provided"
    )

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
  """
  Upload a document satisfying allowed types
  """

  logger.info(f"Received upload request for: {file.filename}")

  # validate file
  validate_file(file)
  # read content
  contents = await file.read()

  # check file size
  file_size_mb = len(contents) / (1024 * 1024)
  if file_size_mb > MAX_FILE_SIZE_MB:
    raise HTTPException(
      status_code=413,
      detail=f"File too large. Max Size: {MAX_FILE_SIZE_MB}MB"
    )

  # generate unique file_name
  file_id = str(uuid.uuid4())
  file_extension = Path(file.filename).suffix
  unique_filename = f"{file_id}{file_extension}"
  file_path = UPLOAD_DIR / unique_filename

  # save file
  with open(file_path, "wb") as f:
    f.write(contents)

  logger.info(f"File saved: {file_path}")

  # extract text
  try:
    logger.info(f"Extracting text from document...")
    extracted_text = process_document(file_path)

    if not extracted_text or not extracted_text.strip():
      raise ValueError("No text could be extracted from the document")

    logger.info(f"Extracted {len(extracted_text)} characters")

    # chunking stuff
    logger.info("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    if not chunks:
      raise ValueError("No valid chunks created from uploaded document")

    logger.info(f"Created {len(chunks)} chunks")

    # embeds
    logger.info(f"Generating embeddings...")
    embeddings = embedding_service.generate_embeddings(chunks)

    logger.info(f"Generated {len(embeddings)} embeddings")

    # store in pinecone
    logger.info("Storing embeddings in Pinecone...")
    num_vectors = vector_store.upsert_chunks(
        file_id=file_id,
        filename=file.filename,
        chunks=chunks,
        embeddings=embeddings,
        additional_metadata={
            "upload_date": datetime.now().isoformat(),
            "file_size_mb": round(file_size_mb, 2)
        }
    )
    
    logger.info(f"Successfully stored {num_vectors} vectors in Pinecone")

    # response message
    return {
      "success": True,
      "file_id": file_id,
      "filename": file.filename,
      "size_bytes": len(contents),
      "size_mb": round(file_size_mb, 2),
      "text_length": len(extracted_text),
      "num_chunks": len(chunks),
      "num_embeddings": len(embeddings),
      "vectors_stored": num_vectors,
      "embedding_dimension": len(embeddings[0]) if embeddings else 0,
      "upload_date": datetime.now().isoformat(),
      "message": "File uploaded, processed, and stored in vector database successfully"
    }
    
  except Exception as e:
    # clean up if processing failes
    logger.error(f"Processing failed: {str(e)}")
    file_path.unlink()
    raise HTTPException(
      status_code=500,
      detail=f"Failed to process document: {str(e)}"
    )

@router.get("/list")
async def list_files():
  """
  List all uploaded files
  """

  files = []
  for file_path in UPLOAD_DIR.iterdir():
    if file_path.is_file():
      files.append({
        "filename": file_path.name,
        "size": file_path.stat().st_size,
        "upload_date": datetime.fromtimestamp(
            file_path.stat().st_ctime
        ).isoformat()      
       })

  return {
    "files" : files,
    "total" : len(files)
  }

@router.delete("/{file_id}")
async def delete_file(file_id: str):
  """
  Delete an uploaded file
  """

  found = False
  for file_path in UPLOAD_DIR.iterdir():
    if file_path.stem == file_id:
      # deletes from file path
      file_path.unlink()
      found = True

      # Delete from Pinecone
      logger.info(f"Deleting vectors for file {file_id} from Pinecone")
      vector_store.delete_by_file_id(file_id)
      break

  if not found:
    raise HTTPException(
      status_code=404,
      detail="File not found"
    )

  return {
    "success" : True,
    "message" : "File deleted successfully" 
  }

@router.get("/stats")
async def get_vector_stats():
    """
    Get statistics about the vector database
    """
    try:
        stats = vector_store.get_index_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stats: {str(e)}"
        )