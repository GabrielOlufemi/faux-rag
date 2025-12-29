from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# router imports
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router 

# 
app = FastAPI(
  title = "FAUX - RAG CHATBOT APP",
  description="Document upload and RAG powered chat API",
  version="1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# routers
app.include_router(upload_router)
app.include_router(chat_router)  

@app.get("/")
def read_root():
    return {
      "message": "RAG Chatbot API",
      "status": "running",
      "endpoints": {
          "upload": "/upload",
          "chat": "/chat",
          "docs": "/docs"
      }
    }

# health check
@app.get("/health")
def health_check():
  return {"status" : "healthy"}