import google.generativeai as genai
import logging
from app.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for generating responses using Google Gemini
    """
    
    def __init__(self):
        """Initialize Gemini"""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        logger.info("Initializing Gemini...")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("Gemini initialized successfully")
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response based on query and context
        Args:
            query: User's question
            context: Retrieved document context  
        Returns:
            Generated response
        """
        prompt = f"""

        You are a FAUX an AI assistant developed by Gabriel that answers questions based on the provided context from documents.

        Context from documents:
        {context}

        User question: {query}

        Instructions:
        - Answer the question based ONLY on the information in the context above
        - If the context doesn't contain enough information to answer, say so
        - Be concise but thorough
        - Cite specific information from the context when relevant

        Answer:"""

        try:
          response = self.model.generate_content(
              prompt,
              generation_config=genai.types.GenerationConfig(
                  temperature=0.7,
                  top_p=0.95,
                  top_k=40,
                  max_output_tokens=800,
              )
          )
          
          return response.text   
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise Exception(f"Failed to generate response: {str(e)}")

# Singleton instance
llm_service = LLMService()