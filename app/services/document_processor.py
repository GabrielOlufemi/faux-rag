from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Changed this line

def extract_text_from_pdf(file_path: Path) -> str:
  """
  Extract text from PDF file
  """
  text = []

  with open(file_path, 'rb') as file:
    reader = PdfReader(file)
    for page in reader.pages:
      text.append(page.extract_text())

  return "\n".join(text)

def extract_text_from_docx(file_path: Path) -> str:
  """Extract text from DOCX file"""
  doc = Document(file_path)
  text = []

  for paragraph in doc.paragraphs:
    if paragraph.text.strip():
      text.append(paragraph.text)

  return "\n".join(text)

def extract_text_from_txt(file_path: Path) -> str:
  """Extract text from txt file"""
  try:
    with open(file_path, 'r', encoding='utf-8') as file:
      return file.read()
  except UnicodeDecodeError:
    with open(file_path, 'r', encoding='latin-1') as file:
      return file.read()

def process_document(file_path: Path) -> str:
  """
  Process Document and extract based on file type
  """

  file_type = file_path.suffix.lower()

  if file_type == '.pdf':
    return extract_text_from_pdf(file_path)
  elif file_type == '.docx':
    return extract_text_from_docx(file_path)
  elif file_type == '.txt':
    return extract_text_from_txt(file_path)
  else:
    raise ValueError(f"Unsupported file type : {file_type}")


# CHUNKING
def chunk_text(
  text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:

  """
  Splits text into overlapping chunks for context
  Args:
      text: The text to split
      chunk_size: Maximum size of each chunk in characters
      chunk_overlap: Number of characters to overlap between chunks
  Returns:
      List of text chunks
  """

  if not text or not text.strip():
    return []

  splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
  )

  chunks = splitter.split_text(text)

  chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50 ]

  return chunks