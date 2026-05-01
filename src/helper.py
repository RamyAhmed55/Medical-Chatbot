from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List


# ========================================================================================================================

# Load data 
def load_pdf(data : str) -> List[Document]:
    """
    summary:
        Load PDF files from a directory and convert to document
    Args:
        data (str): Path folder contains Pdfs

    return:
        List[Any]: return list of pdf documents "text"

    NextStep:
        i will pass the output to `text_split()` or any splitter for chunking into smaller parts (Chuncks)
    """

    loader = DirectoryLoader(data, glob="*.pdf", # Any PDF
                    loader_cls=PyPDFLoader)
    documents =loader.load()

    return documents


# ========================================================================================================================

# Split data to Chuncks
def text_split(extracted_data: List[Document]) -> List[Document]:
    """
    Summary:
        Splits large PDF documents into smaller text chunks for better processing by LLM models.

    Args:
        extracted_data (List[Any]): Loaded documents from PDF files.

    Returns:
        List[Any]: List of smaller text chunks.

    NextStep:
        The chunks are passed to embedding generation using `download_hugging_face_embeddings()`.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# ========================================================================================================================


def download_hugging_face_embeddings():

    """
    Summary:
        we will use it to convert the document into embedding vectors to be stored in Vector DB 

    Args: None

    Returns:
         object -> HuggingFaceEmbeddings

    NextStep:
        save embedding in vectorDB By pinecone
    """

    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings