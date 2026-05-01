from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec,Pinecone
from pathlib import Path
import os


load_dotenv()


# ========================================================================================================================

# Call Api key from .env
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ========================================================================================================================

# ========================================================================================================================

BASE_DIR =Path(__file__).resolve().parent
DATA_PATH = BASE_DIR /"Data"

# Call and read data & split it into chunks
extracted_data=load_pdf(str(DATA_PATH))
text_chunks=text_split(extracted_data)

# call embedding
embeddings = download_hugging_face_embeddings()

# ========================================================================================================================

# Create Pinecone to store vectors we got from embedding

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

# initialize Pinecone client with API key
pc = Pinecone(
    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY"))


index_name="medicalbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine", # can we choose dotproudct  
        spec=ServerlessSpec(
            cloud="aws",   # free plan
            region="us-east-1"
        )
    )

print("Index ready:", pc.list_indexes().names())

# ========================================================================================================================

# Start Storeing the Vectors
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)