from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import Pinecone, ServerlessSpec
# from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


'''
Connect to Pinecone Vector Database
'''

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


'''
CREATE text_chunks and embeddings
'''

extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

# pinecone.init(
#     api_key = PINECONE_API_KEY,
#     environment = "us-east-1"
# )

index_name = "medibot"

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

# index_name = "medibot"

# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine"
#         # spec=ServerlessSpec(
#         #     cloud="aws",
#         #     region="us-east-1"
#         # )
#     )

# index = pinecone.Index(index_name)

# Embed each chunk and upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name = index_name,
    embedding = embeddings,
)




