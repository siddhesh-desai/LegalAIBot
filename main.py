from dotenv import load_dotenv


from langchain_core.prompts import ChatPromptTemplate
from ChatOpenRouter.chat_open_router import ChatOpenRouter

load_dotenv()

# llm = ChatOpenRouter(model_name="deepseek/deepseek-r1:free")

# prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")

# openrouter_chain = prompt | llm

# Direct output kliye
# print(openrouter_chain.invoke({"topic": "apple"}))


# Stream output kliye
# try:
#     for chunk in openrouter_chain.stream({"topic": "banana"}):
#         print(chunk.content, end="", flush=True)
# except:
#     print("Error")
#     pass


# from DataIngestor.DataIngestor import load_pdf_into_docling, load_pdf_into_pinecone


# splits = load_pdf_into_docling("files/finance-laws.pdf")


# load_pdf_into_pinecone(splits, "legalaibot-litigation", "finance-laws")

from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeEmbeddings
import constants
from dotenv import load_dotenv
import os

from pinecone import Pinecone

# Import the Pinecone library
# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time


load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("legalaibot-litigation")

query = "What is NBFC?"

# Convert the query into a numerical vector that Pinecone can search with
query_embedding = pc.inference.embed(
    model=constants.PINECONE_EMBEDDING_MODEL,
    inputs=[query],
    parameters={"input_type": "query"},
)

results = index.query(
    namespace="finance-laws",
    vector=query_embedding[0].values,
    top_k=3,
    include_values=False,
    include_metadata=True,
)

new_result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query=query,
    documents=[match["metadata"]["text"] for match in results["matches"]],
    top_n=3,
    return_documents=True,
    parameters={"truncate": "END"},
)

for match in results["matches"]:
    text = match["metadata"].get("text", "No text available")
    print(f"Score: {match['score']}\nText: {text}\n")

print("=====================================")

print("Reranked results:")
print(new_result)
