from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker, HierarchicalChunker
from langchain_docling.loader import ExportType
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore

import time
import constants
import os
import json


def load_pdf_into_docling(path_to_pdf):

    loader = DoclingLoader(
        file_path=path_to_pdf,
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(tokenizer=constants.EMBED_MODEL_ID, max_tokens=1024),
        # chunker=HierarchicalChunker(),
    )

    splits = loader.load()

    for doc in splits:
        if "dl_meta" in doc.metadata:
            print("---")
            print(doc.metadata["dl_meta"])
            # Convert nested metadata to a JSON string
            doc.metadata["dl_meta"] = json.dumps(doc.metadata["dl_meta"])

    for d in splits[:3]:
        print(f"- {d.page_content=}")
    print("...")

    return splits


def load_pdf_into_pinecone(docling_splits, index_name, namespace):

    embeddings = PineconeEmbeddings(
        model=constants.PINECONE_EMBEDDING_MODEL,
        pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
        batch_size=32,
        document_params={"input_type": "passage", "truncation": "END"},
    )

    pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing_indexes = [index_info["name"] for index_info in pinecone.list_indexes()]

    if index_name not in existing_indexes:
        pinecone.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD"), region=os.getenv("PINECONE_CLOUD")
            ),
        )
        while not pinecone.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pinecone.Index(index_name)

    docsearch = PineconeVectorStore.from_documents(
        documents=docling_splits,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )

    time.sleep(5)

    # See how many vectors have been upserted
    print("Index after upsert:")
    print(pinecone.Index(index_name).describe_index_stats())
    print("\n")
    time.sleep(2)

    return True


if __name__ == "__main__":
    load_dotenv()

    load_pdf_into_docling("files/1.pdf")
