from langchain_google_genai import ChatGoogleGenerativeAI
import textwrap

from dotenv import load_dotenv

from pydantic import BaseModel, Field

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


class QueryAgentOutput(BaseModel):
    queries: list[str] = Field(..., description="List of queries to be executed")
    namespaces: list[str] = Field(..., description="List of namespaces to query from")


class LegalAIBot:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index("legalaibot-litigation")

    def choose_query_and_namespace(self, user_query):
        system_prompt = """
        You are an intelligent assistant that recognises the intent of the user query and returns the query and the namespace of the vector database to query from.
        You can modify the given user query to get better results. You can also choose multiple namespaces and to query from. But make sure that you query relevant namespaces according to the user query.

        Available namespaces are:
        - finance-laws: Contains information about finance laws in India.
        - litigation: Contains information about litigation in India.

        You have to return a json in following format only:
        {
            queries: ["What is NBFC?", "What is the procedure to file a case in court?"],
            namespaces: ["finance-laws", "litigation"]
        }   
        """
        system_prompt = textwrap.dedent(system_prompt).strip()
        print(system_prompt)

        messages = [
            ("system", system_prompt),
            ("user", "User query: " + user_query),
        ]

        structured_llm = self.llm.with_structured_output(QueryAgentOutput)
        result = structured_llm.invoke(messages)
        print(result)

        return result

    def summarizeOutput(self, user_query, retrieved_data):
        prompt = f"""
        You are a legal expert and language simplifier. For the given user query, you have to summarize the retrieved data and answer the user query in a simple and concise manner while maintaining legal accuracy. If no data is present, you have to mention that no data was found. You can also provide additional information if you think it is relevant."""

        prompt = textwrap.dedent(prompt).strip()
        print(prompt)

        messages = [
            ("system", prompt),
            (
                "user",
                "User query: " + user_query + "\Supporting data:\n" + retrieved_data,
            ),
        ]

        ai_msg = self.llm.invoke(messages)
        print(ai_msg)

        return ai_msg

    def query_database(self, query, namespace, num=3):
        print(f"Querying database with query: {query} and namespace: {namespace}")

        query_embedding = self.pc.inference.embed(
            model=constants.PINECONE_EMBEDDING_MODEL,
            inputs=[query],
            parameters={"input_type": "query"},
        )

        results = self.index.query(
            namespace=namespace,
            vector=query_embedding[0].values,
            top_k=num,
            include_values=False,
            include_metadata=True,
        )

        new_result = self.pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            documents=[match["metadata"]["text"] for match in results["matches"]],
            top_n=3,
            return_documents=True,
            parameters={"truncate": "END"},
        )

        texts = [item["document"]["text"] for item in new_result.data]

        print(texts)

        return texts

    def generateOutput(self, user_query):
        print("=== Generating output ===")

        database_queries = self.choose_query_and_namespace(user_query)
        print(database_queries)

        retrieved_data = ""

        for query, namespace in zip(
            database_queries.queries, database_queries.namespaces
        ):
            ans = self.query_database(query, namespace)

            retrieved_data += f"Query: {query}\nNamespace: {namespace}\n---\n"
            retrieved_data += "\n\n".join(ans)
            retrieved_data += "\n\n"

        result = self.summarizeOutput(user_query, retrieved_data)

        return result


if __name__ == "__main__":
    load_dotenv()
    bot = LegalAIBot()
    # bot.choose_query_and_namespace("What is NBFC?")
    bot.generateOutput("What are the steps involved in filing a lawsuit in India?")
