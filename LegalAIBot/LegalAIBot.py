from pinecone import Pinecone
import constants
import textwrap
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import os
from langchain.memory import ConversationBufferMemory


class QueryOutputFormat(BaseModel):
    """Pydantic model for the output of the choose_query_and_namespace method"""

    queries: list[str] = Field(..., description="List of queries to be executed")
    namespaces: list[str] = Field(..., description="List of namespaces to query from")


class QueryAgent:
    """Agent jo database ko query karega and relevant results return karega"""

    def __init__(self):
        """Initialize karo QueryAgent class ko"""

        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(constants.PINECONE_INDEX_NAME)

        self.llm = ChatGoogleGenerativeAI(
            model=constants.QUERY_AGENT_MODEL,
            temperature=0.8,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        ).with_structured_output(QueryOutputFormat)

    def choose_query_and_namespace(self, user_query) -> tuple[list[str], list[str]]:
        """User query ke hisaab se namespace choose karo and query ko modify karo"""

        system_prompt = textwrap.dedent(constants.QUERY_AGENT_SYSTEM_PROMPT).strip()

        messages = [
            ("system", system_prompt),
            ("user", "\nUser query: " + user_query),
        ]

        result = self.llm.invoke(messages)

        return (result.queries, result.namespaces)

    def query_database(self, query, namespace, top_k=3):
        """Database ko query karega and top_k results return karega"""

        query_embedding = self.pc.inference.embed(
            model=constants.PINECONE_EMBEDDING_MODEL,
            inputs=[query],
            parameters={"input_type": "query"},
        )

        intermediate_results = self.index.query(
            namespace=namespace,
            vector=query_embedding[0].values,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
        )

        if not intermediate_results["matches"]:
            return "No relevant data found"

        reranked_results = self.pc.inference.rerank(
            model=constants.PINECONE_RERANKER_MODEL,
            query=query,
            documents=[
                match["metadata"]["text"] for match in intermediate_results["matches"]
            ],
            top_n=top_k,
            return_documents=True,
            parameters={"truncate": "END"},
        )

        retrieved_results = [item["document"]["text"] for item in reranked_results.data]

        retrieved_results_str = "---\n" + query + "\n"
        retrieved_results_str += "\n".join(retrieved_results)
        retrieved_results_str += "\n---\n"

        return retrieved_results_str

    def retrieve_relevant_data(self, user_query):
        """User query ke hisaab se relevant data retrieve karega"""

        queries, namespaces = self.choose_query_and_namespace(user_query)

        retrieved_data = []

        for query, namespace in zip(queries, namespaces):
            retrieved_data.append(self.query_database(query, namespace))

        return retrieved_data


class SummarizationAgent:
    """Agent jo complex legal text ko easily summarise karega user ko batayega"""

    def __init__(self):
        """Initialize karo SummarizationAgent class ko"""

        self.llm = ChatGoogleGenerativeAI(
            model=constants.SUMMARIZATION_AGENT_MODEL,
            temperature=0.8,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def generate_summary(self, user_query, retrieved_data, chat_history=None):
        """User query ke hisaab se retrieved data ko summarise karke, jawab dega"""

        system_prompt = textwrap.dedent(
            constants.SUMMARIZATION_AGENT_SYSTEM_PROMPT
        ).strip()

        user_prompt = "---\n" + "User Query" + user_query + "\n---\n"
        user_prompt += "\nRetrieved Data:\n"
        user_prompt += "\n".join(retrieved_data)

        if chat_history:
            user_prompt += "\nChat History:\n"
            user_prompt += "\n".join(chat_history)

        messages = [
            ("system", system_prompt),
            ("user", user_prompt),
        ]

        result = self.llm.invoke(messages)

        return result.content


class LegalAIBot:
    """LegalAIBot class jo QueryAgent and SummarizationAgent ko manage karega and conversation rakhega chalu"""

    def __init__(self):
        """Initialize karo LegalAIBot class ko"""

        self.query_agent = QueryAgent()
        self.summarization_agent = SummarizationAgent()
        self.memory = ConversationBufferMemory()

    def generate_output(self, user_query):
        """User query ke hisaab se relevant data retrieve karega and summarize karke output dega"""

        chat_history = self.memory.load_memory_variables({})["history"]

        if len(chat_history) > 6:
            chat_history = chat_history[-6:]

        retrieved_data = self.query_agent.retrieve_relevant_data(user_query)
        summary = self.summarization_agent.generate_summary(
            user_query, retrieved_data, chat_history
        )

        self.memory.save_context({"input": user_query}, {"output": summary})

        return summary

    def get_chat_history(self):
        """Chat history ko retrieve karega"""

        return self.memory.load_memory_variables({})["history"]
