from dotenv import load_dotenv


from langchain_core.prompts import ChatPromptTemplate
from ChatOpenRouter.chat_open_router import ChatOpenRouter

load_dotenv()

llm = ChatOpenRouter(model_name="deepseek/deepseek-r1:free")

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")

openrouter_chain = prompt | llm

# Direct output kliye
# print(openrouter_chain.invoke({"topic": "apple"}))


# Stream output kliye
# try:
#     for chunk in openrouter_chain.stream({"topic": "banana"}):
#         print(chunk.content, end="", flush=True)
# except:
#     print("Error")
#     pass
