import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel
from tools import search_tool, save_tool

load_dotenv()

tools = [search_tool, save_tool]

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
    ("system",
     """ 
     You are a research assistant that can help generate
     a research paper.answer the user query and use necessary tools to
      answer the query.wrap the output in this format and provide
       no other text \n{format_instructions}
       """,
     ),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt,
    tools = tools
    )

agent_executor = AgentExecutor(agent=agent, tools=tools)
query = input("Enter a query: ")
raw_response = agent_executor.invoke({"query": query})
try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
