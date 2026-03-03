import sys

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from llama_index.core import Settings
import os
import asyncio

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import  VectorStoreIndex


from llama_index.core.tools import QueryEngineTool

from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import  VectorStoreIndex
from llama_index.core.tools import QueryEngineTool

from llama_index.core.tools import FunctionTool

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

#-------------------------------------------------------------
# Tool 1 : Function that returns the list of items in an order
#-------------------------------------------------------------
def get_order_items(order_id: int) -> List[str] :
    """Given an order Id, this function returns the 
    list of items purchased for that order"""
    
    order_items = {
            1001: ["Laptop","Mouse"],
            1002: ["Keyboard","HDMI Cable"],
            1003: ["Laptop","Keyboard"]
        }
    if order_id in order_items.keys():
        return order_items[order_id]
    else:
        return []

#-------------------------------------------------------------
# Tool 2 : Function that returns the delivery date for an order
#-------------------------------------------------------------
def get_delivery_date(order_id: int) -> str:
    """Given an order Id, this function returns the 
    delivery date for that order"""

    delivery_dates = {
            1001: "10-Jun",
            1002: "12-Jun",
            1003: "08-Jun"       
    }
    if order_id in delivery_dates.keys():
        return delivery_dates[order_id]
    else:
        return []

#----------------------------------------------------------------
# Tool 3 : Function that returns maximum return days for an item
#----------------------------------------------------------------
def get_item_return_days(item: str) -> int :
    """Given an Item, this function returns the return support
    for that order. The return support is in number of days"""
    
    item_returns = {
            "Laptop"     : 30,
            "Mouse"      : 15,
            "Keyboard"   : 15,
            "HDMI Cable" : 5
    }
    if item in item_returns.keys():
        return item_returns[item]
    else:
        #Default
        return 45


#nest_asyncio.apply()

#API info. Replace with your own keys and end points
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = "https://mrs-test-1.openai.azure.com/"
api_version = "2024-05-01-preview"

#Setup the LLM
Settings.llm=AzureOpenAI(
    deployment_name="gpt-4o",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version='2025-01-01-preview',
)

#Setup the embedding model RAG
Settings.embed_model= AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version='2023-05-15',
)

#Create indexes from vector search
splitter=SentenceSplitter(chunk_size=1024)

#-------------------------------------------------------------
# Tool 4 : Vector DB that contains customer support contacts
#-------------------------------------------------------------
#Setup vector index for return policies
support_docs=SimpleDirectoryReader(input_files=["Customer Service.pdf"]).load_data()

splitter=SentenceSplitter(chunk_size=1024)
support_nodes=splitter.get_nodes_from_documents(support_docs)
support_index=VectorStoreIndex(support_nodes)
support_query_engine = support_index.as_query_engine()

#Create tools for the 3 functions and 1 index
order_item_tool = FunctionTool.from_defaults(fn=get_order_items)
delivery_date_tool = FunctionTool.from_defaults(fn=get_delivery_date)
return_policy_tool = FunctionTool.from_defaults(fn=get_item_return_days)

support_tool = QueryEngineTool.from_defaults(
    query_engine=support_query_engine,
    description=(
        "Customer support policies and contact information"
    ),
)

#Setup the Agent worker in LlamaIndex with all the Tools
#This is the tool executor process
agent = FunctionAgent(
    tools=[order_item_tool, 
     delivery_date_tool,
     return_policy_tool,
     support_tool,
    ], 
    llm=Settings.llm, 
    system_prompt="Function Agent tool",
)

#Create an Agent Orchestrator with LlamaIndex
#agent = AgentRunner(agent_worker)

#Get return policy for an order
async def main(query):
    response = await agent.run(
        query
        #"What is the return policy for order number 1001"
    )
    print("\n Final output : \n", response)

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
