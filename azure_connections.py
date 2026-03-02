import sys

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from llama_index.core import Settings
import os
import nest_asyncio

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import  VectorStoreIndex


from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector


nest_asyncio.apply()

#API info. Replace with your own keys and end points
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = "https://mrs-test-1.openai.azure.com/"
api_version = "2024-05-01-preview"

#Setup the LLM
Settings.llm=AzureOpenAI(
    #model="gpt-5.2-chat",
    deployment_name="gpt-5-chat",
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

#-------------------------------------------------------------------
#Setup Aeroflow document index
#-------------------------------------------------------------------
aeroflow_documents=SimpleDirectoryReader(
    input_files=["AeroFlow_Specification_Document.pdf"])\
            .load_data()

#Read documents into nodes
aeroflow_nodes=splitter.get_nodes_from_documents(aeroflow_documents)
#Create a vector Store
aeroflow_index=VectorStoreIndex(aeroflow_nodes)
#Create a query engine
aeroflow_query_engine = aeroflow_index.as_query_engine()

#-------------------------------------------------------------------
#Setup EchoSprint document index
#-------------------------------------------------------------------
ecosprint_documents=SimpleDirectoryReader(
    input_files=["EcoSprint_Specification_Document.pdf"])\
            .load_data()
#Read documents into nodes
ecosprint_nodes=splitter.get_nodes_from_documents(ecosprint_documents)
#Create a vector Store
ecosprint_index=VectorStoreIndex(ecosprint_nodes)
#Create a query engine
ecosprint_query_engine = ecosprint_index.as_query_engine()

#Create a query engine Tool for NoSQL
aeroflow_tool = QueryEngineTool.from_defaults(
    query_engine=aeroflow_query_engine,
    name="Aeroflow specifications",
    description=(
        "Contains information about Aeroflow : Design, features, technology, maintenance, warranty"
    ),
)

#Create a query engine Tool for NLP
ecosprint_tool = QueryEngineTool.from_defaults(
    query_engine=ecosprint_query_engine,
    name="EcoSprint specifications",
    description=(
        "Contains information about EcoSprint : Design, features, technology, maintenance, warranty"
    ),
)

#Create a Router Agent. Provide the Tools to the Agent
router_agent=RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        aeroflow_tool,
        ecosprint_tool,
    ],
    verbose=True
)


#response = router_agent.query("What colors are available for AeroFlow ?")
response = router_agent.query(sys.argv[1])
print("\nResponse: ",str(response))


