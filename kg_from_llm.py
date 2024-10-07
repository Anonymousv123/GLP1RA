# -*- coding: utf-8 -*-
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough
from langchain.chains import GraphCypherQAChain
import pandas as pd
import os
import pickle
import time
import requests


# Load from environment
NEO4J_URI = ''
NEO4J_DATABASE = ''
NEO4J_PASSWORD = ''
NEO4J_USERNAME = ''

OPENAI_API_KEY = ''

PTH ="..."
save_path = '/....pkl'

# BUILD A KG from posts

## combined_social_data.csv is a combined csv file of all social media posts (twitter and reddit)
df = pd.read_csv(os.path.join(PTH,"combined_social_data.csv"))


# Set up connection to graph instance using LangChain
kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)


# Remove duplicate rows based on 'col1' (keeping the first occurrence)
df = df.drop_duplicates(subset='combined_content', keep='first')

# Function to count words in a text
def word_count(text):
    return len(str(text).split())

# Remove rows with text shorter than 30 words
df_filtered = df[df['combined_content'].apply(word_count) >= 30]
df_filtered

"""## Graph"""
# connect to Neo4j graph DB
graph = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)

from langchain_core.documents import Document

# build a graph only from posts
documents = []
for index, row in df_filtered.iterrows():
    doc = Document(
      page_content= row['combined_content'],
      metadata={
          'author' : row['Post Author'],
          'post_date' : row['post_date'],
          'post_id' : row['Post ID'],
          'subreddit' : row['subreddit'],
      }
    )
    documents.append(doc)


# Load existing progress if available
graph_documents_list=[]
start_index = 0

try:
    with open(save_path, 'rb') as f:
        graph_documents_list = pickle.load(f)
    start_index = len(graph_documents_list) * 50 # Start the loop where it left off based on the length of the saved data
    print(f"Resuming from index: {start_index}")
except FileNotFoundError:
    print("No existing file found.", f"Starting from index: {start_index}")



llm = ChatOpenAI(temperature=0,
                 model="gpt-4o-mini",
                 openai_api_key = OPENAI_API_KEY
                 )

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=[],
    allowed_relationships=[],
    )


for i in range(start_index, len(documents),50):
    print(f"Processing document {i+1}/{len(documents)}")
    try:
        graph_documents = llm_transformer.convert_to_graph_documents(documents[i:i+50])
    except Exception as e:
        print(f"Error: {e}")
        if "sending request too quickly" in str(e):
            time.sleep(3)  # Sleep for 3 seconds before retrying
            graph_documents = llm_transformer.convert_to_graph_documents(documents[i:i+50])
    time.sleep(0.5)  # General throttle to avoid errors

    graph_documents_list.append(graph_documents)

    # Save progress after every 10 iteration
    if (i // 5) % 10 == 0:
      with open(save_path, 'wb') as f:
          pickle.dump(graph_documents_list, f)
          print(f"Progress saved after processing document {i+1}/{len(documents)} to: {os.getcwd()}")

print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")

# Storing to graph database in NEO4J
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

"""## Query NEO4J"""
# connect to Neo4j graph DB and read the graph
graph = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)


llm = ChatOpenAI(temperature=0,
                 model="gpt-4o-mini",
                 openai_api_key = OPENAI_API_KEY
                 )

chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True)

# Initialize the Neo4j driver
driver = GraphDatabase.driver(
        uri = NEO4J_URI,
        auth = (NEO4J_USERNAME,
                NEO4J_PASSWORD)
        )

# count number of nodes and edges in NEO4J
def count_nodes_and_edges(driver):
    # Define Cypher queries
    node_query = "MATCH (n) RETURN COUNT(n) AS NumberOfNodes"
    edge_query = "MATCH ()-[r]->() RETURN COUNT(r) AS NumberOfEdges"

    with driver.session() as session:
        # Execute the queries and fetch results
        node_result = session.run(node_query)
        edge_result = session.run(edge_query)

        # Extract counts from results
        number_of_nodes = node_result.single()['NumberOfNodes']
        number_of_edges = edge_result.single()['NumberOfEdges']

        # Print the results
        print(f"Number of Nodes: {number_of_nodes}")
        print(f"Number of Edges: {number_of_edges}")

# Call the function
count_nodes_and_edges(driver)

"""### query the graph using LLM"""

# vector search (RAG)
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(model='text-embedding-3-large',dimensions=1536, openai_api_key = OPENAI_API_KEY),
    search_type="hybrid",
    node_label="Document",
    url = NEO4J_URI,
    username = NEO4J_USERNAME,
    password = NEO4J_PASSWORD,
    database = NEO4J_DATABASE,
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

"""### Graph RAG: identify relevant node entities in the prompt and then return their direct neighborhood in the graph
"""

graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
    )


## Detect entities in the question of a user to map it into the graph for search
class Entities(BaseModel):
    """Identifying information about relevant entities."""

    names: list[str] = Field(
        ...,
        description="""
        All the drug side effects that appear in the text
        """,
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting the following entities: Healthcondission, Health issue, Drug, Medication, Health_concern, Health_condition, Healthissue, Condition, Health symptom, Health_symptom, Side effect, Hormonalcondition, Physiological state, Medical condition, Side_effect, Sideeffect, or Symptom entities that appear in the text entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)

# test the Entities class to see if it can identify entities
entity_chain.invoke({"question": "What are the side effects of GLP-1 RA medication?"}).names

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print(f"Generated Query: {full_text_query}")
    print(full_text_query)
    return full_text_query.strip()


# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

print(structured_retriever("what are the side effects of ozempic?"))

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    return final_data