from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, JSONLoader
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_ollama import OllamaEmbeddings
import streamlit as st
from pyvis.network import Network
from langchain_core.runnables import RunnableLambda
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()



class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description="""Any  person, organization, ideas, or business entities that 
        appear in the text
        
        """,
    )

class GraphRag:
    
    def __init__(self):
        
        
        self.url = os.getenv('NEO4J_URI')
        self.username = os.getenv('NEO4J_USERNAME')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.llm_model = os.getenv('LLAMA_MODEL')
        self.embedding_model = os.getenv('EMBEDDING_MODEL')
        
        self.llm = OllamaFunctions(
            model=self.llm_model, 
            format='json',
            temperature=0, 
            prompt="Please return the output strictly as valid JSON"            
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100 )
        

        
        self.graph = Neo4jGraph(
                url=self.url,  
                username=self.username,
                password=self.password
            )
        
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            )
        
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        self.prompt_chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting relevant information from the provided text such as person, organization, business entities, etc. "
                    "Your task is to return information that answers the user's query accurately and concisely, based on the content of the text."
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )
        
        self.graph_documents = None
        self.vector_retriever = None
        self.initial_chain = None
        self.conversation_history = []
    
    
    def clear_graph(self):


        driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
        st.write("All nodes and relationships have been deleted from the graph.")
    
    
    def ingest(self, uploaded_file):
        self.clear_graph()
        
        # Save the uploaded file to a temporary file
        file_extension = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        if file_extension.lower() == ".pdf":
            docs = PyPDFLoader(file_path=tmp_file_path).load()
        elif file_extension.lower() == ".txt":
            docs = TextLoader(file_path=tmp_file_path).load()
        elif file_extension.lower() == ".docx":
            docs = Docx2txtLoader(file_path=tmp_file_path).load()
        elif file_extension.lower() == '.json':
            docs = JSONLoader(file_path=tmp_file_path, jq_schema='.', text_content=False).load()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
  
        chunks = self.text_splitter.split_documents(docs)
        
        self.graph_documents = self.llm_transformer.convert_to_graph_documents(documents=chunks)
        
        for i, graph_doc in enumerate(self.graph_documents):
            st.sidebar.write(f"Graph Document {i+1}:")
            st.sidebar.write(graph_doc)  
        
        self.graph.add_graph_documents(
            self.graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        
        vector_index = Neo4jVector.from_existing_graph(
            self.embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
            url = self.url,  
            username = self.username,
            password = self.password
        )
        
        self.vector_retriever = vector_index.as_retriever()
        
        entity_chain = self.llm.with_structured_output(Entities)
        
        self.initial_chain = self.prompt_chain | entity_chain
        
    def showGraph(self):

        driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))
        session = driver.session()

        query = "MATCH (n)-[r]->(m) RETURN n, r, m"
        results = session.run(query)
        net = Network(height="750px", width="100%", notebook=False, cdn_resources='in_line')

        nodes = set()
        for record in results:
            n = record['n']
            m = record['m']
            r = record['r']
            # Nodes
            n_id = n.id
            m_id = m.id
            n_props = dict(n.items())
            m_props = dict(m.items())
            if n_id not in nodes:
                nodes.add(n_id)
                node_label = n_props.get('name') or n_props.get('text') or n_props.get('id') or str(n_id)
                node_title = "<br>".join(f"{key}: {value}" for key, value in n_props.items())
                net.add_node(n_id, label=node_label, title=node_title)
            if m_id not in nodes:
                nodes.add(m_id)
                node_label = m_props.get('name') or m_props.get('text') or m_props.get('id') or str(m_id)
                node_title = "<br>".join(f"{key}: {value}" for key, value in m_props.items())
                net.add_node(m_id, label=node_label, title=node_title)
        
            r_type = r.type
            r_props = dict(r.items())
            edge_title = "<br>".join(f"{key}: {value}" for key, value in r_props.items())
            net.add_edge(n_id, m_id, label=r_type, title=edge_title)
        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0
            },
            "minVelocity": 0.75
          }
        }
        """)
        graph_html = net.generate_html(notebook=False)
        
        return graph_html
        
    # Collects the neighborhood of entities mentioned in the question
    def graph_retriever(self, question):

        result = ""
        entities = self.initial_chain.invoke(question)
        for entity in entities.names:
            response = self.graph.query(
                """
                CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit: 2})
                YIELD node, score
                CALL {
                  WITH node
                  MATCH (node)-[r:MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                } RETURN output LIMIT 50
                """,
                {"query": entity},
            )
            result += "\n".join([el['output'] for el in response])

        return result
    
    def full_retriever(self, question: str):
        graph_data = self.graph_retriever(question)
        vector_data = [el.page_content for el in self.vector_retriever.invoke(question)]
        final_data = f"""Graph data:
                    {graph_data}
                    vector data:
                    {"#Document ". join(vector_data)}
                    """
        return final_data


    def output(self, query, conversation_history):
        if not self.initial_chain:
            return "Please provide a PDF document"

        # Build the conversation history into the prompt
        conversation = ""
        for msg in conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation += f"{role}: {msg['content']}\n"

        template = """You are an assistant that answers questions based on the provided context and conversation history.

        Conversation history:
        {conversation}

        Context:
        {context}

        Current question:
        {question}

        Use natural language and be concise.

        Answer:"""


        prompt = ChatPromptTemplate.from_template(template)
        conversation_runnable = RunnableLambda(lambda x: conversation)
        
        output_chain =  (
                {
                    "conversation": conversation_runnable,
                    "context": self.full_retriever,  
                    "question": RunnablePassthrough(),
                }
            | prompt  
            | self.llm
            | StrOutputParser()  
            
            )
        try:
            answer = output_chain.invoke(query)
            return answer
        
        # If user asks a query that results in an empty entities.names
        except ValueError as ve:
            st.warning(f"ValueError occurred: {ve}. Passing query directly to LLM.")
            
            new_template = """You are an assistant that answers questions based on conversation history.

                Conversation history:
                {conversation}

                Current question:
                {question}

                Use natural language and be concise.

                Answer:"""
                
            new_prompt = ChatPromptTemplate.from_template(new_template)
            output_chain = (
                    {
                        "conversation": conversation_runnable,
                        "question": RunnablePassthrough(),
                    }
                | new_prompt  
                | self.llm
                | StrOutputParser() 
                
                )
            fallback_answer = output_chain.invoke(query)
            return fallback_answer 

        except Exception as e:
            st.error(f"An error occurred: {e}. Passing query directly to LLM.")
            new_template = """You are an assistant that answers questions based on conversation history.

                Conversation history:
                {conversation}

                Current question:
                {question}

                Use natural language and be concise.

                Answer:"""
                
            new_prompt = ChatPromptTemplate.from_template(new_template)
            output_chain = (
                    {
                        "conversation": conversation_runnable,
                        "question": RunnablePassthrough(),
                    }
                | new_prompt  
                | self.llm
                | StrOutputParser() 
            )      
            fallback_answer = output_chain.invoke(query)
            return fallback_answer
        

