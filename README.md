# GraphRag ChatBot

GraphRag Chatbot is local chatbot that turns uploaded documents into a knowledge graph using Neo4j, Langchain, and Ollama

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/GraphRagChatbot.git
   cd GraphRagChatbot

2. Install Dependencies:

   ```bash
   pip install -r requirements.txt

3. Setup Envirnoment Variables:

   NEO4J_URI= Neo4j URI

   NEO4J_USERNAME= Neo4j username

   NEO4J_PASSWORD= Neo4j password

   LLAMA_MODEL= LLM model name

   EMBEDDING_MODEL= Embedding model name

4. Run Application:
      ```bash
    streamlit run app.py
