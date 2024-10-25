import streamlit as st
import os
import streamlit.components.v1 as components
from graphrag import GraphRag  

def main():
    st.title("💬 GraphRag Chatbot")

    if 'graph_rag' not in st.session_state:
        st.session_state['graph_rag'] = GraphRag()
        st.session_state.document_ingested = False
        
        
    uploaded_file = st.sidebar.file_uploader("Drag and drop or click to upload a document file")

    if uploaded_file is not None:
        st.session_state['graph_rag'].ingest(uploaded_file)
        st.sidebar.success("document file has been ingested and processed.")
        st.session_state.document_ingested = True


        
    if st.session_state.document_ingested:
        st.write("Visualizing the Knowledge Graph...")
        try:
            graph_html = st.session_state.graph_rag.showGraph()
            components.html(graph_html, height=800)
        except Exception as e:
            st.error(f"An error occurred while displaying the graph: {e}")
            
        st.header("Chat with the Knowledge Graph")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you today?"}]

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").write(msg["content"])
        else:
            st.chat_message("assistant", avatar="🤖").write(msg["content"])

    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="🧑‍💻").write(prompt)

        response = st.session_state['graph_rag'].output(prompt, st.session_state["messages"])

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant", avatar="🤖").write(response)

if __name__ == "__main__":
    main()

