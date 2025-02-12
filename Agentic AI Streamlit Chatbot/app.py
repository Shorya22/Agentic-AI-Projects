import os
import streamlit as st
from langchain_community.document_loaders import RecursiveUrlLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchResults
from huggingface_hub import login
import time

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv("HF_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Login to HuggingFace
login(token=huggingface_api_key)

# Set up embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit Application Setup
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖", layout="wide")

# Apply custom styling for chatbot interface
st.markdown("""
    <style>
        .chat-container {
            background-color: #f0f4f8;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .message {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .user-message {
            background-color: #d1e7ff;
            text-align: right;
            font-size: 20px;
        }
        .agent-message {
            background-color: #e9f7ef;
            text-align: left;
            font-size: 20px;
        }
        .clear-button {
            background-color: #ff4b5c;
            color: white;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
        }
        .clear-button:hover {
            background-color: #f44336;
        }
        .stButton>button {
            background-color: #ff4b5c;
            color: white;
            padding: 12px 30px;
            border-radius: 6px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #f44336;
        }
        .chat-input {
            background-color: #f1f1f1;
            border-radius: 10px;
            padding: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit header
st.title("Chat with Streamlit Chatbot 🤖")
st.markdown("Ask anything related to Streamlit! The bot remembers the conversation and fetches relevant information.")

# Load the vectorstore only once
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = FAISS.load_local(
        folder_path=r"D:\Streamlit Chatbot\streamlit_doc_vectors",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

# Create retriever from the loaded vectorstore
retriever = st.session_state.vectorstore.as_retriever()

# Set up memory
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3, return_messages=True)

# Initialize the Groq model for LLM
if 'llm' not in st.session_state:
    st.session_state.llm = ChatGroq(temperature=0.3, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Load the custom prompt
if 'prompt' not in st.session_state:
    st.session_state.prompt = hub.pull("arklau/react-chat")

# Initialize tools
if 'tools' not in st.session_state:
    search_tool = DuckDuckGoSearchResults()
    rag_tool = create_retriever_tool(retriever=retriever, name="RAG Pipeline", description="Having knowledge base of Streamlit.")
    st.session_state.tools = [rag_tool, search_tool]

# Create the agent
if 'agent' not in st.session_state:
    agent = create_react_agent(llm=st.session_state.llm, tools=st.session_state.tools, prompt=st.session_state.prompt)
    st.session_state.agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=st.session_state.tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)

# Create a text box for user input
user_input = st.chat_input("Ask a question:")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display conversation history in a container
with st.container():
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            message_class = "user-message" if message["role"] == "User " else "agent-message"
            st.markdown(f'<div class="message {message_class}"><strong>{message["role"]}:</strong> {message["content"]}</div>', unsafe_allow_html=True)

# Handle user input and response
if user_input:
    # Add user input to the chat history
    st.session_state.chat_history.append({"role": "User ", "content": user_input})

    # Display user's message first (this helps in smooth conversation flow)
    st.markdown(f'<div class="message user-message"><strong>User:</strong> {user_input}</div>', unsafe_allow_html=True)

    # Show the "thinking" indicator
    with st.spinner("Agent is thinking..."):
        # Create an empty container for streaming response
        response_container = st.empty()

        # Invoke the agent and get the response (use streaming here)
        response = st.session_state.agent_executor.invoke({"input": user_input, "chat_history": st.session_state.chat_history})

        # Stream the response character by character
        final_response = response['output']
        streamed_response = ""
        
        # This simulates the character-by-character response streaming
        for char in final_response:
            streamed_response += char
            response_container.markdown(f'<div class="message agent-message"><strong>Agent:</strong> {streamed_response}</div>', unsafe_allow_html=True)
            time.sleep(0.05)  # Adjust the speed of streaming here

        # Add the agent's final response to the chat history
        st.session_state.chat_history.append({"role": "Agent", "content": final_response})

# Optional: Add a button to clear conversation history
if st.button("Clear Conversation", key="clear"):
    st.session_state.chat_history = []  # Clear the session state history

