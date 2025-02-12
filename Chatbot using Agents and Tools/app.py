from langchain_huggingface import ChatHuggingFace
from langchain.agents import AgentExecutor,create_structured_chat_agent,create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun,DuckDuckGoSearchResults
from langchain_community.tools import YouTubeSearchTool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain import hub
import os
from langchain_groq import ChatGroq
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
import streamlit as st
from streamlit_chat import message
from datetime import datetime

# from langgraph.prebuilt import create_react_agent

os.environ["HUGGINGFACEHUB_API_TOKEN"]=""

llm = HuggingFaceEndpoint(
    repo_id="unsloth/Llama-3.2-1B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
)

chat_model = ChatHuggingFace(llm=llm)

duckduckgo_tool1 = DuckDuckGoSearchRun()
duckduckgo_tool2 = DuckDuckGoSearchResults()
YT = YouTubeSearchTool()

tools=[duckduckgo_tool1,duckduckgo_tool2,YT]

# Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/react")

# llm  = ChatGroq(temperature=0.3, groq_api_key="", model_name="mixtral-8x7b-32768")

system = '''You are a highly intelligent chatbot with real-time access to the latest updates and Your name is AgenticAIBot. You can leverage the tools provided to generate precise, well-thought-out answers. Follow the pipelines and structure defined below for every query.

---

### Tools Provided:
{tools}

---

### Pipeline for Answering Questions:
Follow this pipeline strictly to ensure accuracy and clarity in your response:

#### 1. **Question Understanding**:
   - Break down the input question to understand its core intent and requirements.
   - Identify relevant tools and data sources needed to solve the problem.

#### 2. **Strategy Development**:
   - Develop a step-by-step strategy to address the question.
   - Consider multiple approaches and choose the most effective one.

#### 3. **Iterative Processing**:
   - Use the following iterative loop to solve the problem efficiently:
     - **Thought**: Analyze the current state of the question or task.
     - **Action**: Choose the next logical action to take. Must be one of [{tool_names}].
     - **Action Input**: Provide necessary input to the selected tool.
     - **Observation**: Record and evaluate the output of the action.
   - Repeat this loop until you have gathered enough information to answer the question.

#### 4. **Finalization**:
   - **Final Thought**: Summarize your findings and ensure all steps have been completed successfully.
   - **Final Answer**: Provide the final answer in a clear, concise, and actionable format.

---

### Format to Follow:
Use the following structured format in your response:

1. **Question**: The input question you must answer.
2. **Thought**: Reflect on the question and decide the next step.
3. **Action**: Specify the tool you will use (from [{tool_names}]).
4. **Action Input**: Provide the input for the selected tool.
5. **Observation**: Record the result from the action.
6. **Thought**: Analyze the result and decide the next step.
... (Repeat Thought/Action/Action Input/Observation as needed)
7. **Final Thought**: Summarize and validate the final answer.
8. **Final Answer**: Present the ultimate solution to the input question.

---

### Additional Guidelines:
- Always validate your observations before moving to the next step.
- Cross-check answers when multiple tools are used.
- Be concise but provide enough detail to ensure clarity.
- If any step encounters an error, specify the error and retry or adjust your approach.

---

Begin!

Question: {input}
Thought: {agent_scratchpad}'''

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "What is your question?")
    ]
)

agent = create_react_agent(llm=llm,tools=tools,prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

# Streamlit app layout
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI Chatbot")
st.write("Ask me anything about Agentic AI or request YouTube video suggestions!")

# Custom CSS for modern styling
st.markdown("""
<style>
    body {
        background-color:rgb(227, 39, 136);
        font-family: 'Arial', sans-serif;
    }
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 20px;
        border: 1px solidrgb(227, 19, 19);
        border-radius: 10px;
        background-color:rgb(255, 255, 255);
        box-shadow: 0 4px 10px rgba(255, 255, 255, 0.1);
    }
    .user-message {
        text-align: right;
        color: #007bff;
        font-weight: bold;
    }
    .ai-message {
        text-align: left;
        color: #333;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        border: 1px solid #007bff;
        height: 60px;  /* Adjust height of the input box */
        font-size: 16px;  /* Adjust font size for better readability */
        padding: 10px;
    }
    .stButton>button {
        background-color:rgb(222, 158, 18);
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history in a container
with st.container():
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for i, chat in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User message
            message(chat, is_user=True, key=f"user_{i}")  # Align user's message to the right
        else:  # AI message
            message(chat, is_user=False, key=f"ai_{i}")  # Align AI's message to the left
    st.markdown("</div>", unsafe_allow_html=True)

# User input using chat_input
chat_input = st.chat_input("Type your message here...")

if chat_input:
    with st.spinner("Thinking..."):
        # Invoke the agent
        results = agent_executor.invoke({"input": chat_input})
        response = results['output']
        
        # Store the conversation in session state
        st.session_state.chat_history.append(chat_input)  # Store user input
        st.session_state.chat_history.append(response)    # Store AI response
        
        # Clear the input box and rerun the app
        st.rerun()
