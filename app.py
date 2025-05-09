import streamlit as st
import tempfile
import os
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import AgentExecutor, Tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Streamlit app layout
st.set_page_config(page_title="Local AI Assistant", layout="wide")
st.title("🤖 Local AI Assistant (Mistral via Ollama)")

# Sidebar for settings
st.sidebar.title("🧠 Settings")
use_memory = st.sidebar.checkbox("Enable Memory", value=True)
ollama_model = st.sidebar.selectbox(
    "Choose Model",
    ["mistral", "llama2", "gemma"],
    index=0,
    help="Must pull model first using 'ollama pull <model>'"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    # Initialize Ollama LLM (runs locally)
    llm = Ollama(
        model=ollama_model,
        temperature=0.5,
        top_p=0.9,
        num_ctx=2048
    )
    
    # Initialize memory if enabled
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) if use_memory else None
    
    # Tools setup (DuckDuckGo search)
    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="Web Search",
            func=search.run,
            description="Useful for searching the internet"
        ),
    ]

    # Define agent prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use tools when needed."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create agent
    agent = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | llm_with_tools
        | JSONAgentOutputParser()
    )

    # Create AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True
    )

    # File upload and processing
    uploaded_file = st.file_uploader("📄 Upload a PDF or Text File", type=["pdf", "txt"])
    if uploaded_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)
                
            pages = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(pages)
            
            st.success(f"✅ Document loaded successfully! ({len(docs)} chunks)")

            # Document Q&A
            st.subheader("Ask something about your document:")
            doc_question = st.text_input("🧾 Question about document")
            if doc_question:
                qa_chain = load_qa_chain(llm, chain_type="stuff")
                result = qa_chain.run(input_documents=docs, question=doc_question)
                st.session_state.messages.append(("You (Document)", doc_question))
                st.session_state.messages.append(("Assistant", result))
                st.success(result)

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Main chat interface
    st.subheader("💬 Ask Anything:")
    user_query = st.text_input("Type your message here...")
    if user_query:
        try:
            result = agent_executor.invoke({"input": user_query})
            st.session_state.messages.append(("You", user_query))
            st.session_state.messages.append(("Assistant", result["output"]))
            st.write(result["output"])
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")

    # Display chat history
    st.sidebar.markdown("### 📜 Chat History")
    for role, msg in st.session_state.messages:
        st.sidebar.markdown(f"**{role}:** {msg}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")