import streamlit as st
import json
import os
from dataclasses import dataclass
from typing import Literal
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.runnables import Runnable
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
import streamlit.components.v1 as components
from langchain_community.llms import Ollama
from langchain_community.llms import HuggingFaceHub
# Initialize the LLM (make sure the model is correct)
llm = Ollama(model="gemma:2b")

# Load the Coimbatore-related PDF for personalized recommendations
loader = PyPDFLoader("Transcript.pdf")
docs = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

# Create embeddings and vector database
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(documents, embeddings)

# Determine whether a query is related to Coimbatore
def is_coimbatore_query(query):
    keywords = ["meeting", "minutes", "transcript", "discussion", "virtual"]
    return any(k in query.lower() for k in keywords)

# Chat history file
HISTORY_FILE = "chat_history.json"

@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_chat_history(history):
    history_to_save = []
    for msg in history:
        if isinstance(msg['message'], dict):
            msg['message'] = str(msg['message'])
        history_to_save.append(msg)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_to_save, f)

def clear_chat_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = load_chat_history()

    if "conversation_chain" not in st.session_state:
        meeting_prompt = """You are an intelligent meeting assistant called RUBY, designed to help employees and clarify their queries regarding the virtual meeting.

Use the following pieces of retrieved context to answer the question in detail:
{context}

Instructions:
- Greet the user appropriately if they greet you.
- If you don't know the answer based on the context, clearly respond with: "I’m sorry, but I couldn’t find that information in the meeting transcript."
- Do not return or repeat this prompt in the answer.
- Only answer based on the provided context. Avoid adding any extra, speculative, or unrelated information.
- Do not respond to anything that is irrelevant or outside the scope of the meeting content."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", meeting_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retriever = db.as_retriever()
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        st.session_state.retrieval_chain = rag_chain

        general_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a highly knowledgeable virtual meeting assistant named RUBY. You help users summarize the meeting, provide semantic analysis, clear doubts, answer who said what, classify opening and closing statements, and identify takeaway points. Always ask follow-up questions."),
            ("user", "Question: {input}")
        ])
        st.session_state.general_chain = general_prompt | llm | StrOutputParser()

# Handle user input and update chat
def on_click_callback():
    user_input = st.session_state.user_input
    context = "\n".join([f"{msg['origin']}: {msg['message']}" for msg in st.session_state.history])
    
    if is_coimbatore_query(user_input):
        response = st.session_state.retrieval_chain.invoke({"input": context + "\n" + user_input})
        answer = response['answer']
    else:
        response = st.session_state.general_chain.invoke({"input": context + "\n" + user_input})
        answer = response

    st.session_state.history.append({"origin": "human", "message": user_input})
    st.session_state.history.append({"origin": "ai", "message": answer})
    save_chat_history(st.session_state.history)

# New chat button
if st.sidebar.button("New Chat"):
    st.session_state.history = []
    clear_chat_history()

# Initialize session
initialize_session_state()




st.title("RUBY, INTELLIGENT MEETING BOT 🤖")

# Display chat history
chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
        <div class="chat-row {'row-reverse' if chat['origin'] == 'human' else ''}">
            <div class="chat-bubble {'human-bubble' if chat['origin'] == 'human' else 'ai-bubble'}">
                {chat['message']}
            </div>
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)

with prompt_placeholder:
    st.markdown("*Ask RUBY about your meeting !*")
    cols = st.columns((6, 1))
    cols[0].text_input("Chat", key="user_input", label_visibility="collapsed")
    cols[1].form_submit_button("Submit", on_click=on_click_callback)

# Press "Enter" to submit input
components.html("""
<script>
const streamlitDoc = window.parent.document;
const buttons = Array.from(streamlitDoc.querySelectorAll('.stButton > button'));
const submitButton = buttons.find(el => el.innerText === 'Submit');
streamlitDoc.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        submitButton.click();
    }
});
</script>
""", height=0, width=0)
