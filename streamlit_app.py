import html
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ai_student_impact.rag import RagChatbot


def get_env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


st.set_page_config(page_title="Sagar's RAG Chatbot", layout="wide")

st.markdown(
    """
    <style>
    .chat-shell { display: flex; flex-direction: column; gap: 10px; }
    .chat-bubble { padding: 12px 14px; border-radius: 14px; max-width: 85%; box-shadow: 0 2px 8px rgba(0,0,0,0.08); white-space: pre-wrap; }
    .user-bubble { margin-left: auto; background: linear-gradient(135deg, #2563eb, #3b82f6); color: white; }
    .assistant-bubble { margin-right: auto; background: #f3f4f6; color: #111827; border: 1px solid #e5e7eb; }
    .chat-role { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.8; margin-bottom: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# LOGIN SYSTEM
# ============================================================================
def check_credentials(username: str, password: str) -> bool:
    """Validate user credentials from Streamlit Secrets."""
    try:
        # Get credentials from Streamlit Secrets
        valid_users = st.secrets.get("USERS", {})
        if valid_users and username in valid_users:
            return valid_users[username] == password
    except FileNotFoundError:
        st.error("❌ User credentials not configured. Contact admin.")
        return False
    
    return False


def show_login_page():
    """Display login form."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🔐 Sagar's RAG Chatbot")
        st.markdown("---")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", use_container_width=True):
            if check_credentials(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Welcome, {username}! 👋")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")


# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

# Show login page if not logged in
if not st.session_state["logged_in"]:
    show_login_page()
    st.stop()

# ============================================================================
# MAIN APP (Only visible after login)
# ============================================================================
st.title("Sagar's RAG Chatbot")

# Add logout button in top right
col1, col2 = st.columns([10, 1])
with col2:
    if st.button("🚪 Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["rag_chatbot"] = None
        st.session_state["chat_history"] = []
        st.success("Logged out successfully!")
        st.rerun()

st.markdown(f"*Logged in as: **{st.session_state['username']}***")
st.markdown("---")

with st.sidebar:
    st.header("Configuration")
    st.write(
        "Upload one or two dataset files and your OpenAI credentials to initialize the RAG chatbot. "
        "Supported formats are .csv, .xls, .xlsx, and .pdf. Files must be 20 MB or smaller."
    )

    uploaded_files = st.file_uploader(
        "Upload dataset files (max 2)",
        type=["csv", "xls", "xlsx", "pdf"],
        accept_multiple_files=True,
        help="Upload one or two CSV, Excel, or PDF dataset files, up to 20 MB each.",
    )

    dataset_path = None
    valid_upload = False
    saved_files = []
    if uploaded_files:
        if len(uploaded_files) > 2:
            st.error("Please upload at most 2 files.")
        else:
            upload_path = Path("uploaded_dataset")
            upload_path.mkdir(exist_ok=True)
            for uploaded_file in uploaded_files:
                if uploaded_file.size > 20 * 1024 * 1024:
                    st.error(f"{uploaded_file.name} exceeds the 20 MB limit.")
                    continue
                file_path = upload_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(file_path)
            if saved_files:
                dataset_path = str(upload_path)
                valid_upload = True
    else:
        st.info("Upload one or two dataset files to initialize the chatbot.")

    if valid_upload:
        for file_path in saved_files:
            suffix = file_path.suffix.lower()
            st.markdown(f"**Preview for {file_path.name}:**")
            try:
                if suffix == ".csv":
                    preview_df = pd.read_csv(file_path, nrows=5)
                    st.dataframe(preview_df)
                elif suffix in {".xls", ".xlsx"}:
                    preview_df = pd.read_excel(file_path, nrows=5)
                    st.dataframe(preview_df)
                elif suffix == ".pdf":
                    st.info("PDF upload received. The app will process it during initialization.")
                else:
                    st.warning("Preview not available for this file type.")
            except Exception as exc:
                st.warning(f"Unable to render preview for {file_path.name}: {exc}")

    max_rows = st.number_input("Max rows (CSV/Excel)", min_value=10, max_value=100, value=int(get_env("MAX_ROWS", "100")), step=10)
    openai_model = st.text_input("OpenAI model", value=get_env("OPENAI_MODEL", "gpt-5-mini"))
    
    # SECURE: Get API key from Streamlit Secrets (production) or environment (local)
    # Never ask user to input their API key in production!
    try:
        openai_key = st.secrets.get("OPENAI_API_KEY", "")
        if not openai_key:
            openai_key = get_env("OPENAI_API_KEY", "")
        if not openai_key:
            st.warning("⚠️ No OpenAI API key found. Set it in Streamlit Secrets for deployment.")
    except FileNotFoundError:
        openai_key = get_env("OPENAI_API_KEY", "")
        if not openai_key:
            st.warning("⚠️ No OpenAI API key configured.")

    if st.button("Save config"):
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["OPENAI_MODEL"] = openai_model
        os.environ["DATASET_PATH"] = dataset_path or ""
        os.environ["MAX_ROWS"] = str(max_rows)
        st.success("Settings are applied for this session.")

st.markdown("---")

if "rag_chatbot" not in st.session_state:
    st.session_state["rag_chatbot"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "initializing" not in st.session_state:
    st.session_state["initializing"] = False
if "init_error" not in st.session_state:
    st.session_state["init_error"] = None

if st.session_state["rag_chatbot"] is None:
    st.warning("Upload dataset files and enter your OpenAI API key, then initialize the chat.")
    if st.session_state["initializing"]:
        st.info("Initializing chatbot... This can take a moment.")
        with st.spinner("Loading dataset and building retrieval index..."):
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            if not valid_upload:
                st.session_state["init_error"] = "Cannot initialize: upload valid dataset files first."
            else:
                try:
                    st.session_state["rag_chatbot"] = RagChatbot(dataset_path=dataset_path, max_rows=max_rows, model=openai_model)
                    st.session_state["chat_history"] = []
                    st.session_state["init_error"] = None
                except Exception as exc:
                    st.session_state["init_error"] = f"Initialization failed: {exc}"
            st.session_state["initializing"] = False
            st.experimental_rerun()
    else:
        if st.button("Initialize chat", disabled=st.session_state["initializing"], use_container_width=True):
            st.session_state["initializing"] = True
            st.session_state["init_error"] = None
            st.experimental_rerun()

    if st.session_state["init_error"]:
        st.error(st.session_state["init_error"])

if st.session_state["rag_chatbot"]:
    st.subheader("Chat")
    cols = st.columns([3, 1])
    with cols[0]:
        user_input = st.text_input("Ask a question", key="user_input")
    with cols[1]:
        if st.button("Clear history"):
            st.session_state["chat_history"] = []
            st.experimental_rerun()

    if st.button("Send") and user_input:
        with st.spinner("Generating response..."):
            result = st.session_state["rag_chatbot"].ask(user_input)
        st.session_state["chat_history"].append({
            "user": user_input,
            "assistant": result["answer"],
            "sources": result["sources"],
        })
        st.experimental_rerun()

    if st.session_state["chat_history"]:
        st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
        for turn in st.session_state["chat_history"]:
            st.markdown(
                f"<div class='chat-bubble user-bubble'><div class='chat-role'>You</div>{html.escape(turn['user'])}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='chat-bubble assistant-bubble'><div class='chat-role'>Assistant</div>{html.escape(turn['assistant'])}</div>",
                unsafe_allow_html=True,
            )
            if turn["sources"]:
                st.markdown(
                    f"<div style='font-size:0.8rem; color:#4b5563; margin-left:6px;'>Sources: {html.escape(', '.join(turn['sources']))}</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No conversation history yet. Ask a question to begin.")
