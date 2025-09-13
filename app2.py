# app2_checked_api.py  (paste over your app2.py)
import streamlit as st
import os, tempfile, time, concurrent.futures, logging

import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.summarize import load_summarize_chain
from langchain_experimental.agents import create_pandas_dataframe_agent
from tenacity import retry, wait_random_exponential, stop_after_attempt

st.set_page_config(page_title="CSV AI (checked API)", layout="wide")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# -----------------------
# Utility: Validate API key by doing a tiny Groq ping (safe, low-cost)
# -----------------------
def validate_groq_key(key: str, model: str = "gemma2-9b-it"):
    """
    Performs a short, non-destructive call to Groq to validate the provided key.
    Returns (True, detail) on success, (False, exception) on failure.
    NOTE: requires the `openai` package (Groq-compatible client).
    """
    try:
        # import here to keep startup light if user doesn't want validation
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        # small test prompt, 1 token max
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True, resp
    except Exception as e:
        return False, e

# ----------------------- HOME -----------------------
def home_page():
    st.write("""Select a feature from the dropdown:
    - Chat with CSV
    - Summarize CSV
    - Analyze CSV
    """)

# ----------------------- RETRIEVER -----------------------
@st.cache_resource()
def retriever_func(uploaded_file):
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load()
        except Exception:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            data = loader.load()

        if len(data) == 0:
            st.error("CSV file is empty or invalid")
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        all_splits = text_splitter.split_documents(data)

        vectorstore = FAISS.from_documents(
            documents=all_splits,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    else:
        st.info("Please upload CSV documents to continue.")
        st.stop()
    return retriever, vectorstore

# ----------------------- CHAT -----------------------
def chat(temperature, model_name):
    st.write("# Talk to CSV")
    reset = st.sidebar.button("Reset Chat")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV here 👇:", type="csv")
    retriever, vectorstore = retriever_func(uploaded_file)

    # API key guard: ensure user provided, and set env var (for other libs that read it)
    if not user_api_key:
        st.error("Please enter your Groq API key first (sidebar).")
        st.stop()
    os.environ["GROQ_API_KEY"] = user_api_key  # keep env var in sync

    # IMPORTANT: explicitly pass api_key to ChatGroq to guarantee LangChain uses the exact key
    llm = ChatGroq(api_key=user_api_key, model=model_name, temperature=temperature, streaming=True)
        
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}] 

    store = {}
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Use the following pieces of context to answer the question.  
                      If you don't know, say you don't know. Context: {context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    runnable = prompt | llm
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    def chat_message():
        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            contextt = vectorstore.similarity_search(prompt, k=6)
            context = "\n\n".join(doc.page_content for doc in contextt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                text_chunk = ""
                with st.spinner("Thinking..."):
                    for chunk in with_message_history.stream(
                            {"context": context, "input": prompt},
                            config={"configurable": {"session_id": "abc123"}},
                        ):
                        text_chunk += chunk.content
                        message_placeholder.markdown(text_chunk)
                st.session_state.messages.append({"role": "assistant", "content": text_chunk})
        if reset:
            st.session_state["messages"] = []

    chat_message()

# ----------------------- SUMMARY -----------------------
def summary(model_name, temperature, top_p):
    st.write("# Summary of CSV")
    uploaded_file = st.file_uploader("Upload source document", type="csv", label_visibility="collapsed")
    if uploaded_file is None:
        return

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    file_size_kb = os.path.getsize(tmp_file_path) / 1024.0
    st.write(f"Uploaded file: **{uploaded_file.name}** — {file_size_kb:.1f} KB")

    # Load with CSVLoader
    load_start = time.time()
    try:
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
        data = loader.load()
    except:
        loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
        data = loader.load()
    load_time = time.time() - load_start
    st.write(f"Loaded {len(data)} docs. Load time: {load_time:.2f}s")

    if len(data) == 0:
        st.error("CSV file is empty")
        return

    split_start = time.time()
    # larger chunk_size reduces # of map requests => helps avoid rate limit
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)
    split_time = time.time() - split_start
    st.write(f"Chunks created: {len(texts)} (split time {split_time:.2f}s)")

    default_max = min(10, len(texts))
    max_chunks = st.number_input("Max chunks to use (quick mode)", 1, len(texts), default_max)
    use_texts = texts[:max_chunks]

    gen_sum = st.button("Generate Summary")
    if not gen_sum:
        return

    if not user_api_key:
        st.error("Please provide your Groq API key in the sidebar.")
        st.stop()
    os.environ["GROQ_API_KEY"] = user_api_key

    # pass api_key explicitly
    llm = ChatGroq(api_key=user_api_key, model=model_name, temperature=temperature)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        return_intermediate_steps=False,
        input_key="input_documents",
        output_key="output_text",
    )

    # retry only for rate-limit/network transient errors
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _invoke():
        return chain.invoke({"input_documents": use_texts})

    with st.spinner("Generating summary (this may retry on 429 rate limits)..."):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke)
                result = future.result(timeout=120)
        except concurrent.futures.TimeoutError:
            st.error("⏳ Model call timed out after retries. Try fewer chunks or a smaller model.")
            return
        except Exception as e:
            # If authentication error, show clear message
            msg = str(e)
            if "Invalid API Key" in msg or "invalid_api_key" in msg or "401" in msg:
                st.error("❌ Authentication failed: API key invalid. Please re-check the key in the sidebar.")
            else:
                st.error(f"❌ Model call failed: {e}")
            logger.exception("Model call failed")
            return

    if isinstance(result, dict) and "output_text" in result:
        st.success(result["output_text"])
    else:
        st.write(result)

# ----------------------- ANALYZE -----------------------
def analyze(temperature, model_name):
    st.write("# Analyze CSV")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV here 👇:", type="csv")
    if uploaded_file is None:
        return

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    df = pd.read_csv(tmp_file_path)

    if not user_api_key:
        st.error("Please enter your Groq API key.")
        st.stop()
    os.environ["GROQ_API_KEY"] = user_api_key

    llm = ChatGroq(api_key=user_api_key, model=model_name, temperature=temperature)
    agent = create_pandas_dataframe_agent(llm, df, agent_type="openai-tools", verbose=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
            
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about the CSV:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner("Analyzing CSV..."):
            try:
                msg = agent.invoke({"input": prompt, "chat_history": st.session_state.messages})
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                return
        st.session_state.messages.append({"role": "assistant", "content": msg["output"]})
        st.chat_message("assistant").write(msg["output"])

# ----------------------- MAIN -----------------------
def main():
    st.markdown("<h1 style='text-align:center'>🧠 CSV AI (Groq) — API-key checked</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center'>⚡️ Interact, Analyze & Summarize CSV Files with Groq Models!</h4>", unsafe_allow_html=True)

    global user_api_key
    raw_key = st.sidebar.text_input("#### Enter Groq API key 👇", placeholder="Paste your Groq API key here", type="password")
    user_api_key = raw_key.strip() if isinstance(raw_key, str) else raw_key

    if user_api_key:
        # Masked display to confirm we received it (never show the full key)
        masked = (user_api_key[:4] + "..." + user_api_key[-4:]) if len(user_api_key) > 8 else "****"
        st.sidebar.info(f"API key loaded: {masked}")
        # Add a manual validation button so user can confirm the key works
        if st.sidebar.button("Validate API key"):
            st.sidebar.info("Validating key with a tiny Groq ping...")
            ok, detail = validate_groq_key(user_api_key, model="gemma2-9b-it")
            if ok:
                st.sidebar.success("API key is valid — Groq replied ✅")
                logger.debug("API validation response: %s", detail)
            else:
                st.sidebar.error(f"API key invalid or network issue: {detail}")
                logger.exception("API validation failed")

    MODEL_OPTIONS = [
        "gemma2-9b-it",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "meta-llama/llama-guard-4-12b"
    ]
    model_name = st.sidebar.selectbox("Model", MODEL_OPTIONS, index=0)
    top_p = st.sidebar.slider("Top_P", 0.0, 1.0, 1.0, 0.1)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.01)

    functions = ["home", "Chat with CSV", "Summarize CSV", "Analyze CSV"]
    selected_function = st.selectbox("Select a functionality", functions)

    if selected_function == "home":
        home_page()
    elif selected_function == "Chat with CSV":
        chat(temperature=temperature, model_name=model_name)
    elif selected_function == "Summarize CSV":
        summary(model_name=model_name, temperature=temperature, top_p=top_p)
    elif selected_function == "Analyze CSV":
        analyze(temperature=temperature, model_name=model_name)

if __name__ == "__main__":
    main()
