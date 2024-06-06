import json
import streamlit as st
from streamlit_option_menu import option_menu
from embeddings import LocalEmbedding
from retrieval import Retrieval
from redis_client import RedisClient

def set_retrieval():
    print("Loading source files")
    # prompt = st.chat_input("Ask About NSX Cloud console and NSX ALB")
    source_files_path = "../guides"
    local_embedding_path = "../embedding_index"

    # le = LocalEmbedding(source_files_path, local_embedding_path, "FAISS", "SentenceTransformer")
    le = LocalEmbedding(source_files_path, local_embedding_path, "FAISS", "OpenAI")
    le.load_embeddings("Directory")
    # rt = Retrieval(le.vectorestore, "ChatOpenAI", "TheBloke/una-cybertron-7B-v2-GGUF", True)
    rt = Retrieval(le.vectorestore, "OpenAI", "gpt-3.5-turbo", True)
    rt.conversational_retrieval()
    return rt

def set_session_alias(session_id, redis_client):
    current_alias = redis_client.get_alias(f"alias:{session_id}")
    if current_alias == "New Chat":
        session_details = redis_client.get_session_details(session_id)
        if session_details:
            first_conversation = json.loads(session_details[-1])['data']
            alias = first_conversation['content']
            if len(alias) >= 30:
                alias = alias[:30]
            alias = f"{alias}-{session_id.split(":")[-1]}"
            redis_client.set_alias({f"alias:{session_id}": alias})
            st.session_state.default_selected = alias
            session_count = redis_client.increment_user_session_count(username)
            if session_count >= 5:
                # Delete session
                delete_session_id = f"message_store:{username}:{session_count - 5}"
                redis_client.delete(delete_session_id)
                # Delete alias
                alias = f"alias:{delete_session_id}"
                redis_client.delete(alias)
            st.rerun()

def render_sidebar(username, active_sessions_alias, redis_client):
    active_sessions_alias_keys = list(active_sessions_alias.keys()) + ["New Chat"]
    with st.sidebar:
        index = active_sessions_alias_keys.index(st.session_state.default_selected)
        selected = option_menu(
            menu_title="Active Sessions",
            options=active_sessions_alias_keys,
            manual_select=index,
            key="session_menu"
        )
        print(st.session_state.user_details)
        if selected == "New Chat":
            session_count = redis_client.get_session_count(username)
            session_id = f"message_store:{username}:{session_count}"
            redis_client.set_alias({f"alias:{session_id}": "New Chat"})
        else:
            session_id = str(active_sessions_alias[selected])
            st.session_state.default_selected = selected
        set_session_alias(session_id, redis_client)
        return session_id

def render_message_history(session_id):
    session_details = redis_client.get_session_details(session_id)
    if session_details:
        for message in session_details[::-1]:
            message_detail = json.loads(message)['data']
            with st.chat_message(message_detail["type"]):
                st.markdown(message_detail["content"])

if "retrieval" not in st.session_state:
    print("Creating retrieval")
    cv_rt = set_retrieval()
    st.session_state["retrieval"] = cv_rt
    print("Created retrieval")

if "user_details" not in st.session_state:
    st.title("Opcito HR Agent")
    with st.popover("Start Here"):
        st.markdown("Hello")
        name = st.text_input("What's your username?")
        print(f"name: {name}")
        if name:
            st.session_state.user_details = {}
            st.session_state.user_details["username"] = name
            st.session_state.default_selected = "New Chat"
            rd = RedisClient()
            st.session_state["redis_session"] = rd
            st.rerun()
        else:
            st.stop()
else:
    username = st.session_state.user_details["username"]
    st.title(f"Hello {username}, How I can help you?") 
    redis_client = st.session_state.redis_session
    active_sessions_list = redis_client.get_active_user_sessions(username)
    # active_sessions_alias = {redis_client.get_alias(f"alias:{active_session}") if redis_client.get_alias(f"alias:{active_session}") else "New Chat": active_session for active_session in active_sessions_list}
    # active_sessions_alias = {f"{redis_client.get_alias(f"alias:{active_session}")-{active_session.split(":")[-1]}}": active_session for index, active_session in enumerate(active_sessions_list)}
    active_sessions_alias = {}
    for index, active_session in enumerate(active_sessions_list):
        alias = redis_client.get_alias(f"alias:{active_session}")
        active_sessions_alias[alias] = active_session
    # Create the option list for last 5 sessions
    session_id = render_sidebar(username, active_sessions_alias, redis_client)

    render_message_history(session_id)
    session_number = int(session_id.split(":")[-1])

    # Accept user input
    if prompt := st.chat_input("Ask anything about Opcito HR Policies?"):
        # Display user message in chat message container
        answer = "I don't know"
        with st.chat_message("user"):
            st.markdown(prompt)
            answer = st.session_state.retrieval.rag.invoke(
                    {"input": prompt},
                    config={
                        "configurable": {"user_id": username, "conversation_id": str(session_number)}
                    },  # constructs a key "abc123" in `store`.
            )["answer"]

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write(answer)
        set_session_alias(session_id, redis_client)