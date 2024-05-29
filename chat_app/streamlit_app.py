import streamlit as st
from embeddings import LocalEmbedding
from retrieval import Retrieval

st.title("Opcito HR Agent")


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

if "retrieval" not in st.session_state:
    print("Creating retrieval")
    cv_rt = set_retrieval()
    st.session_state["retrieval"] = cv_rt
    print("Created retrieval")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask anything about NSX ALB and NSX Cloud Console?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    answer = "I don't know"
    with st.chat_message("user"):
        st.markdown(prompt)
        answer = st.session_state.retrieval.rag.invoke(
                {"input": prompt},
                config={
                    "configurable": {"user_id": "user1", "conversation_id": "1"}
                },  # constructs a key "abc123" in `store`.
        )["answer"]

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # stream = st.session_state.retrieval.chat_with_app("user1", "1")
        response = st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# def load_files():
#     print("Loading source files")
#     prompt = st.chat_input("Ask About NSX Cloud console and NSX ALB")
#     source_files_path = "../guides"
#     local_embedding_path = "../embedding_index"

#     le = LocalEmbedding(source_files_path, local_embedding_path, "FAISS", "SentenceTransformer")
#     le.load_embeddings("Directory")
# load_files()