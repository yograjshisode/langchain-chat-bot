from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import (
    ConfigurableFieldSpec
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from typing import List

class Retrieval:
    def __init__(self, vstore, llm_type, llm_model, enable_chat_history=False):
        self.vstore = vstore
        self.llm_model = llm_model
        self.llm = self._get_llm(llm_type)
        self.enable_chat_history = enable_chat_history
        self.store = {}
        self.rag = None
    
    def _get_llm(self, llm_type):
        if llm_type == "OpenAI":
            return OpenAI()
        elif llm_type == "ChatOpenAI":
            return ChatOpenAI(
                openai_api_key="edhjwef",
                base_url="http://localhost:1234/v1",
                temperature=0.7,
                max_tokens=500,
                model=self.llm_model
            )
        else:
            raise Exception(f"LLM of type {llm_type} not supportted")
    
    def set_memory():
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        memory.load_memory_variables({})

    def conversational_retrieval(self):
        retriever = self.vstore.as_retriever()
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is. """
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )
        qa_system_prompt = """You are an assistant for question-answering tasks. \
            Use only following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            Use three sentences maximum and keep the answer concise.\

            {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        self.rag = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="Conversation ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
            ]
        )
    
    def chat_with_app(self, user_id, conversession_id):
        while True:
            user_input = input("Ask a question. Type 'exit' to quit.\n>")
            if user_input=="exit":
                break  
            answer = self.rag.invoke(
                {"input": user_input},
                config={
                    "configurable": {"user_id": user_id, "conversation_id": conversession_id}
                },  # constructs a key "abc123" in `store`.
            )["answer"]
            print("AI:", answer)

    def get_session_history(self, user_id: str, conversation_id: str) -> BaseChatMessageHistory:
        if (user_id, conversation_id) not in self.store:
            self.store[(user_id, conversation_id)] = InMemoryHistory()
        return self.store[(user_id, conversation_id)]

# For production use cases, you will want to use a persistent implementation
# of chat message history, such as ``RedisChatMessageHistory``.
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        if len(self.messages) >= 10:
             self.messages.pop(0) # remove the oldest one
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []
