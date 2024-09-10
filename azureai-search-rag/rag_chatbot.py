import os
import uuid
from typing import Dict, Tuple

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()

class RAGChatbot:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.rag_chain = None
        self.conversational_rag_chain = None
        self.store: Dict[str, ChatMessageHistory] = {}

    def initialize_components(self):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )

        self.vector_store = AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            embedding_function=self.embeddings.embed_query,
            additional_search_client_options={"retry_total": 4}
        )

        self.llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

    def create_rag_chain(self):
        retriever = self.vector_store.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
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

        system_prompt = (
            "You are an assistant for question-answering tasks, providing responses "
            "suitable for an expert audience. Use only the following pieces of retrieved "
            "context to answer the question. If the context doesn't contain relevant "
            "information, state that you don't have enough information to answer. "
            "Avoid answering without sources. When possible, provide a link or reference "
            "to the source of the information. Use technical language appropriate for "
            "experts in the field. Prioritize precision and depth over simplification. "
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        document_prompt = PromptTemplate.from_template("""
            Content: {page_content}
            Source: {source}
            """)
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt, document_prompt=document_prompt)

        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def chat(self, message: str, history: list, session_id: str = None) -> Tuple[str, Dict, str]:
        if session_id is None:
            session_id = str(uuid.uuid4())
        try:
            response = self.conversational_rag_chain.invoke(
                {"input": message},
                config={"configurable": {"session_id": session_id}}
            )
            return response["answer"], response["context"], session_id
        except Exception as e:
            print(f"Error in chat function: {e}")
            return "I'm sorry, but I encountered an error. Please try again.", {}, session_id

    def save_to_db(self):
        markdown_path = "./knowledgebase/"
        loader = DirectoryLoader(markdown_path, glob='./*.md', loader_cls=UnstructuredMarkdownLoader)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        self.vector_store.add_documents(documents=docs)

    def setup(self):
        self.initialize_components()
        self.create_rag_chain()
        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def clear_session(self, session_id: str):
        self.store.pop(session_id, None)
        return str(uuid.uuid4())

    def delete_previous(self, history: list, session_id: str):
        if len(history) > 0:
            history = history[:-1]
        if session_id in self.store:
            self.store[session_id].messages = self.store[session_id].messages[:-1]
        return history
