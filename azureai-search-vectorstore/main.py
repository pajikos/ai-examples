import os
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
import gradio as gr

# Load environment variables
load_dotenv()

# Initialize embeddings and vector store
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    embedding_function=embeddings.embed_query,
    additional_search_client_options={"retry_total": 4}
)

def save_to_db():
    markdown_path = "./knowledgebase/"
    loader = DirectoryLoader(markdown_path, glob='./*.md', loader_cls=UnstructuredMarkdownLoader)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    vector_store.add_documents(documents=docs)

def similarity_search(query, k=3):
    results = vector_store.similarity_search(
        query=query,
        k=k,
        search_type="similarity"
    )
    return format_results(results)

def similarity_search_with_relevance_scores(query, k=3, score_threshold=0.5):
    results = vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=k,
        score_threshold=score_threshold
    )
    return format_results_with_scores(results)

def hybrid_search(query, k=3):
    results = vector_store.similarity_search(
        query=query,
        k=k,
        search_type="hybrid"
    )
    return format_results(results)

def format_results(results):
    formatted = "# Search Results\n\n"
    for i, result in enumerate(results, 1):
        formatted += f"## Result {i}\n\n"
        formatted += f"**Content:** {result.page_content}\n\n"
        formatted += f"**Source:** {result.metadata.get('source', 'Unknown')}\n\n"
        formatted += "---\n\n"
    return formatted

def format_results_with_scores(results):
    formatted = "# Search Results\n\n"
    for i, (doc, score) in enumerate(results, 1):
        formatted += f"## Result {i}\n\n"
        formatted += f"**Content:** {doc.page_content}\n\n"
        formatted += f"**Source:** {doc.metadata.get('source', 'Unknown')}\n\n"
        formatted += f"**Relevance Score:** {score:.4f}\n\n"
        formatted += "---\n\n"
    return formatted

# ... (keep the rest of the functions)

def gradio_search(query, num_results, search_type, score_threshold=0.5):
    k = int(num_results)
    if search_type == "Similarity Search":
        return similarity_search(query, k)
    elif search_type == "Similarity Search with Relevance Scores":
        return similarity_search_with_relevance_scores(query, k, score_threshold)
    elif search_type == "Hybrid Search":
        return hybrid_search(query, k)
    else:
        return "Invalid search type selected"

iface = gr.Interface(
    fn=gradio_search,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your search query here..."),
        gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Number of results"),
        gr.Radio(["Similarity Search", "Similarity Search with Relevance Scores", "Hybrid Search"], label="Search Type"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Score Threshold (for Similarity Search with Relevance Scores)")
    ],
    outputs=gr.Markdown(label="Search Results"),
    title="Document Search",
    description="Enter a query to search the document database using different search methods.",
)

if __name__ == "__main__":
    # Uncomment the following line to save documents to the database
    # save_to_db()

    iface.launch()
