# Azure AI Search Vector Store

This repository contains a Python application that uses Azure AI to search a vector store with documents. The application leverages the power of OpenAI embeddings and Azure Search to provide similarity search, similarity search with relevance scores, and hybrid search capabilities. The application also provides a Gradio interface for user interaction.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Functions](#functions)
- [Gradio Interface](#gradio-interface)
- [License](#license)

## Prerequisites

- Python 3.8+
- An Azure account with access to Azure OpenAI and Azure Search

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/pajikos/ai-examples.git
    cd ai-examples/azureai-search-vectorstore
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. **Create a `.env` file:**
    ```bash
    cp .env.default .env
    ```

2. **Edit the `.env` file** with your Azure OpenAI and Azure Search credentials:
    ```env
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT=https://your-openai-endpoint.openai.azure.com/
    AZURE_OPENAI_API_KEY=your-openai-api-key
    AZURE_OPENAI_API_VERSION=2023-05-15
    AZURE_OPENAI_DEPLOYMENT=text-embedding-3-large

    # Vector Store Configuration
    AZURE_SEARCH_ENDPOINT=https://your-search-endpoint.search.windows.net
    AZURE_SEARCH_KEY=your-search-key
    AZURE_SEARCH_INDEX_NAME=langchain-vector-demo
    ```

## Usage

1. **Save documents to the database:** Uncomment the `save_to_db()` line in `main.py` and run the script.
    ```python
    if __name__ == "__main__":
        # Uncomment the following line to save documents to the database
        save_to_db()
        iface.launch()
    ```

    ```bash
    python hackaton/azureai-search-vectorstore/main.py
    ```

2. **Run the application:**
    ```bash
    python hackaton/azureai-search-vectorstore/main.py
    ```

3. **Access the Gradio interface:** Open your browser and go to the URL provided by Gradio (e.g., `http://127.0.0.1:7860`).

## Functions

### `save_to_db()`
Loads markdown files from the `./knowledgebase/` directory, splits them into smaller chunks, and adds them to the vector store.

### `similarity_search(query, k=3)`
Performs a similarity search on the vector store using the provided query and returns the top `k` results.

### `similarity_search_with_relevance_scores(query, k=3, score_threshold=0.5)`
Performs a similarity search on the vector store with relevance scores using the provided query and returns the top `k` results that meet the score threshold.

### `hybrid_search(query, k=3)`
Performs a hybrid search on the vector store using the provided query and returns the top `k` results.

## Gradio Interface

The Gradio interface provides a user-friendly way to interact with the document search functionality. The interface includes the following inputs:

- **Query:** A textbox to enter your search query.
- **Number of results:** A slider to select the number of results to return.
- **Search Type:** A radio button to choose between Similarity Search, Similarity Search with Relevance Scores, and Hybrid Search.
- **Score Threshold:** A slider to set the score threshold for Similarity Search with Relevance Scores.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
