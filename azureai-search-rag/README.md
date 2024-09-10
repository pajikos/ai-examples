# RAG Chatbot Documentation

## Overview

This repository contains a Retrieval-Augmented Generation (RAG) chatbot that leverages Azure AI services and LangChain libraries to provide accurate and contextually relevant answers to user queries. The chatbot is designed to retrieve relevant documents from a knowledge base and use them to generate precise answers, suitable for an expert audience.

## Prerequisites

1. **Python 3.8 or higher**
2. **Azure Account** with access to Azure OpenAI and Azure Search services.
3. **Required Python libraries** listed in the `requirements.txt` file.

## Folder Structure

```
azureai-search-rag/
├── .env.default
├── gradio_interface.py
├── rag_chatbot.py
├── requirements.txt
```

## Environment Configuration

The `.env.default` file contains the necessary configuration for Azure services and LangChain API. Before running the application, copy this file to `.env` and update the placeholders with actual values.

```bash
cp .env.default .env
```

Update the `.env` file with your Azure and LangChain credentials:

```plaintext
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-azure-openai-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS=text-embedding-3-large
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-latest

# Vector Store Configuration
AZURE_SEARCH_ENDPOINT=https://your-azure-search-endpoint.search.windows.net
AZURE_SEARCH_KEY=your-azure-search-key
AZURE_SEARCH_INDEX_NAME=langchain-vector-demo

LANGCHAIN_API_KEY=your-langchain-api-key
LANGCHAIN_TRACING_V2=false
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=HackingWiki
```

## Installation

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

## Loading Data to the Vector Store

To load documents into the vector store, follow these steps:

1. Initialize and set up the RAGChatbot.
2. Uncomment the `save_to_db` line in the script.
3. Run the script to save documents to the database.
4. Comment the `save_to_db` line again to avoid reloading the data.

```python
from rag_chatbot import RAGChatbot

rag_chatbot = RAGChatbot()
rag_chatbot.setup()

# Uncomment the following line to save documents to the database
rag_chatbot.save_to_db()

# Comment the line again after running to avoid reloading
# rag_chatbot.save_to_db()
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any bug fixes or enhancements.

## Contact

For any questions or issues, please contact the project maintainer.

---

Feel free to ask for any changes or additional content you want to include in the documentation.