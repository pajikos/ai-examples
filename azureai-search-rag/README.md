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

## Running the Chatbot

### Step 1: Initialize the Chatbot

First, initialize and set up the RAGChatbot by running the following code:

```python
from rag_chatbot import RAGChatbot

rag_chatbot = RAGChatbot()
rag_chatbot.setup()

# Uncomment the following line to save documents to the database
# rag_chatbot.save_to_db()
```

### Step 2: Launch the Gradio Interface

To launch the Gradio interface for the chatbot, run:

```python
import gradio as gr
from rag_chatbot import RAGChatbot
from gradio_interface import create_gradio_interface

if __name__ == "__main__":
    rag_chatbot = RAGChatbot()
    rag_chatbot.setup()

    demo = create_gradio_interface(rag_chatbot)
    demo.launch()
```

## File Descriptions

### `rag_chatbot.py`

This file contains the main class `RAGChatbot` which handles the initialization of Azure services, creation of RAG chains, and interaction with the user.

Key Methods:
- `initialize_components()`: Initializes Azure OpenAI and Azure Search components.
- `create_rag_chain()`: Creates the Retrieval-Augmented Generation chain.
- `get_session_history(session_id: str)`: Retrieves chat history for a session.
- `chat(message: str, history: list, session_id: str = None)`: Handles user interaction and responses.
- `save_to_db()`: Loads and processes documents into the vector store.
- `setup()`: Sets up the components and chains.
- `clear_session(session_id: str)`: Clears the session history.
- `delete_previous(history: list, session_id: str)`: Deletes the last message in the chat history.

### `gradio_interface.py`

This file contains the code to create a Gradio interface for the chatbot. It defines the layout and interaction logic for the user interface.

Key Functions:
- `create_gradio_interface(chatbot: RAGChatbot)`: Creates and returns the Gradio interface.
- `user(user_message, history)`: Handles user input.
- `bot(history, session_id)`: Generates a response from the chatbot.
- `clear_session(session_id_str)`: Clears the chat session.
- `delete_previous(history, session_id)`: Deletes the previous message from the chat history.

### `.env.default`

This file contains the default environment variables required for the application to run. It includes placeholders for Azure and LangChain API keys and endpoints.

### `requirements.txt`

This file lists the Python libraries required to run the application.

```plaintext
azure-identity==1.17.1
azure-search-documents==11.5.1
backports.tarfile==1.2.0
gradio==4.43.0
importlib-metadata==8.0.0
jaraco.text==3.12.1
langchain-community==0.2.16
langchain-openai==0.1.23
markdown==3.7
python-dotenv==1.0.1
tomli==2.0.1
unstructured==0.15.9
```

## Usage

1. Start the chatbot by running the Gradio interface script.
2. Interact with the chatbot via the Gradio web interface.
3. Use the "Clear" button to clear the chat history.
4. Use the "Delete Previous" button to delete the last message.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any bug fixes or enhancements.

## Contact

For any questions or issues, please contact the project maintainer.