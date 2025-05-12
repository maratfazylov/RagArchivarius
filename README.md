# RagArchivarius# Project: RAG Archivist Bot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system designed to answer questions about the history of a university department (or any specific domain) based on a knowledge base of historical documents. It features a Telegram bot interface for user interaction and leverages both local and cloud-based Large Language Models (LLMs) for different stages of the process.

The system uses a local LLM (run via LM Studio) for initial query analysis and a powerful cloud LLM (GigaChat) to synthesize final answers based on retrieved context, complete with source citations.

## Features

*   **Retrieval-Augmented Generation (RAG):** Answers questions using information retrieved from a document knowledge base.
*   **Vector Database:** Utilizes Qdrant to store and search document chunks using vector embeddings.
*   **Semantic Search:** Employs Sentence Transformers (`paraphrase-multilingual-mpnet-base-v2`) for generating embeddings and finding semantically relevant context.
*   **Intelligent Query Analysis:** Uses a local LLM (e.g., Qwen via LM Studio) to:
    *   Classify user input as a question requiring search (`question`) or casual chat (`banter`).
    *   Extract personality names (`person_tag`) mentioned in the query, referencing a dynamically loaded list of known names from Qdrant.
    *   Refine the user query for better search results.
    *   Attempt to resolve pronouns using conversation history.
*   **Advanced Answer Generation:** Leverages Sber's GigaChat API to generate comprehensive answers based on the retrieved context and user query.
*   **Source Citation:** GigaChat is prompted to cite the source documents (filename, chunk index) for the information used in its answers.
*   **Dialog History Management:** Maintains conversation context for both the query analysis LLM and the answer generation LLM.
*   **Personality Filtering:** Filters Qdrant search results based on the `person_tag` extracted by the local LLM, improving relevance for person-specific queries.
*   **Telegram Bot Interface:** Provides a user-friendly chat interface via Telegram (`python-telegram-bot`).
*   **Console Interface:** Allows direct interaction with the RAG agent via the command line (`rag_agent.py`).
*   **Configuration:** Uses a `.env` file for managing sensitive credentials (Telegram token, GigaChat API keys).

## Architecture

1.  **User Input:** Received via Telegram bot or console.
2.  **Query Analysis (Local LLM):**
    *   The user query and recent dialog history are sent to the local LLM (via LM Studio API).
    *   A system prompt guides the LLM to classify the query (`question`/`banter`) and extract a `person_tag` (using a list of known names loaded from Qdrant).
    *   The LLM returns a JSON object with `query_type`, `person_tag`, and `refined_query`.
3.  **(If 'question'): Vector Search (Qdrant):**
    *   The `refined_query` is converted into a vector embedding.
    *   Qdrant is searched for relevant document chunks using the query vector.
    *   The search is filtered using the `person_tag` (if extracted).
4.  **(If 'question'): Answer Generation (GigaChat):**
    *   The retrieved document chunks (context), the original user query, dialog history, and a system prompt instructing GigaChat to act as an archivist and cite sources are sent to the GigaChat API.
    *   GigaChat generates the final answer.
5.  **(If 'banter'): Direct Response:** A simple, predefined friendly response is returned.
6.  **Output:** The generated answer (or banter response) is sent back to the user via Telegram or console.
7.  **History Update:** The conversation history is updated for the next turn.

## Core Components

*   **`rag_agent.py`:** Contains the core RAG logic, including functions for interacting with the local LLM, Qdrant, GigaChat, and orchestrating the response generation flow. Can be run directly for console interaction.
*   **`telegram_bot.py`:** Implements the Telegram bot interface, handles user sessions, initializes components, and calls `rag_agent.py`.
*   **`qdrant_loader.py` (Assumed):** A script (not detailed here, assumed to exist or be run separately) responsible for processing source documents, generating embeddings, and loading them into the Qdrant collection with appropriate metadata (like `source_file`, `chunk_index`, `person_tag`).
*   **Qdrant:** Vector database service. Needs to be running separately.
*   **LM Studio:** Desktop application used to download, configure, and run the local LLM, providing an OpenAI-compatible API endpoint. Needs to be running separately with the chosen model loaded.
*   **GigaChat API:** External cloud LLM service used for final answer generation.

## Setup

1.  **Prerequisites:**
    *   Python 3.10+
    *   `pip` package manager
    *   Git (optional, for cloning)
    *   Access to a running Qdrant instance.
    *   LM Studio installed and configured with a suitable local LLM (e.g., Qwen).
    *   GigaChat API credentials.
    *   Telegram Bot Token.

2.  **Clone the Repository (Optional):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

3.  **Create a Virtual Environment:**
    ```bash
    python -m venv arch_venv
    source arch_venv/bin/activate  # Linux/macOS
    # OR
    .\arch_venv\Scripts\activate  # Windows
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure you have a `requirements.txt` file listing all necessary packages: `requests`, `python-dotenv`, `qdrant-client`, `sentence-transformers`, `python-telegram-bot`, etc.)*

5.  **Configure Environment Variables:**
    *   Create a file named `.env` in the project root.
    *   Add the following variables:
        ```dotenv
        TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
        GIGACHAT_CLIENT_ID="YOUR_GIGACHAT_CLIENT_ID"
        GIGACHAT_CLIENT_SECRET="YOUR_GIGACHAT_CLIENT_SECRET"
        ```

6.  **Setup Qdrant:**
    *   Ensure your Qdrant instance is running (e.g., via Docker).
    *   Verify connection details (`QDRANT_HOST`, `QDRANT_GRPC_PORT` in `rag_agent.py`).
    *   Create the collection (`COLLECTION_NAME` in `rag_agent.py`) with the correct vector parameters (matching the embedding model) and payload index for `person_tag`.
    *   Run your data loading script (`qdrant_loader.py` or similar) to populate the Qdrant collection.

7.  **Setup LM Studio:**
    *   Launch LM Studio.
    *   Download and select the desired local LLM (e.g., Qwen).
    *   Configure the model (e.g., context length `n_ctx`). Ensure GPU acceleration is enabled if desired.
    *   Start the local API server (using the OpenAI API format).
    *   Note the IP address and Port of the server.
    *   Update `LOCAL_LLM_SERVER_IP` and `LOCAL_LLM_SERVER_PORT` constants in `rag_agent.py` accordingly.

## Running the Bot

1.  Ensure Qdrant and LM Studio (with the API server running) are active.
2.  Activate your virtual environment.
3.  Run the Telegram bot script:
    ```bash
    python telegram_bot.py
    ```
4.  Interact with your bot on Telegram.

## Running the Console Version

1.  Ensure Qdrant and LM Studio (with the API server running) are active.
2.  Activate your virtual environment.
3.  Run the RAG agent script directly:
    ```bash
    python rag_agent.py
    ```
4.  Interact with the agent in your console.

## Configuration Summary (`.env`)

*   `TELEGRAM_BOT_TOKEN`: Your Telegram bot's unique token.
*   `GIGACHAT_CLIENT_ID`: Your GigaChat application client ID.
*   `GIGACHAT_CLIENT_SECRET`: Your GigaChat application client secret.

*(Other configurations like Qdrant host/port, LM Studio host/port, model names, collection name are currently hardcoded as constants in `rag_agent.py` but could be moved to `.env` or a config file for more flexibility).*

## Potential Improvements

*   **Context Window Management:** Implement dynamic truncation or summarization of dialog history passed to LLMs to robustly handle context limits.
*   **More Robust `person_tag` Handling:** Improve LLM prompting or add post-processing logic for more reliable extraction and normalization of person names, especially handling different orders (FIO vs IOF) or partial names.
*   **Configuration File:** Move constants from `rag_agent.py` into a dedicated configuration file (e.g., `config.yaml`) or `.env`.
*   **Error Handling:** Enhance error handling for API calls and component initialization.
*   **Asynchronous Operations:** Make the Telegram bot fully asynchronous (e.g., using `asyncio` for potentially long-running LLM/Qdrant calls) to avoid blocking.
*   **Qdrant Management Interface:** Add tools or a simple UI for easier management and inspection of the Qdrant collection.
*   **Document Preprocessing Pipeline:** Formalize the document loading, chunking, and tagging process.
*   **Evaluation Framework:** Implement metrics and tests to evaluate the quality of retrieval and generation.