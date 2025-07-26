# HSC Bangla Literature RAG API

This project is a Retrieval-Augmented Generation (RAG) system designed to answer questions about HSC-level Bangla literature. It processes a PDF of a Bangla story, extracts the text using OCR, and uses a conversational AI chain powered by Google's Gemini models to answer user queries based on the text.

---

## Features

- **PDF Processing**: Converts specified pages from a PDF into images for text extraction.
- **Advanced OCR**: Uses Tesseract with custom image pre-processing (sharpening, contrast) to accurately extract Bengali text.
- **Vector Storage**: Embeds text chunks and stores them in a ChromaDB vector store.
- **Conversational RAG Chain**: Uses LangChain and Google's Gemini models to generate answers based on retrieved context, maintaining chat history.
- **FastAPI Endpoint**: Exposes the conversational chain via an API endpoint, made publicly accessible with ngrok.

---

## Setup Guide

1.  **Clone the Repository**
    ```bash
    git clone [your-github-repo-url]
    cd [your-repo-name]
    ```

2.  **Install Dependencies**
    The system requires several Python packages and system libraries. You can install them using pip and apt.
    ```bash
    # Python Packages
    pip install langchain langchain-community langchain-google-genai chromadb llama-index pdf2image pytesseract unstructured tiktoken opencv-python-headless fastapi uvicorn pyngrok

    # System Libraries (for OCR and PDF processing)
    sudo apt-get update
    sudo apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-ben
    ```

3.  **Set Up API Keys**
    This project requires a Google Gemini API key and an ngrok authentication token.
    - In your Google Colab environment, store your Gemini API key as a secret named `GOOGLE_API_KEY_1`.
    - In the API code block, replace `"30Mp4YhMiIT6evuX74fObq8eISF_6dRn6Qz8sYvnBQBdMrUat"` with your ngrok authentication token.

4.  **Add Data**
    Place your PDF file in the `/content/` directory and ensure its name is `HSC26-Bangla1st-Paper.pdf`, as this path is hardcoded in the notebook.

5.  **Run the Application**
    Execute the cells in the Jupyter Notebook (`api_HSC_RAG.ipynb`) in sequential order. The final cell will start the FastAPI server and provide a public ngrok URL to access the API.

---

## Used Tools, Libraries, and Packages

-   **Core AI/ML**: `langchain`, `google-generativeai`, `llama-index`, `pytorch`
-   **Vector Database**: `chromadb`
-   **Web Framework**: `fastapi`, `uvicorn`
-   **PDF/Image Processing**: `pdf2image`, `pytesseract`, `opencv-python-headless`, `Pillow`
-   **System Tools**: `poppler-utils`, `tesseract-ocr`
-   **Tunneling**: `pyngrok`

---

## API Documentation

The API provides a single endpoint for interacting with the RAG chain.

### POST `/chat`

Handles a single turn in a conversation. It requires a session ID to maintain chat history.

**Request Body:**

```json
{
  "session_id": "string",
  "question": "string"
}
