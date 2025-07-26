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
```

## Evaluation Matrix

The RAG pipeline was evaluated using the `ragas` framework on a small, curated dataset of questions and ground-truth answers. The results are as follows:

| Question | Answer | Ground Truth | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | অনুপমের ভাষায় সুপুরুষ বলা হয়েছে শস্তুনাথবাবুকে... | শম্ভুনাথ | 0.277 | 1.0 | 1.0 | 0.723 |
| কাকে অনুপমের/লেখকের ভাগ্যদেবতা বলে উল্লেখ করা হয়েছে? | অনুুপমের/লেখকের মামাকে তার ভাগ্যদেবতা বলে... | মামা | 0.267 | 1.0 | 1.0 | 0.800 |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | তথ্যসূত্র অনুযায়ী, কল্যাণীর বয়স ষোলো কি সতেরো... | পনেরো | 0.250 | 1.0 | 0.733 | 0.568 |

### Summary of Results:

The evaluation shows that the system is excellent at finding the correct information (**`context_recall`** = 1.0) and sticking to the facts it finds (**`faithfulness`** ≈ 1.0). However, the very low **`context_precision`** scores indicate that the retriever fetches too much irrelevant information, which negatively impacts the final **`answer_relevancy`**.

---
---

## Part 2: Answering the Required Questions

Here are the answers to the required questions, based on your notebook.

### **1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**

I used a two-step method for text extraction. First, I used the **`pdf2image`** library to convert specific pages of the PDF document into high-resolution PNG images. Then, I used the **`pytesseract`** library, which is a Python wrapper for Google's Tesseract-OCR Engine, to perform Optical Character Recognition (OCR) on these images and extract the Bengali text.

This approach was chosen because the source PDF is a scanned document, not a digitally native text file. The text cannot be simply copied and pasted, so OCR was the only viable method for extraction.

Yes, I faced significant formatting challenges. The initial OCR output was noisy and inaccurate. To solve this, I implemented an image pre-processing step using **OpenCV (`cv2`)** and **Pillow (`PIL`)**. I converted the images to a different color space, applied a sharpening kernel, and increased the contrast. This enhancement made the text clearer for Tesseract, which dramatically improved the accuracy of the extracted Bengali text.

### **2. What chunking strategy did you choose? Why do you think it works well for semantic retrieval?**

I chose a **character-limit-based** chunking strategy using LangChain's **`RecursiveCharacterTextSplitter`**. The configuration was set to a `chunk_size` of 1000 characters and a `chunk_overlap` of 200 characters.

This strategy works well for semantic retrieval for two main reasons:
1.  **Maintains Cohesion**: The `RecursiveCharacterTextSplitter` is ideal because it doesn't just blindly cut the text at the character limit. It first tries to split along natural semantic boundaries like paragraphs (`\n\n`), then sentences (`.`), and so on. This keeps related sentences together within a chunk, preserving their meaning.
2.  **Prevents Context Loss**: The `chunk_overlap` of 200 characters is crucial. If a sentence or important idea happens to fall at the boundary where a chunk is split, the overlap ensures that the complete thought is available in one of the chunks, preventing loss of context at the edges.

### **3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**

I used the **`models/text-embedding-004`** model from Google, accessed via the `GoogleGenerativeAIEmbeddings` class in LangChain.

I chose this model because it's a powerful and state-of-the-art embedding model designed by Google to be compatible with their Gemini family of generative models, which I used for the final answer generation. Using models from the same ecosystem often ensures better performance and compatibility. It also has strong multilingual capabilities, making it well-suited for processing the Bengali text of the source document.

This model captures the meaning of text by transforming it into a high-dimensional numerical vector. It is a transformer-based model that was pre-trained on a massive dataset, allowing it to understand complex semantic relationships, context, and nuance. Words and sentences with similar meanings are mapped to points that are close to each other in this vector space.

### **4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?**

The comparison is done through **vector similarity search**. Here's the process:
1.  **Storage**: The document chunks are converted into embeddings (vectors) and stored in a **ChromaDB** vector store. I used ChromaDB because it's lightweight, fast, and very easy to set up for prototyping, especially in an in-memory configuration as used in this project.
2.  **Comparison**: When a user's query comes in, it is also converted into an embedding using the *same* model. The system then performs a similarity search within ChromaDB to find the stored chunk vectors that are "closest" to the query vector. The underlying mathematical method for this comparison is typically **Cosine Similarity**, which measures the angle between two vectors to determine how similar they are in meaning.

I chose this vector search approach because it is the standard and most effective method for semantic retrieval. It goes beyond simple keyword matching and finds chunks that are contextually and semantically related to the query, which is essential for a high-quality RAG system.

### **5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?**

Meaningful comparison is ensured by a fundamental principle: **using the exact same embedding model (`text-embedding-004`) to convert both the document chunks during storage and the incoming user questions at query time.** This guarantees that both the documents and the questions are mapped into the same high-dimensional vector space. When they exist in the same "semantic space," the distance and angle between their vectors are a meaningful representation of their contextual similarity.

If a query is vague or missing context (e.g., "tell me more about his thoughts"), its resulting vector will be generic. When the retriever searches for the closest vectors in the database, it will likely retrieve chunks that are also general in nature or an average of several topics mentioned in the text. This "garbage in, garbage out" principle means the retrieved context will be poor, leading the final LLM to either:
-   Provide a very generic, unhelpful answer.
-   State that it cannot answer the question based on the information provided.
-   Hallucinate an answer based on the muddled context.

### **6. Do the results seem relevant? If not, what might improve them?**

Based on the evaluation matrix, the results are **partially relevant but have clear areas for improvement.**

The `context_recall` score of **1.0** is excellent, indicating the retriever successfully finds the correct documents. However, the `context_precision` score is very low (around **0.25-0.28**), which means it's retrieving a lot of irrelevant information alongside the correct chunks. This "noise" negatively impacts the final answer, as shown by the mediocre `answer_relevancy` scores and one factually incorrect answer about Kalyani's age.

Several things could improve the results:
1.  **Reduce the `k` Value**: The single most impactful change would be to **reduce the number of retrieved documents**. The retriever is set to fetch `k=15` chunks, which is too high and floods the context with noise. Reducing this to a smaller number, like **`k=3` or `k=5`**, would likely improve precision dramatically.
2.  **Improve Chunking Strategy**: Experimenting with a smaller `chunk_size` could create more focused and semantically distinct chunks, making it easier for the retriever to pick only the most relevant ones.
3.  **Prompt Engineering**: The system prompt could be enhanced with an instruction telling the LLM to more aggressively ignore irrelevant information within the provided context when formulating its answer.
