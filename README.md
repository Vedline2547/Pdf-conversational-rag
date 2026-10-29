# PDF Conversational RAG with Ollama and Hugging Face

## Overview
This project implements a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about PDF documents and receive context-aware answers.

The application uses **Gemma 3:1B** through Ollama for local answer generation, Hugging Face sentence-transformer embeddings for semantic representation, and FAISS for efficient similarity-based document retrieval. LangChain connects the components into a conversational question-answering pipeline.

## Features
- **PDF Loading:** Extracts text from PDF documents using PyPDFLoader.
- **Text Chunking:** Splits documents into manageable chunks using RecursiveCharacterTextSplitter.
- **Text Embeddings:** Generates 384-dimensional embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Search:** Stores and retrieves relevant document chunks using FAISS.
- **Local LLM Inference:** Uses Gemma 3:1B through Ollama to generate answers locally.
- **Conversational Retrieval:** Rephrases follow-up questions into standalone questions using conversation history.
- **Source Documents:** Returns retrieved document chunks to help inspect the context behind generated answers.

## Technologies Used
- Python
- LangChain
- Ollama
- Gemma 3:1B
- Hugging Face Sentence Transformers
- FAISS
- PyPDF
- LangChain Community Integrations

## How It Works
1. Load a PDF document and extract its text.
2. Split the extracted text into overlapping chunks.
3. Convert each chunk into a numerical embedding.
4. Store the embeddings in a FAISS vector database.
5. Convert the user's question into a retrieval query, using conversation history when necessary.
6. Retrieve relevant document chunks through similarity search.
7. Pass the retrieved context to Gemma 3:1B to generate an answer.
8. Return the answer and source documents.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Pdf-conversational-rag.git
cd pdf-conversational-rag
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

Activate it on Windows:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install langchain-classic langchain-community langchain-core langchain-text-splitters langchain-ollama langchain-huggingface faiss-cpu pypdf sentence-transformers
```

### 4. Install and start Ollama
Download Ollama from https://ollama.com/download.

Pull the model:
```bash
ollama pull gemma3:1b
```

### 5. Add your PDF
Place your PDF in the project directory and update the file path in `main.py` to match its location.

### 6. Run the application
```bash
python main.py
```

## Example Query
```python
query = "What are the acknowledgments in this paper?"
```

The application retrieves relevant PDF passages and uses the language model to generate an answer.

## Potential Applications
- Research paper exploration
- Academic document question answering
- Technical report analysis
- Document-based information retrieval
- Personal PDF knowledge assistants

## Limitations
- Answer quality depends on document text extraction and retrieval quality.
- Scanned PDFs may require OCR before their contents can be retrieved.
- Small language models may struggle with complex reasoning or long context.
- Retrieved passages should be checked when factual accuracy is important.

## Future Improvements
- Add a user interface using Streamlit or Gradio.
- Support multiple PDF uploads.
- Preserve conversation history across multiple questions.
- Display source page numbers alongside answers.
- Evaluate retrieval quality and answer accuracy.
- Add persistent vector storage to avoid re-embedding documents after every restart.

## Learning Outcomes
This project demonstrates practical experience with Retrieval-Augmented Generation, semantic embeddings, vector databases, document processing, local LLM inference, and conversational question answering.

## License
Choose a license appropriate for your project before publishing it.
