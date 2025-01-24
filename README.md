# Multi-Modal Question Answering with LangChain🦜🔗  & Streamlit

This project enables document,video and image question answering using LangChain and Streamlit. Users can upload PDF, DOCX files, or YouTube videos, and interact with the content by querying specific information. The project utilizes advanced language models to process documents, transcribe videos, and answer questions based on the content.

## Tech Stack

- **Frontend:**
  - **Streamlit**: For building the web interface that allows users to upload documents and videos, and ask questions.
  
- **Backend:**
  - **LangChain**: A framework to manage document processing, embeddings, and question answering chains.
  - **HuggingFace Embeddings**: Used for embedding document content into vector representations for similarity-based search.
  - **FAISS**: A vector search engine used to index the document embeddings for fast retrieval.
  - **Whisper**: An open-source model for automatic speech recognition (ASR), used for transcribing video content.
  
- **Language Models:**
  - **Google Gemini**: A conversational model for answering questions based on document or video content.
  - **Mistral AI**: Another conversational model used to generate responses from document or video context.
  - **Cerebras Llama**: Used to answer questions based on document or video content.
  
- **Other Libraries:**
  - **PyMuPDF**: For loading and processing PDF files.
  - **YouTubeLoader**: For extracting transcripts from YouTube videos.
  - **faiss**: A library for fast vector search and indexing.
  - **dotenv**: For managing environment variables such as API keys.

## Features

1. **Document Question Answering:**
   - Upload PDF or DOCX files.
   - Documents are loaded, split into chunks, and embedded using HuggingFace embeddings.
   - FAISS vector store is used to index document embeddings for efficient search.
   - Users can query the document and get detailed, concise answers from the content.

2. **Video Question Answering:**
   - Upload videos (MP4, MOV, AVI) or provide a YouTube URL.
   - Video transcriptions are generated using Whisper.
   - Transcripts are processed and embedded similarly to documents.
   - Users can ask questions related to the video, and responses are generated based on the transcript content.

## How to Run

### 1. Clone the repository:
```bash
git clone https://github.com/Raghu6798/Multi_Modal_RAG.git

```
