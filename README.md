# Multi-Modal Question Answering with LangChain🦜🔗  & Streamlit

This project enables document,video and image question answering using LangChain and Streamlit. Users can upload PDF, DOCX files, or YouTube videos, and interact with the content by querying specific information. The project utilizes advanced language models to process documents, transcribe videos, and answer questions based on the content.

 ### The below tree structure of the multi-page Web App 
C:.
│   app.py
├───pages
│       1_Document_QA.py
│       2_Image_QA.py
│       3_Video_QA.py

## Tech Stack

- **Frontend:**
  - **Streamlit**: For building the web interface that allows users to upload documents and videos, and ask questions.
  
- **Backend:**
  - **LangChain**: I leveraged langchain for efficient multi-modal data ingestion , retrieval , ChatModel Integrations with Cerebras , Google Gemini , Mistral AI , HuggingFaceEmbeddings and Question-Answering pipelines creating a robust , and effective Multi-modal RAG.
  - **HuggingFace Embeddings**: Used for embedding document content into vector representations for similarity-based search.
  - **FAISS**: FAISS (Facebook AI Similarity Search) is an open-source library for fast and efficient similarity search and clustering of dense vectors. Optimized for large-scale data, it supports nearest neighbor searches, GPU acceleration, and customizable indexing 
      techniques. Widely used in recommendation systems, natural language processing, and large-scale machine learning tasks.
  - **Whisper**: An open-source model for automatic speech recognition (ASR) by open AI , that whenever a video files like (.mp4,.webm , .mp3 .etc ) are uploadd , they are transcribed into English Text required as a knowledge base for RAG Question Answering.
  
- **Language Models:**
  - **Google Gemini-2.0-flash**: A Multi-Modal model with the ability to parse multi-modal data such as complicated handwritten notes, graph plots , abstract paintings , documentations .
  - **open-mistral-nemo**: Another conversational model used to generate responses from document or video context.
  - **Llama-3.3-70b**: Used to answer questions based on document or video content.
  - **sentence-transformers/all-mpnet-base-v2**: This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
  
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
### 2. Install all the required dependencies 
```bash
pip install requirements.txt
```
### 3. Run the streamlit web app on localhost:8501
```python
streamlit run app.py
```
