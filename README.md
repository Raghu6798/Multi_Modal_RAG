# Multi-Modal Retrieval Augmented Generation

This project enables document,video and image question answering using LangChain and Streamlit. Users can upload PDF, DOCX files, or YouTube videos, and interact with the content by querying specific information. The project utilizes advanced language models to process documents, transcribe videos, and answer questions based on the content.

 ### The below tree structure of the multi-page Web App 
 ```
C:
│   app.py
├───pages
│       1_Document_QA.py
│       2_Image_QA.py
│       3_Video_QA.py
```
## Tech Stack

- **Frontend:**
  - **Streamlit**: For building the web interface that allows users to upload documents and videos, and ask questions.
  
- **Backend:**
  - **LangChain**: I leveraged langchain for efficient multi-modal data ingestion , retrieval , ChatModel Integrations with Cerebras , Google Gemini , Mistral AI , HuggingFaceEmbeddings and Question-Answering pipelines creating a robust , and effective Multi-modal RAG.
  - **HuggingFace Embeddings**: Used for embedding document content into vector representations for similarity-based search.
  - **FAISS**: FAISS (Facebook AI Similarity Search) is an open-source library for fast and efficient similarity search and clustering of dense vectors. Optimized for large-scale data, it supports nearest neighbor searches, GPU acceleration, and customizable indexing 
      techniques. Widely used in recommendation systems, natural language processing, and large-scale machine learning tasks.
    ![image](https://github.com/user-attachments/assets/f292ed42-cc6b-4b15-a520-fb781dccad19)

  - **Whisper**: An open-source model for automatic speech recognition (ASR) by open AI , that whenever a video files like (.mp4,.webm , .mp3 .etc ) are uploadd , they are transcribed into English Text required as a knowledge base for RAG Question Answering.
  - ![Uploading image.png…]()

  
- **Language Models:**
  - **Google Gemini-2.0-flash**: A Multi-Modal model with the ability to parse multi-modal data such as complicated handwritten notes, graph plots , abstract paintings , documentations .

   ![image](https://github.com/user-attachments/assets/237ba3c0-0b1c-439b-9433-85be21a4e7fb)
  - **open-mistral-nemo**: Another conversational model used to generate responses from document or video context.
  - ![image](https://github.com/user-attachments/assets/8ddfcfa0-247b-4c7d-9160-7532e97920cc)
  - **Llama-3.3-70b**: Used to answer questions based on document or video content.
  - ![image](https://github.com/user-attachments/assets/a9b35ce1-777e-4821-b22d-9bdfd20396a7)
  - **sentence-transformers/all-mpnet-base-v2**: This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
  - ![image](https://github.com/user-attachments/assets/37d2a86e-7c4a-4e38-8264-a783d7843b52)

 
     ```plaintext
     GOOGLE_AI_STUDIO_API_KEY=your_google_ai_studio_api_key
     CEREBRAS_API_KEY=your_cerebras_api_key
     MISTRAL_API_KEY = your_mistral_api_key
     ```
![image](https://github.com/user-attachments/assets/9174fce6-5995-46f4-b89b-09f99f943a9f)

## Features

1. **Document Question Answering:**
   - Upload PDF or DOCX files.
   - Documents are loaded, split into chunks, and embedded using HuggingFace embeddings.
   - FAISS vector store is used to index document embeddings for efficient search.
   - Users can query the document and get detailed, concise answers from the content.

2. **Video Question Answering:**
   - Upload videos (MP4, MOV, AVI) or provide a YouTube URL.
   - ![Screenshot 2025-01-24 181657](https://github.com/user-attachments/assets/2f64110f-4874-42e9-85b6-1a6a041de284)
   - Video transcriptions are generated using Whisper.
   - Transcripts are processed and embedded similarly to documents.
   - ![Screenshot 2025-01-24 191312](https://github.com/user-attachments/assets/45c22aed-9064-4511-8946-a08710073fd5)

   - Users can ask questions related to the video, and responses are generated based on the transcript content.
   - ![image](https://github.com/user-attachments/assets/91859835-60f1-41e8-a97b-05e19cec5d0a)


3. **Image Input Options:**
   - Users can either **upload an image file** (formats: `.jpeg`, `.jpg`, `.png`) or **provide an image URL** to analyze its content.
     ![image](https://github.com/user-attachments/assets/bcc15a69-9580-4316-be39-7516ca4f4d14)





2. **Versatile Image Analysis:**
   - Supports a wide range of image types, including:
     - **General Imagery**: Objects, people, and natural scenes.
     - **Handwritten Notes**: Analyzes and interprets handwritten content.
     - **Research Work and Academic Papers**: Extracts insights from complex diagrams, graphs, and text-based images.
     - **Medical Images**: Handles X-Rays, MRIs, and other medical imaging for content analysis.
     - **Data Visualizations**: Processes charts, graphs, and other visual data representations.

3. **Multi-Modal AI for Enhanced Understanding:**
   - Gemini-2.0 Flash does an in-depth image content analysis and generate contextual answers to user queries.
   - After uploading or adding an image address from the internet , any user can ask questions related to the intricate details within the image:
   ![Screenshot 2025-01-24 191738](https://github.com/user-attachments/assets/318c7358-aadf-48d8-97c1-a2791759ea82)


## Conversations from all the modalites are stored in same session state

## How to Run

### 1. Clone the repository:
```bash
git clone https://github.com/Raghu6798/Multi_Modal_RAG.git
cd .\Multi_Modal_RAG\
```
### 2. Install all the required dependencies 
```bash
pip install requirements.txt
```
### 3. Run the streamlit web app on localhost:8501
```python
streamlit run app.py
```
