import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cerebras import ChatCerebras
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from uuid import uuid4
import whisper 
import torch 
import tempfile
import faiss
import os 
import torch
from dotenv import load_dotenv
import logging
import asyncio

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

async def async_invoke_chain(chain, input_data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, chain.invoke, input_data)

# Initialize session state for models
if "models" not in st.session_state:
    st.session_state.models = {
        "Gemini": ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.8,
            verbose=True,
            api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
        ),
        "Mistral": ChatMistralAI(
            model_name="open-mistral-nemo", temperature=0.8, verbose=True,api_key=os.getenv("MISTRAL_API_KEY")
        ),
        "Llama": ChatCerebras(
            model="llama-3.3-70b",
            temperature=0.8,
            verbose=True,
            api_key=os.getenv("CEREBRAS_API_KEY"),
        ),
    }

# Initialize embeddings
if "embeddings" not in st.session_state:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": device}, encode_kwargs={"normalize_embeddings": False}
    )

# Recursive text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert explainer of video content. Your goal is to provide comprehensive and insightful answers to user questions based on the provided video transcript. You will combine information from the transcript with your general knowledge to give a well-rounded understanding.

Here's how you should approach each question:

1. **Direct Transcript Answer:** First, directly answer the user's question using relevant excerpts from the provided transcript. Use quotation marks to clearly indicate text taken directly from the transcript.

2. **Detailed Explanation:** Expand on the transcript's information with detailed explanations, context, and background information from your general knowledge. Explain any technical terms or concepts that might be unfamiliar to the user.

3. **Examples and Analogies:** Use examples, analogies, and real-world scenarios to illustrate complex ideas and make them easier to understand.

4. **Code Snippets/URLs (If Applicable):** If the video discusses code or refers to external resources, provide relevant code snippets (formatted for readability) or URLs to further enhance the explanation.

5. **Structure and Clarity:** Present your answers in a clear, structured, and easy-to-read format. Use headings, bullet points, and numbered lists where appropriate.

Context (Video Transcript):
{context}"""),
    ("user", "{question}")
])

st.title("Video QA with LangChain 🦜🔗 & Streamlit") 
st.logo(image=r"C:\Users\Raghu\Downloads\Gen AI Projects\image_logo.jpg",icon_image=r"C:\Users\Raghu\Downloads\Gen AI Projects\image_logo.jpg")

upload_option = st.radio("Select video source:", ["Upload File", "YouTube URL"])
video_path = None
video_url = None

if upload_option == "Upload File":
    uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "webm", "mkv"])
    if uploaded_file:
        # Save the uploaded file to the 'Nothing' folder
        video_path = f"./{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("File uploaded successfully.")
else:
    video_url = st.text_input("Enter YouTube video URL:")


if video_url and st.button("Generate Transcript"):
    with st.spinner("Fetching transcript..."):
        try:
            # Load transcript using YoutubeLoader
            loader = YoutubeLoader.from_youtube_url(
                video_url, add_video_info=False
            )
            transcript = loader.load()

            # Split into documents for chunking
            docs = [Document(page_content=entry.page_content) for entry in transcript]
            chunks = text_splitter.split_documents(docs)

            # **Clear the previous vector store**
            index = faiss.IndexFlatL2(len(st.session_state.embeddings.embed_query("hello world")))
            st.session_state.vector_store = FAISS(
                embedding_function=st.session_state.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

            # Add the new documents to the vector store
            ids = [str(uuid4()) for _ in range(len(chunks))]
            st.session_state.vector_store.add_documents(documents=chunks, ids=ids)
            st.success("You are ready to Ask anything about the Video")
        except Exception as e:
            st.error(f"Error fetching transcript: {e}")

if video_path and st.button("Generate Transcript"):
    with st.spinner("Transcribing video..."):
        try:
            # Transcribe the video using Whisper
            model = whisper.load_model("small")
            result = model.transcribe(video_path)
            transcript = result["text"]

            # Split into documents for chunking
            docs = [Document(page_content=transcript)]
            chunks = text_splitter.split_documents(docs)

            # **Clear the previous vector store**
            index = faiss.IndexFlatL2(len(st.session_state.embeddings.embed_query("hello world")))
            st.session_state.vector_store = FAISS(
                embedding_function=st.session_state.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

            # Add the new documents to the vector store
            ids = [str(uuid4()) for _ in range(len(chunks))]
            st.session_state.vector_store.add_documents(documents=chunks, ids=ids)
            st.success("You are ready to Ask anything about the Video")
        except Exception as e:
            st.error(f"Error transcribing video: {e}")

if video_url:
    st.video(video_url)
# QA Section
if "vector_store" in st.session_state:
    def get_retrieved_context(query):
        video_retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )
        
        retrieved_documents = video_retriever.get_relevant_documents(query)
        return "\n".join(doc.page_content for doc in retrieved_documents)


    user_input = st.chat_input("Ask a question about the video:")
    if user_input:
        context = get_retrieved_context(user_input)
        qa_chain = prompt | st.session_state.models["Llama"] | StrOutputParser()
        response_message = asyncio.run(async_invoke_chain(qa_chain, {"question": user_input, "context": context}))

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response_message})

        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["content"])
else:
    st.error("No transcription available. Please upload or process a video first.")
