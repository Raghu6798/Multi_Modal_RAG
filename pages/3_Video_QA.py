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
            api_key=st.secrets["GOOGLE_AI_STUDIO_API_KEY"],
        ),
        "Mistral": ChatMistralAI(
            model_name="open-mistral-nemo", temperature=0.8, verbose=True,api_key=st.secrets["MISTRAL_API_KEY"]
        ),
        "Llama": ChatCerebras(
            model="llama-3.3-70b",
            temperature=0.8,
            verbose=True,
            api_key=st.secrets["CEREBRAS_API_KEY"],
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
    ("system", "Answer the following question on the given context: {context} as well as from your base cut-off knowledge. Provide clear and structured answers, including code snippets or URLs where applicable."),
    ("user", "{question}")
])

# Streamlit UI
st.title("Video QA with LangChain & Streamlit")

upload_option = st.radio("Select video source:", ["Upload File", "YouTube URL"])
video_path = None
video_url = None

if upload_option == "Upload File":
    uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "webm", "mkv"])
    if uploaded_file:
        # Save the uploaded file to the 'Nothing' folder
        video_path = f"./Nothing/{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("File uploaded successfully.")
else:
    video_url = st.text_input("Enter YouTube video URL:")

# Transcription
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

            # Initialize FAISS vector store
            if "vector_store" not in st.session_state:
                index = faiss.IndexFlatL2(len(st.session_state.embeddings.embed_query("hello world")))
                st.session_state.vector_store = FAISS(
                    embedding_function=st.session_state.embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )

            ids = [str(uuid4()) for _ in range(len(chunks))]
            st.session_state.vector_store.add_documents(documents=chunks, ids=ids)
            st.success("Transcript fetched and indexed.")
        except Exception as e:
            st.error(f"Error fetching transcript: {e}")

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
