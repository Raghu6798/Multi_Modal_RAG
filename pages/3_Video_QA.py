import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cerebras import ChatCerebras
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from uuid import uuid4
import whisper
import torch
import tempfile
import faiss
from dotenv import load_dotenv
import logging
import asyncio
import os

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

if "models" not in st.session_state:
    st.session_state["models"] = {
        "Gemini": ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.8,
            verbose=True,
            api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        ),
        "Deepseek-R1-distill-llama-70b": ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            temperature=0.8,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("GROQ_API_KEY"),
        ),
        "Mistral": ChatMistralAI(
            model_name="open-mistral-nemo",
            temperature=0.8,
            verbose=True
        ),
        "Llama": ChatCerebras(
            model="llama-3.3-70b",
            temperature=0.8,
            verbose=True,
            api_key=os.getenv("CEREBRAS_API_KEY")
        )
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

1. **Video Introductory Overview:** First, directly answer the user's question using relevant excerpts from the provided transcript. Use quotation marks to clearly indicate text taken directly from the transcript.

2. **Detailed Explanation:** Expand on the transcript's information with detailed explanations, context, and background information from your general knowledge. Explain any technical terms or concepts that might be unfamiliar to the user.

3. **Examples and Analogies:** Use examples, analogies, and real-world scenarios to illustrate complex ideas and make them easier to understand.

4. **Code Snippets/URLs (If Applicable):** If the video discusses code or refers to external resources, provide relevant code snippets (formatted for readability) or URLs to further enhance the explanation.

5. **Structure and Clarity:** Present your answers in a clear, structured, and easy-to-read format. Use headings, bullet points, and numbered lists where appropriate.

Context (Video Transcript):
{context}"""),
    ("user", "{question}")
])

st.title("Video QA with LangChain 🦜🔗 & Streamlit")

# Upload video file
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

video_url = None

if uploaded_video:
    st.video(uploaded_video)
    if st.button("Generate Transcript from Video"):
        with st.spinner("Transcribing video..."):
            try:
                # Save the uploaded video file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(uploaded_video.getvalue())
                    temp_file_path = temp_file.name

                # Load Whisper model and transcribe the video
                model = whisper.load_model("small")
                model = model.to("cpu")
                result = model.transcribe(temp_file_path)

                # Get the transcript text
                transcript = result["text"]
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
                st.error(f"Error fetching transcript: {e}")

else:
    # YouTube video input
    video_url = st.text_input("Enter YouTube video URL:")

    if video_url and st.button("Generate Transcript from YouTube"):
        st.video(video_url)
        
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
        qa_chain = prompt | st.session_state.models[ "Deepseek-R1-distill-llama-70b"] | StrOutputParser()
        response_message = asyncio.run(async_invoke_chain(qa_chain, {"question": user_input, "context": context}))

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response_message})

        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["content"])
else:
    st.error("No transcription available. Please upload or process a video first.")
