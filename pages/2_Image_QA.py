import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_cerebras import ChatCerebras
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from uuid import uuid4
import faiss
import os
from dotenv import load_dotenv
import logging
import httpx
import base64
import asyncio

# Initialize environment variables and logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Async function to invoke chain
async def async_invoke_chain(chain, input_data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, chain.invoke, input_data)

# Initialize session state for messages and models
if "messages" not in st.session_state:
    st.session_state.messages = []

if "models" not in st.session_state:
    st.session_state.models = {
        "Gemini": ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.8,
            verbose=True,
            api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY")
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

# Initialize embeddings model
if "embeddings" not in st.session_state:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

st.header("📸📈📊 ֎ Image Content Analysis and Question Answering")

# Brief overview for image content analysis
description = """
Upload an image, and the AI will analyze its content and answer your questions. 
It can interpret various types of images including:
- General imagery (objects, people, scenes)
- Diagrams, graphs, and data visualizations
- Scientific and medical images
- Text-based images (documents, screenshots)
"""

# Display the brief description
st.write(description)

# File upload and URL input
st.header("Upload Image for Question Answering")
uploaded_file = st.file_uploader("Upload an image (.jpeg, .jpg, .png, etc.):", type=["jpeg", "jpg", "png"])

st.header("Or Enter the Image URL :")
image_url = st.text_input("Enter the image URL")

image_data = None

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    image_data = base64.b64encode(uploaded_file.read()).decode("utf-8")
elif image_url:
    try:
        with httpx.Client() as client:
            response = client.get(image_url)
            response.raise_for_status()
            st.image(response.content, caption="Image from URL", use_column_width=True)
            image_data = base64.b64encode(response.content).decode("utf-8")
    except Exception as e:
        st.error(f"Error fetching image from URL: {e}")

if image_data:
    message = HumanMessage(content=[{
            "type": "text", "text": "Describe what is in the image in detail."
        }, {
            "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        }])

    # Generate response from the model
    response = asyncio.run(async_invoke_chain(st.session_state.models["Gemini"], [message]))
    knowledge = [Document(page_content=response.content)]

    # Split text into chunks for indexing
    text_splitter = RecursiveCharacterTextSplitter(separators="\n\n", chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(knowledge)

    # Create FAISS IndexHNSWFlat for indexing image embeddings
    index = faiss.IndexFlatL2(len(st.session_state.embeddings.embed_query("hello world")))

    # Create FAISS vector store for document retrieval
    vector_store = FAISS(
        embedding_function=st.session_state.embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Generate unique IDs and add documents to the store
    ids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=ids)

    # Update the mapping between FAISS index and document IDs
    for idx, doc_id in enumerate(ids):
        vector_store.index_to_docstore_id[idx] = doc_id

    # Create image retriever with the FAISS index
    image_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    def get_retrieved_context(query):
        retrieved_documents = image_retriever.get_relevant_documents(query)
        return "\n".join(doc.page_content for doc in retrieved_documents)

    # User query for image QA
    user_input = st.chat_input("Ask a question about the image:")

    prompt = ChatPromptTemplate.from_messages([(
            "system", "You are an expert in analyzing images. Use the context: {context} to answer the query."
        ), ("human", "{question}")])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        qa_chain = prompt | st.session_state.models["Mistral"] | StrOutputParser()
        context = get_retrieved_context(user_input)
        response_message = asyncio.run(async_invoke_chain(qa_chain, {"question": user_input, "context": context}))
        st.session_state.messages.append({"role": "assistant", "content": response_message})
        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["content"])
