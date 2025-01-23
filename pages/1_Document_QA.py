import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
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
from dotenv import load_dotenv
import logging
import asyncio

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Async function to invoke chain
async def async_invoke_chain(chain, input_data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, chain.invoke, input_data)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "models" not in st.session_state:
    st.session_state["models"] = {
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

if "embeddings" not in st.session_state:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    st.session_state["embeddings"] = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

# Streamlit UI Header
st.header("🗂️ ✨ Document Question Answering")
st.write("""
Upload a document and query its content. Supported formats include:
- PDF Files (.pdf)
- Word Documents (.docx)
""")

# File uploader for document
uploaded_doc = st.file_uploader("Upload your document (.pdf, .docx):", type=["pdf", "docx"])

# Process uploaded PDF document
if uploaded_doc and uploaded_doc.name.endswith(".pdf"):
    with st.spinner("Processing the uploaded PDF document..."):
        # Save the uploaded file temporarily
        temp_path = f"temp_{uuid4().hex}.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_doc.read())

        # Load document using PyMuPDFLoader
        loader = PyMuPDFLoader(temp_path)
        documents = loader.load()

        # Remove the temporary file
        os.remove(temp_path)

        st.success(f"Successfully loaded {len(documents)} pages from the uploaded PDF.")

        # Embed the documents into FAISS index
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        embedding_dim = len(st.session_state["embeddings"].embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)
        vector_store = FAISS(
            embedding_function=st.session_state["embeddings"],
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        ids = [str(uuid4()) for _ in range(len(chunks))]
        vector_store.add_documents(chunks, ids=ids)

        for idx, doc_id in enumerate(ids):
            vector_store.index_to_docstore_id[idx] = doc_id

        # Create retriever with the FAISS index
        doc_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        def get_retrieved_context(query):
            retrieved_documents = doc_retriever.get_relevant_documents(query)
            return "\n".join(doc.page_content for doc in retrieved_documents)

        user_input = st.chat_input("Ask your queries about the document/documents:")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in analyzing a large number of documents. Use the context: {context} to answer the query related to the document in a concise but detailed manner."),
            ("human", "{question}")
        ])

        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            qa_chain = prompt_template | st.session_state["models"]["Mistral"] | StrOutputParser()
            context = get_retrieved_context(user_input)
            response_message = asyncio.run(async_invoke_chain(qa_chain, {"question": user_input, "context": context}))
            st.session_state["messages"].append({"role": "assistant", "content": response_message})

            for message in st.session_state["messages"]:
                st.chat_message(message["role"]).markdown(message["content"])
