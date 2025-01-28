import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cerebras import ChatCerebras
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader
from streamlit_pdf_viewer import pdf_viewer
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
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

if 'pdf_ref' not in st.session_state:
    st.session_state.pdf_ref = None

# Async function to invoke chain
async def async_invoke_chain(chain, input_data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, chain.invoke, input_data)

# Initialize session state for messages and models
if "messages" not in st.session_state:
    st.session_state["messages"] = []


llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b ",
    temperature=0.8,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)



if "models" not in st.session_state:
    st.session_state["models"] = {
        "Gemini": ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
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

if "embeddings" not in st.session_state:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    st.session_state["embeddings"] = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

# Streamlit UI Header
st.header("🗂️ ✨ Document Question Answering")
st.write("""Upload a document and query its content. Supported formats include:
- PDF Files (.pdf)
- Word Documents (.docx)
""")

# File uploader for document
uploaded_doc = st.file_uploader("Upload your document (.pdf, .docx):", type=["pdf", "docx"])

# Process uploaded PDF document
if uploaded_doc and uploaded_doc.name.endswith(".pdf"):
    # Store the uploaded PDF file in session state for preview in sidebar
    st.session_state.pdf_ref = uploaded_doc

    # Display PDF preview in the sidebar
    with st.sidebar:
        binary_data = st.session_state.pdf_ref.getvalue()
        pdf_viewer(input=binary_data, width=700)

    # Process the PDF file
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
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=1200, chunk_overlap=200)
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
        doc_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        def get_retrieved_context(query):
            retrieved_documents = doc_retriever.get_relevant_documents(query)
            return "\n".join(doc.page_content for doc in retrieved_documents)

        # User input for querying the document
        user_input = st.chat_input("Ask your queries about the document/documents:")

        # Define prompt template
        prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        You are an expert document analyst with the ability to process large volumes of text efficiently. 
        Your task is to extract key insights and answer questions based on the content of the provided document : {context}
        When asked a question, you should provide a direct, detailed, and concise response, only using the information available from the document. 
        If the answer cannot be found directly, you should clarify this and provide relevant context or related information if applicable.
        Focus on uncovering critical information, whether it's specific facts, summaries, or hidden insights within the document.
    """),
    ("human", "{question}")
])


        # Handle user input and display responses
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            qa_chain = prompt_template | st.session_state["models"]["Deepseek-R1-distill-llama-70b":] | StrOutputParser()
            context = get_retrieved_context(user_input)
            response_message = asyncio.run(async_invoke_chain(qa_chain, {"question": user_input, "context": context}))
            st.session_state["messages"].append({"role": "assistant", "content": response_message})

            for message in st.session_state["messages"]:
                st.chat_message(message["role"]).markdown(message["content"])
