# import streamlit as st
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain_community.docstore import InMemoryDocstore
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_cerebras import ChatCerebras
# from langchain_mistralai import ChatMistralAI
# from langchain_core.messages import HumanMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import StrOutputParser
# from uuid import uuid4
# import numpy as np
# import faiss
# import whisper
# import torch
# import os
# from dotenv import load_dotenv
# import logging
# import base64
# import asyncio
# from concurrent.futures import ThreadPoolExecutor

# # Load environment variables
# load_dotenv()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# st.set_page_config(
#     page_title = "Multi-Modal RAG",
#     page_icon = ":red RED",
#     layout="wide"
# )

# if "models" not in st.session_state:
#     st.session_state.models = {
#         "Gemini": ChatGoogleGenerativeAI(
#             model="gemini-2.0-flash-exp",
#             temperature=0.8,
#             verbose=True,
#             api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY")
#         ),
#         "Mistral": ChatMistralAI(
#             model_name="open-mistral-nemo",
#             temperature=0.8,
#             verbose=True
#         ),
#         "Llama": ChatCerebras(
#             model="llama-3.3-70b",
#             temperature=0.8,
#             verbose=True,
#             api_key=os.getenv("CEREBRAS_API_KEY")
#         )
#     }

# if "embeddings" not in st.session_state:
#     model_name = "sentence-transformers/all-mpnet-base-v2"
#     model_kwargs = {'device': 'cpu'}
#     encode_kwargs = {'normalize_embeddings': False}
#     st.session_state.embeddings = HuggingFaceEmbeddings(
#         model_name=model_name,
#         model_kwargs=model_kwargs,
#         encode_kwargs=encode_kwargs
#     )

# # Initialize Streamlit app
# st.title("Multi-Modal Retrieval-Augmented Generation (RAG)")
# st.sidebar.title("Options")
# st.sidebar.write("""
# 1. Video Question Answering RAG (.mp4, .mp3, etc.)
# 2. Image Question Answering RAG (.jpeg, .jpg, .png, etc.)
# 3. Document Question Answering RAG (.pdf, .docx, etc.)
# """)

# # Initialize session state for conversation history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Function to process LLM calls asynchronously
# async def async_invoke_chain(chain, input_data):
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(None, chain.invoke, input_data)

# async def process_video(video_path):
#     st.info("Transcribing video...")
#     model = whisper.load_model("small")
#     model = model.to(device="cuda" if torch.cuda.is_available() else "cpu")
#     loop = asyncio.get_event_loop()
#     result = await loop.run_in_executor(None, model.transcribe, video_path)
#     return result["text"]   

# # Handle video processing
# # Correct the variable name mismatch and properly handle video retrieval.
# if modal_choice == "Video":
#     st.header("Step 1: Upload or Provide Video")
#     uploaded_file = st.file_uploader("Upload a video file (.mp4, .webm, etc.):", type=["mp4", "webm", "mkv"])

#     if st.button("Process Video"):
#         if uploaded_file:
#             save_path = "./uploads"
#             os.makedirs(save_path, exist_ok=True)
#             video_path = os.path.join(save_path, uploaded_file.name)
#             with open(video_path, "wb") as f:
#                 f.write(uploaded_file.read())
#             transcription = asyncio.run(process_video(video_path))  # Fetch transcription text
#             st.session_state.transcription = transcription
#             st.success("Transcription complete!")
#         else:
#             st.error("Please upload a video file.")
        
#         # Handle LLM response after transcription
#         if "transcription" in st.session_state:
#             response = asyncio.run(async_invoke_chain(st.session_state.models["Gemini"], transcription))  # Correcting the input here
#             knowledge = [Document(page_content=response.content)]
#             text_splitter = RecursiveCharacterTextSplitter(separators="\n\n", chunk_size=1500, chunk_overlap=200)
#             chunks = text_splitter.split_documents(knowledge)

#             # Creating the FAISS index
#             index = faiss.IndexFlatL2(len(st.session_state.embeddings.embed_query("hello world")))

#             # Create FAISS vector store for document retrieval
#             vector_store = FAISS(
#                 embedding_function=st.session_state.embeddings,
#                 index=index,
#                 docstore=InMemoryDocstore(),
#                 index_to_docstore_id={},
#             )

#             # Generate unique IDs and add documents to the store
#             ids = [str(uuid4()) for _ in range(len(chunks))]
#             vector_store.add_documents(documents=chunks, ids=ids)

#             # Update the mapping between FAISS index and document IDs
#             for idx, doc_id in enumerate(ids):
#                 vector_store.index_to_docstore_id[idx] = doc_id

#             # Create video retriever with the FAISS index
#             video_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

#             def get_retrieved_context(query):
#                 retrieved_documents = video_retriever.get_relevant_documents(query)
#                 return "\n".join(doc.page_content for doc in retrieved_documents)

#             # User query for video QA
#             user_input = st.chat_input("Ask a question about the video:")
#             prompt = ChatPromptTemplate.from_messages([(
#                 "system", "You are an expert in analyzing videos. Use the context: {context} to answer the query."
#             ), ("human", "{question}")])

#             if user_input:
#                 st.session_state.messages.append({"role": "user", "content": user_input})
#                 qa_chain = prompt | st.session_state.models["Mistral"] | StrOutputParser()
#                 context = get_retrieved_context(user_input)
#                 response_message = asyncio.run(async_invoke_chain(qa_chain, {"question": user_input, "context": context}))
#                 st.session_state.messages.append({"role": "assistant", "content": response_message})
#                 for message in st.session_state.messages:
#                     st.chat_message(message["role"]).markdown(message["content"])


# # Handle image processing
# elif modal_choice == "Image":
#     st.header("Step 2: Upload Image for Question Answering")
#     uploaded_file = st.file_uploader("Upload an image (.jpeg, .jpg, .png, etc.):", type=["jpeg", "jpg", "png"])

#     if uploaded_file:
#         st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
#         image_data = base64.b64encode(uploaded_file.read()).decode("utf-8")

#         message = HumanMessage(content=[{
#             "type": "text", "text": "Describe what is in the image in detail."
#         }, {
#             "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
#         }])
#         response = asyncio.run(async_invoke_chain(st.session_state.models["Gemini"], [message]))
#         knowledge = [Document(page_content=response.content)]

#         # Split text into chunks for indexing
#         text_splitter = RecursiveCharacterTextSplitter(separators="\n\n", chunk_size=1500, chunk_overlap=200)
#         chunks = text_splitter.split_documents(knowledge)

#         # Create FAISS IndexHNSWFlat for indexing image embeddings
#         index = faiss.IndexFlatL2(len(st.session_state.embeddings.embed_query("hello world")))

#         # Create FAISS vector store for document retrieval
#         vector_store = FAISS(
#             embedding_function=st.session_state.embeddings,
#             index=index,
#             docstore=InMemoryDocstore(),
#             index_to_docstore_id={},
#         )

#         # Generate unique IDs and add documents to the store
#         ids = [str(uuid4()) for _ in range(len(chunks))]
#         vector_store.add_documents(documents=chunks, ids=ids)

#         # Update the mapping between FAISS index and document IDs
#         for idx, doc_id in enumerate(ids):
#             vector_store.index_to_docstore_id[idx] = doc_id

#         # Create image retriever with the FAISS index
#         image_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

#         def get_retrieved_context(query):
#             retrieved_documents = image_retriever.get_relevant_documents(query)
#             return "\n".join(doc.page_content for doc in retrieved_documents)

#         # User query for image QA
#         user_input = st.chat_input("Ask a question about the image:")
#         prompt = ChatPromptTemplate.from_messages([(
#             "system", "You are an expert in analyzing images. Use the context: {context} to answer the query."
#         ), ("human", "{question}")])

#         if user_input:
#             st.session_state.messages.append({"role": "user", "content": user_input})
#             qa_chain = prompt | st.session_state.models["Mistral"] | StrOutputParser()
#             context = get_retrieved_context(user_input)
#             response_message = asyncio.run(async_invoke_chain(qa_chain, {"question": user_input, "context": context}))
#             st.session_state.messages.append({"role": "assistant", "content": response_message})
#             for message in st.session_state.messages:
#                 st.chat_message(message["role"]).markdown(message["content"])

# elif modal_choice == "Document":
#     st.header("Step 3: Upload Document for Question Answering")
#     uploaded_doc = st.file_uploader("Upload a document (.pdf, .docx, etc.):", type=["pdf", "docx"])
#     if uploaded_doc:
#         pass
import streamlit as st

# Set up Streamlit page
st.set_page_config(page_title="Multi-Modal RAG", page_icon=":red_circle:",initial_sidebar_state="expanded", layout="wide",menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    })

# Title of the web app
st.title("Multi-Modal Retrieval-Augmented Generation (RAG)")

# Explanation for Non-Technical Users
st.write("""
    **Multi-Modal RAG** stands for **Multi-Modal Retrieval-Augmented Generation**. It's a process that allows you to ask questions about different types of media, such as videos, images, or documents, and get answers powered by artificial intelligence.
    
    In this app, you can interact with the following features:
    
    1. **Video Question Answering**: Upload a video, and the app will transcribe it. You can then ask questions about the video's content.
    2. **Image Question Answering**: Upload an image, and the app will describe it. You can ask questions about the contents of the image.
    3. **Document Question Answering**: Upload a document (PDF, Word, etc.), and the app will extract the relevant information to answer your questions.
    
    Here's how it works:
    
    - **Video QA**: The app first transcribes the video to text. Then, it allows you to ask any questions about the video. Based on the transcription, it retrieves relevant information to help answer your query.
    
    - **Image QA**: Upload an image, and the app will analyze the image, describing its contents. You can then ask questions about what’s in the image.
    
    - **Document QA**: Upload a document (like a PDF or Word file). The app extracts key information from the document to help answer your questions.
    
    Each feature uses a combination of AI models and sophisticated algorithms to give you the best possible answers.
""")
