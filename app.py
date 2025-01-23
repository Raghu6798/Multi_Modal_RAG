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
