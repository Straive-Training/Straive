import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import faiss
import numpy as np

# Define Hugging Face model name for embeddings
model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Or use another model like 'distilbert-base-uncased'

# Initialize LangChain's HuggingFace Embeddings class
embedding = HuggingFaceEmbeddings(model_name=model_name)

# Upload PDF files
st.header("My first Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract the text from the uploaded PDF
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Break it into chunks for easier processing
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Use LangChain's Hugging Face embedding to generate embeddings
    embeddings = embedding.embed_documents(chunks)

    # Check embedding dimension by looking at the length of the first embedding
    embedding_dimension = len(embeddings[0])  # Get the length of the first embedding
    index = faiss.IndexFlatL2(embedding_dimension)  # L2 distance index

    # Add embeddings to FAISS index
    embeddings = [embedding for embedding in embeddings]  # Ensure embeddings are in the correct format
    index.add(np.array(embeddings).astype('float32'))  # Convert to numpy array and add to FAISS

    # Now use LangChain's FAISS class to store the index and text chunks
    vector_store = FAISS.from_texts(chunks, embedding)

    # Get user question
    user_question = st.text_input("Type your question here")

    # Perform similarity search
    if user_question:
        # Convert user question to embedding
        user_embedding = embedding.embed_documents([user_question])

        # Perform the search (get top 5 most relevant chunks)
        D, I = index.search(np.array(user_embedding).astype('float32'), k=5)  # Top 5 results

        # Get the most relevant chunks based on the search
        relevant_chunks = [chunks[i] for i in I[0]]

        # Show the relevant chunks to the user
        # st.write("Most relevant document chunks:")
        # for i, chunk in enumerate(relevant_chunks):
        #     st.write(f"Chunk {i + 1}:")
        #     st.write(chunk)

        # Optional: Use Hugging Face's question-answering model to get an answer from the chunks
        # qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

        # Get the answer from the concatenated chunks
        # context_for_qa = " ".join(relevant_chunks)  # Combine the relevant chunks into one context
        # answer = qa_pipeline(question=user_question, context=context_for_qa)

        # Show the answer from the QA model
        # st.write(f"Answer (from QA model): {answer['answer']}")

        # Now we pass the concatenated chunks to an LLM for MCQ generation
        # Initialize a Hugging Face pipeline for text generation
        generator = pipeline("text-generation", model="facebook/bart-large")  # You can choose any other model like GPT-3, GPT-4, etc.
        context_for_llm = f"Context: {text}\n\nQuestion: {user_question}"

        # Generate a response using the LLM
        llm_response = generator(context_for_llm, max_length=250, num_return_sequences=1)

        # Show the LLM-generated response
        st.write("LLM Response (Generated MCQs):")
        st.write(llm_response[0]['generated_text'])
