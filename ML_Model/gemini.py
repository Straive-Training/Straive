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
    embeddings = np.array(embeddings).astype('float32')  # Convert to numpy array

    # Check embedding dimension by looking at the length of the first embedding
    embedding_dimension = embeddings.shape[1]  # Ensure you are extracting the correct dimension

    # Initialize FAISS index with the embedding dimension
    index = faiss.IndexFlatL2(embedding_dimension)

    # Add embeddings to FAISS index
    index.add(embeddings)

    # Now use LangChain's FAISS class to store the index and text chunks
    vector_store = FAISS.from_texts(chunks, embedding)

    # Get user question
    user_question = st.text_input("Type your question here")

    # Perform similarity search
    if user_question:
        # Convert user question to embedding
        user_embedding = embedding.embed_documents([user_question])

        # Perform the search
        D, I = index.search(np.array(user_embedding).astype('float32'), k=5)  # Top 1 result

        # Get the most relevant chunk based on the search
        most_relevant_chunk = chunks[I[0][0]]

        # Show the most relevant chunk to the user
        st.write("Most relevant document chunk:")
        st.write(most_relevant_chunk)

        # Optional: Use Hugging Face's question-answering model to get an answer from the chunk
        qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

        # Get the answer from the most relevant chunk
        answer = qa_pipeline(question=user_question, context=most_relevant_chunk)

        # Show the answer from the QA model
        st.write(f"Answer (from QA model): {answer['answer']}")

        # Step 2: Now we pass both the question and the retrieved chunk to an LLM for further processing
        # Example LLM: GPT-3 using OpenAI API or any other LLM (HuggingFace has some generative models too)
        # Here, we use Hugging Face's GPT-2 as an example of a generative LLM.
        # You can replace this with another more powerful model if needed.

        # Initialize a Hugging Face pipeline for text generation
        generator = pipeline("text-generation", model="distilgpt2")  # You can choose any other model like GPT-3, GPT-4, etc.

        # Combine the question with the relevant chunk and provide it to the LLM
        user_question = "generate 5 mcq's"
        context_for_llm = f"Context: {most_relevant_chunk}\n\nQuestion: {user_question}"
        max_input_length = 512  # Adjust this based on the model's max token limit
        context_for_llm = context_for_llm[:max_input_length]

        # Generate a response using the LLM
        llm_response = generator(context_for_llm, max_length=150, num_return_sequences=1)

        # Show the LLM-generated response
        st.write("LLM Response:")
        st.write(llm_response[0]['generated_text'])
