from flask import Flask, request, jsonify
import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

faqs = [
    "How can I reset my online banking password?",
    "How do I check my account balance?",
    "What should I do if my debit card is lost?",
    "How do I activate international transactions on my credit card?",
    "How can I open a new savings account?",
    "What is the minimum balance required?",
    "How do I update my registered mobile number?",
    "How can I apply for a home loan?",
    "What is the process for closing my bank account?",
    "How do I check my loan EMI schedule?",
    "How can I download my account statement?",
    "What is the daily withdrawal limit from an ATM?",
    "How do I enable UPI payments?",
    "Can I increase my credit card limit?",
    "What is the process to block a stolen credit card?",
    "How can I register for mobile banking?",
    "How do I apply for a personal loan?",
    "What is the penalty for not maintaining minimum balance?",
    "How can I dispute a wrong transaction?",
    "What are the bank's working hours?"
]

def setup_faiss_index():
    if not GOOGLE_API_KEY:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

    print("Generating embeddings for FAQs...")
    faq_embeddings = embeddings_model.embed_documents(faqs)  # list of lists

    embeddings_np = np.array(faq_embeddings).astype('float32')
    faiss.normalize_L2(embeddings_np)

    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product index for cosine similarity

    index.add(embeddings_np)

    print(f"Added {len(faqs)} FAQs to FAISS index.")

    return index, embeddings_model

def rag_answer(user_query, index, embeddings_model):
    query_embedding = embeddings_model.embed_query(user_query)
    query_np = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_np)

    D, I = index.search(query_np, k=3)

    retrieved_faqs = [faqs[i] for i in I[0]]

    context = "\n".join([f"- {faq}" for faq in retrieved_faqs])

    prompt = f"""You are a helpful banking assistant. Use the following relevant FAQs from the knowledge base to answer the user's question.

Relevant FAQs:
{context}

User question: {user_query}

Please provide a conversational and helpful answer based on the FAQs above. If the question is not covered by the FAQs, politely mention that and provide general guidance."""

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        max_tokens=1000
    )

    response = llm.invoke(prompt)

    return response.content


# Flask app setup
app = Flask(__name__)

# Initialize index and embedding model once at startup
index, embeddings_model = setup_faiss_index()

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json(force=True)
    question = data.get('question', '').strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    try:
        answer = rag_answer(question, index, embeddings_model)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=5000)
