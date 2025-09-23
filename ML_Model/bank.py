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

    # Convert embeddings to numpy float32 array
    embeddings_np = np.array(faq_embeddings).astype('float32')

    # Build FAISS index (using Inner Product for cosine similarity; normalize vectors before)
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings_np)

    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product (cosine similarity if vectors normalized)

    # Add embeddings to index
    index.add(embeddings_np)

    print(f"Added {len(faqs)} FAQs to FAISS index.")

    return index, embeddings_model

def rag_answer(user_query, index, embeddings_model):
    # Embed query and normalize for cosine similarity
    query_embedding = embeddings_model.embed_query(user_query)
    query_np = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_np)

    # Search top 3 FAQs
    D, I = index.search(query_np, k=3)  # I contains indexes of nearest neighbors

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

def main():
    try:
        index, embeddings_model = setup_faiss_index()

        print("Initializing Gemini LLM...")

        user_query = "I forgot my online banking password. What should I do?"

        print(f"\nUser Query: {user_query}")
        print("Generating answer...")

        answer = rag_answer(user_query, index, embeddings_model)

        print(f"\nAI Answer: {answer}")

        print("\n" + "=" * 50)
        print("Interactive FAQ Bot (type 'quit' to exit)")
        print("=" * 50)

        while True:
            user_input = input("\nYour question: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Thank you for using the banking FAQ bot!")
                break

            if not user_input:
                continue

            try:
                answer = rag_answer(user_input, index, embeddings_model)
                print(f"\nAnswer: {answer}")
            except Exception as e:
                print(f"Error generating answer: {e}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Set GOOGLE_API_KEY environment variable")
        print("2. Install required packages: pip install faiss-cpu langchain-google-genai")

if __name__ == "__main__":
    main()
