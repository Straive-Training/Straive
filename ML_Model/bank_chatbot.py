from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ----------------------
# Step 1: Create Banking FAQs
# ----------------------
faqs = [
    "How can I reset my online banking password?",
    "How do I check my account balance?",
    "What should I do if my debit card is lost?",
    "How can I apply for a personal loan?",
    "How do I activate international transactions on my credit card?"
]

answers = {
    faqs[
        0]: "You can reset your password by clicking 'Forgot Password' on the login page and following the verification steps.",
    faqs[1]: "You can check your balance using the mobile app, internet banking, or by visiting an ATM.",
    faqs[2]: "Report the lost debit card immediately through the customer service helpline or the banking app.",
    faqs[3]: "You can apply for a personal loan online through the bank’s portal or by visiting your nearest branch.",
    faqs[4]: "International transactions can be activated via the mobile banking app or by contacting customer care."
}

# ----------------------
# Step 2: Load Embedding Model
# ----------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
# lightweight & good for semantic search

# Encode FAQs into embeddings
faq_embeddings = model.encode(faqs)

# ----------------------
# Step 3: Create FAISS Index
# ----------------------
dimension = faq_embeddings.shape[1]  # embedding size
index = faiss.IndexFlatL2(dimension)  # L2 distance index
index.add(np.array(faq_embeddings))

# ----------------------
# Step 4: User Query
# ----------------------
user_query = "How do I reset my online banking password?"
query_embedding = model.encode([user_query])

# ----------------------
# Step 5: Search Closest FAQ
# ----------------------
k = 1  # retrieve top 1
distances, indices = index.search(np.array(query_embedding), k)

closest_faq = faqs[indices[0][0]]
answer = answers[closest_faq]

# ----------------------
# Step 6: Output
# ----------------------
print("User Query:", user_query)
print("Matched FAQ:", closest_faq)
print("Answer:", answer)




