import string
from collections import Counter
import matplotlib.pyplot as plt

# Step 1: Load and clean the review text
with open('amazon_reviews.txt', encoding='utf-8') as file:
    text = file.read()

lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
tokenized_words = cleaned_text.split()

# Step 2: Define stop words
stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

# Step 3: Remove stop words
final_words = [word for word in tokenized_words if word not in stop_words]

# Step 4: Load emotion words from amazon_emotion.txt
emotion_dict = {}
with open('amazon_emotion.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if ':' in line:
            word, emotion = line.split(':')
            emotion_dict[word.strip()] = emotion.strip()

# Step 5: Match words to emotions
emotion_list = []
for word in final_words:
    if word in emotion_dict:
        emotion_list.append(emotion_dict[word])

# Step 6: Count and display emotions
emotion_counts = Counter(emotion_list)

print("Detected emotions:", emotion_counts)

# Step 7: Plot the result
colors = {'positive': 'green', 'negative': 'red'}
bar_colors = [colors.get(emotion, 'blue') for emotion in emotion_counts.keys()]

plt.figure(figsize=(6, 4))
plt.bar(emotion_counts.keys(), emotion_counts.values(), color=bar_colors)
plt.title("Amazon Review Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("amazon_graph.png")
plt.show()
