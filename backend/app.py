import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch # Or tensorflow if using TF weights

# --- Configuration ---
TRANSCRIPT_FILE = "transcript.json"
MODEL_NAME = "google/flan-t5-small" # Small, good for Q&A
MAX_CONTEXT_SNIPPETS = 3 # How many relevant snippets to feed the LLM
MAX_LLM_ANSWER_TOKENS = 150 # Max length of the generated answer

# --- Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for local development

print("Loading transcript...")
try:
    with open(TRANSCRIPT_FILE, 'r') as f:
        transcript_data = json.load(f)
    # Prepare data for retrieval
    corpus = [item['text'] for item in transcript_data]
    timestamps = [{'start': item['start_time'], 'end': item['end_time']} for item in transcript_data]
except FileNotFoundError:
    print(f"ERROR: Transcript file '{TRANSCRIPT_FILE}' not found.")
    exit()
except json.JSONDecodeError:
    print(f"ERROR: Could not decode JSON from '{TRANSCRIPT_FILE}'.")
    exit()
except KeyError as e:
    print(f"ERROR: Missing key {e} in transcript data.")
    exit()


print("Initializing TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

print(f"Loading LLM ({MODEL_NAME})... (This may take a moment)")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
print("LLM Loaded.")

# --- Helper Functions ---
def find_relevant_snippets(query, top_n=MAX_CONTEXT_SNIPPETS):
    """Finds the most relevant transcript snippets based on TF-IDF cosine similarity."""
    if not query or not corpus:
        return [], []

    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get indices of top N snippets, sorted by similarity
    sorted_indices = similarities.argsort()[::-1]
    relevant_indices = sorted_indices[:top_n]

    # Filter out low-similarity results (optional threshold)
    # relevant_indices = [idx for idx in relevant_indices if similarities[idx] > 0.1] # Example threshold

    if not relevant_indices.size:
         return [], []

    relevant_snippets_text = [corpus[i] for i in relevant_indices]
    relevant_timestamps = [timestamps[i] for i in relevant_indices]

    # Return snippets sorted chronologically for better context flow? Maybe not necessary.
    # Keep them sorted by relevance for now.
    print(f"Found {len(relevant_snippets_text)} relevant snippets for query: '{query}'")
    return relevant_snippets_text, relevant_timestamps

def generate_answer(query, context_snippets):
    """Generates an answer using the LLM based on the query and context."""
    if not context_snippets:
        return "I couldn't find relevant information in the transcript to answer that.", None

    # Simple Prompt Engineering for Flan-T5
    context_str = "\n".join(context_snippets)
    prompt = f"Based on the following video transcript context:\n---\n{context_str}\n---\nAnswer the question: {query}\nAnswer:"

    print(f"\n--- LLM Prompt ---\n{prompt}\n-------------------\n")

    try:
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids # Ensure prompt fits model context
        # Use torch.no_grad() for inference if using PyTorch
        with torch.no_grad():
             outputs = model.generate(
                 input_ids,
                 max_new_tokens=MAX_LLM_ANSWER_TOKENS,
                 # num_beams=4, # Optional: Beam search for potentially better quality
                 # early_stopping=True # Optional
                 )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"LLM Generated Answer: {answer}")
        return answer

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "Sorry, I encountered an error while generating the answer.", None


# --- API Endpoint ---
@app.route('/query', methods=['POST'])
def handle_query():
    """Handles user queries, finds context, generates answer, and returns results."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    print(f"\nReceived query: {query}")

    # 1. Retrieve Relevant Snippets & Timestamps
    relevant_texts, relevant_times = find_relevant_snippets(query)

    # 2. Generate Answer using LLM
    llm_answer = generate_answer(query, relevant_texts)

    # 3. Determine suggested timestamp (e.g., start time of the most relevant snippet)
    suggested_time = None
    if relevant_times:
        # Use the start time of the *most* relevant snippet (index 0 after sorting by similarity)
        suggested_time = relevant_times[0]['start']
        print(f"Suggesting timestamp: {suggested_time}s")

    response = {
        "answer": llm_answer,
        "suggested_timestamp": suggested_time # Can be null if no relevant context found
    }

    return jsonify(response)

# --- Main Execution ---
if __name__ == '__main__':
    # Consider host='0.0.0.0' if running in Docker or need external access
    app.run(debug=True, port=5000) # debug=True is helpful for development