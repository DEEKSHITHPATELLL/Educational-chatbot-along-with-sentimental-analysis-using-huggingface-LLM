from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS  
import requests
import re
import time
from textblob import TextBlob  
import matplotlib.pyplot as plt
import io
import dotenv
import os
dotenv.load_dotenv()


app = Flask(__name__, template_folder="templates") 
CORS(app)

HF_TOKEN = os.getenv("TOKEN")
API_URL = os.getenv("API_URL")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

EDUCATIONAL_TOPICS = [
    "mathematics", "science", "history", "literature", "technology","English",
    "physics", "chemistry", "biology", "computer science", "engineering",
    "economics", "philosophy", "art", "psychology", "education",
    "artificial intelligence", "machine learning", "deep learning",
    "neural networks", "natural language processing", "computer vision",
    "reinforcement learning", "transformers", "large language models",
    "tensorflow", "pytorch", "scikit-learn", "hugging face", "openai",
    "python", "java", "javascript", "c++", "c#", "ruby", "go", "rust",
    "typescript", "php", "swift", "r programming", "dart",
    "data science", "big data", "pandas", "numpy", "matplotlib",
    "seaborn", "power bi", "tableau", "sql", "data visualization",
    "data engineering","ai",
    "database management system", "sql", "mysql", "postgresql",
    "mongodb", "oracle", "firebase", "cassandra", "redis",
    "cloud computing", "aws", "azure", "google cloud", "docker",
    "kubernetes", "ci/cd", "devops", "serverless computing",
    "cybersecurity", "ethical hacking", "penetration testing",
    "encryption", "cryptography", "blockchain technology",
    "smart contracts", "ethereum", "bitcoin",
    "chatgpt", "langchain", "stable diffusion", "midjourney",
    "openai", "anthropic", "vector databases", "llamaindex",
    "matlab", "wolfram mathematica", "statistical analysis",
    "symbolic computation"
]

sentiment_history = []

def is_educational(prompt):
    """Check if the user's prompt is related to an educational topic."""
    prompt_lower = prompt.lower()
    return any(topic.lower() in prompt_lower for topic in EDUCATIONAL_TOPICS)

def clean_response(user_input, generated_text):
    """
    Cleans the bot response by removing the original question if it appears in the response.
    """
    if not generated_text:
        return "I'm not sure how to respond."
    
    # Remove the user question if it appears at the beginning of the response
    cleaned_text = re.sub(rf'^{re.escape(user_input)}[\s:]*', '', generated_text, flags=re.IGNORECASE).strip()
    
    return cleaned_text if cleaned_text else generated_text  # Return original if cleaning fails

def analyze_sentiment(text):
    """Analyze the sentiment of the given text."""
    blob = TextBlob(text)
    return blob.sentiment.polarity

def generate_tips(prompt):
    """Generate tips based on the user's prompt."""
    tips = {
        "study": "Make sure to take regular breaks and review your notes frequently.",
        "project": "Break down the project into smaller tasks and set achievable deadlines.",
        "exam": "Practice past exam papers and focus on understanding the concepts.",
        # Add more tips as needed
    }
    for keyword, tip in tips.items():
        if keyword in prompt.lower():
            return tip
    return "Stay focused and keep a positive mindset!"

def generate_response(prompt):
    """Generate a response using Hugging Face API with retry logic."""
    if not is_educational(prompt):
        return "Sorry, I can only answer educational questions."

    for attempt in range(3):  # Retry up to 3 times
        try:
            response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})

            if response.status_code == 200:
                response_data = response.json()
                if isinstance(response_data, list) and "generated_text" in response_data[0]:
                    raw_response = response_data[0]["generated_text"]
                    cleaned_response = clean_response(prompt, raw_response)
                    
                    sentiment = analyze_sentiment(cleaned_response)
                    sentiment_text = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
                    tips = generate_tips(prompt)
                    
                    # Store sentiment history
                    sentiment_history.append(sentiment)
                    
                    return f"{cleaned_response}\n\nSentiment: {sentiment_text}\nTip: {tips}"
                else:
                    return "Error: Unexpected response format from Hugging Face API."

            elif response.status_code == 503:
                print(f"Attempt {attempt + 1}: API is busy. Retrying in 5 seconds...")
                time.sleep(5)  # Wait 5 seconds before retrying

            else:
                return f"Error: Unable to fetch response from Hugging Face API. Status Code: {response.status_code}"

        except requests.exceptions.RequestException as e:
            print("Error:", e)
            return "Sorry, an error occurred while fetching the response."

    return "Error: Hugging Face API is still unavailable after multiple attempts."

@app.route("/")
def home():
    """Serve the frontend page."""
    return render_template("index.html") 

@app.route("/chat", methods=["POST"])
def chat():
    """Chatbot API endpoint."""
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "Please enter a valid question."})

    response = generate_response(user_input)
    return jsonify({"response": response})

@app.route("/sentiment-graph")
def sentiment_graph():
    """Generate and return sentiment data as JSON."""
    labels = list(range(1, len(sentiment_history) + 1))
    sentiments = sentiment_history
    return jsonify({"labels": labels, "sentiments": sentiments})

if __name__ == "__main__":
    app.run(debug=True)