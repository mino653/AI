from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import re

app = Flask(__name__)
# Enable CORS for the front-end (index.html)
CORS(app) 

# --- API KEY & AI Initialization ---
# RENDER/GITHUB DEPLOYMENT REQUIRES READING THE KEY FROM ENVIRONMENT VARIABLES
# The GOOGLE_API_KEY must be set in the Render Environment settings.
api_key = os.environ.get("GOOGLE_API_KEY") 

llm = None
try:
    if not api_key:
        # If the key is not set, we cannot initialize the model
        print("FATAL ERROR: GOOGLE_API_KEY environment variable not set. Model not initialized.")
        llm = None
    else:
        # Define the system persona/prompt
        system_prompt = """
        Your name is Test. You are an advanced, general-purpose AI assistant. 
        You were created and developed by Proll. Your primary purpose is to serve 
        the users of this custom application with highly knowledgeable, friendly, 
        and enthusiastic expert guidance.

        Your core intelligence is powered by Google's cutting-edge Gemini 2.5 Flash model.
        
        Use Markdown extensively (headings, bolding, lists) to make responses easy to read. 
        CRITICAL: When providing code snippets, always wrap the code in three backticks (```) 
        specifying the language (e.g., ```python) and ensure there is an empty line before the next paragraph.
        """
        
        # Initialize the model with the key from the environment variable
        llm = ChatGoogleGenerativeAI(
            # Using the less-restricted model for better free tier quota limits
            model="gemini-2.5-flash-lite", 
            temperature=0.2,
            api_key=api_key, 
            system_instruction=system_prompt
        )
        print("AI Model and Flask server initialized successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not initialize AI component with API key. Error: {e}")
    llm = None 

# --- Routes ---

@app.route('/', methods=['GET'])
def serve_index():
    """Serves the main HTML page."""
    # Assuming index.html is in the same directory as app.py
    return send_file('index.html')

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Handles the incoming chat request from the website."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    if not llm:
        return jsonify({"response": "Server setup error: Gemini API key is missing or invalid."}), 503

    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Please provide a message."}), 400
    
    try:
        # IMPORTANT: The front-end is sending the full conversation history as 'message',
        # so LangChain processes the full context here.
        response = llm.invoke(user_input)
        return jsonify({"response": response.content})
    except Exception as e:
        print(f"Error during AI invocation: {e}")
        # Check for quota error specifically
        if "RESOURCE_EXHAUSTED" in str(e):
             return jsonify({"response": "I apologize, the free tier usage quota has been exceeded. Please try again later or consider enabling billing for higher limits."}), 500
        return jsonify({"response": "I apologize, there was an issue processing your request. Please try again."}), 500

# --- Production Entry Point ---
if __name__ == '__main__':
    # This block is for LOCAL testing only.
    # In production (Render), Gunicorn will call the 'app' instance directly.
    # We use '0.0.0.0' to force binding to all interfaces for better local testing compatibility.
    # We use a default port of 5000, but Gunicorn/Render will use $PORT
    app.run(host='0.0.0.0', port=5000, debug=True)

# NOTE: No changes are needed to index.html for this deployment.