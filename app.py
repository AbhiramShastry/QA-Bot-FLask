import os
import json
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain_openai import OpenAI
from dotenv import load_dotenv

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain




# Load environment variables from a .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

# Set up OpenAI API key securely using environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
llm = OpenAI(api_key=api_key)

# Create a prompt template for formatting the questions
question_prompt = PromptTemplate(
    input_variables=["question", "document"],
    template="Given the following document, answer the question: {question}\nDocument: {document}"
)

# Create a chain that connects the prompt with the LLM
llm_chain = LLMChain(prompt=question_prompt, llm=llm)

# Function to answer a question using LLM and the loaded document content
def answer_question(question, document_content):
    # Format the prompt with the question and document content
    return llm_chain.run(question=question, document=document_content)


# Ensure you have a folder to store uploaded files temporarily
UPLOAD_FOLDER = '/workspaces/QA-Bot-FLask/venv/'
ALLOWED_EXTENSIONS = {'pdf', 'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if a file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load document based on type
def load_document(file_path, file_type):
    # Ensure file_type is valid and process accordingly
    if file_type == 'pdf':
        # PyPDFLoader expects a file path as a string
        loader = PyPDFLoader(file_path)
    elif file_type == 'json':
        # JSONLoader also expects a file path
        loader = JSONLoader(file_path)
    else:
        raise ValueError("Unsupported file type.")
    
    # Load and return the document's contents
    return loader.load()


# Route for the home page
@app.route('/')
def index():
    return "Welcome to the Question-Answering Bot API. Use /upload to POST questions and documents.", 200

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        question_file = request.files.get('questions')
        document_file = request.files.get('document')

        # Validate received files
        if not question_file or not document_file:
            return jsonify({"error": "Both 'questions' and 'document' files are required."}), 400

        if not allowed_file(question_file.filename) or not allowed_file(document_file.filename):
            return jsonify({"error": "Unsupported file type. Only PDF and JSON are allowed."}), 400

        # Log received files
        logging.info(f"Received files: {question_file.filename}, {document_file.filename}")

        # Read and parse the JSON content of the questions file
        questions = json.loads(question_file.read())
        logging.info(f"Questions: {questions}")

        # Save document temporarily to server
        document_filename = secure_filename(document_file.filename)
        document_path = os.path.join(app.config['UPLOAD_FOLDER'], document_filename)
        document_file.save(document_path)

        # Log document loaded successfully
        logging.info(f"Document saved temporarily at {document_path}")
        
        # Load document
        file_type = document_filename.rsplit('.', 1)[1].lower()
        document = load_document(document_path, file_type)
        
        # Log document content
        logging.info("Document content loaded successfully.")
        
        answers = {}
        for question in questions:
            answer = answer_question(question, document)  # Get answer for each question
            answers[question] = answer

        # Clean up by removing the temporary document file
        os.remove(document_path)

        # Return the answers in the response
        return jsonify({"answers": answers}), 200

    except Exception as e:
        logging.error(f"Error processing files: {e}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
