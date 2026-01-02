import os
import json
import time
from flask import Flask, request, jsonify, render_template_string
from google import genai
from google.genai import types
import PyPDF2
from io import BytesIO

# --- Configuration ---
# Set your API Key here
API_KEY = "AIzaSyDFA9cQBjO_V8sIMTXXREHwsCseDz8qLJQ" 
client = genai.Client(api_key=API_KEY)

app = Flask(__name__)

# --- HTML Template for UI ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered ATS System</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 min-h-screen p-8">
    <div class="max-w-3xl mx-auto bg-white p-8 rounded-xl shadow-md">
        <h1 class="text-3xl font-bold text-blue-600 mb-2">ATS RESUME CHECKER</h1>
        <p class="text-gray-600 mb-8">Upload a Resume (PDF) and Job Description to get an AI analysis.</p>

        <form id="atsForm" class="space-y-6">
            <div>
                <label class="block text-sm font-medium text-gray-700">Job Description</label>
                <textarea name="jd" id="jd" rows="5" class="mt-1 block w-full border-gray-300 rounded-md shadow-sm border p-2 focus:ring-blue-500 focus:border-blue-500" placeholder="Paste the job requirements here..."></textarea>
            </div>

            <div>
                <label class="block text-sm font-medium text-gray-700">Resume (PDF)</label>
                <input type="file" name="resume" id="resume" accept=".pdf" class="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100">
            </div>

            <button type="submit" id="submitBtn" class="w-full bg-blue-600 text-white py-3 px-4 rounded-md font-bold hover:bg-blue-700 transition">Analyze Resume</button>
        </form>

        <div id="loading" class="hidden mt-8 text-center text-blue-600 font-bold">Processing...</div>

        <div id="result" class="hidden mt-8 border-t pt-8">
            <h2 class="text-2xl font-bold mb-4">Analysis Result</h2>
            <div id="scoreContainer" class="mb-4">
                <span class="text-lg font-semibold">Match Score:</span>
                <span id="matchScore" class="text-3xl font-bold text-green-600">0%</span>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="p-4 bg-green-50 rounded-lg">
                    <h3 class="font-bold text-green-800">Matching Skills</h3>
                    <ul id="matchingSkills" class="list-disc ml-5 text-sm"></ul>
                </div>
                <div class="p-4 bg-red-50 rounded-lg">
                    <h3 class="font-bold text-red-800">Missing Skills</h3>
                    <ul id="missingSkills" class="list-disc ml-5 text-sm"></ul>
                </div>
            </div>
            <div class="mt-4 p-4 bg-blue-50 rounded-lg">
                <h3 class="font-bold text-blue-800">Suggestions</h3>
                <p id="suggestions" class="text-sm"></p>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('atsForm').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData();
            formData.append('jd', document.getElementById('jd').value);
            formData.append('resume', document.getElementById('resume').files[0]);

            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('result').classList.add('hidden');
            document.getElementById('submitBtn').disabled = true;

            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await response.json();
                
                if(data.error) {
                   throw new Error(data.error);
                }

                document.getElementById('matchScore').innerText = data.match_percentage + '%';
                
                const matchUl = document.getElementById('matchingSkills');
                matchUl.innerHTML = '';
                data.matching_skills.forEach(s => { const li = document.createElement('li'); li.innerText = s; matchUl.appendChild(li); });

                const missUl = document.getElementById('missingSkills');
                missUl.innerHTML = '';
                data.missing_skills.forEach(s => { const li = document.createElement('li'); li.innerText = s; missUl.appendChild(li); });

                document.getElementById('suggestions').innerText = data.suggestions;
                
                document.getElementById('result').classList.remove('hidden');
            } catch (err) {
                // Modified error handling to be more descriptive for quota issues
                alert(err.message || "Error processing your request.");
            } finally {
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('submitBtn').disabled = false;
            }
        };
    </script>
</body>
</html>
"""

# --- Helper Functions ---

def extract_text_from_pdf(pdf_file):
    """Extracts raw text from an uploaded PDF file object."""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def get_gemini_analysis(resume_text, job_description):
    """Calls Gemini API using google-genai SDK with exponential backoff and error handling for 429s."""
    # Using 1.5 Flash as it's generally more stable for free-tier quotas than 2.0 during peak times
    model_id = 'gemini-2.5-flash'
    
    system_prompt = """
    You are an expert Technical Recruiter and ATS (Applicant Tracking System). 
    Your task is to analyze the provided Resume Text against a Job Description.
    Compare them based on skills, experience, and tools.
    You must respond ONLY with a JSON object.
    """

    user_query = f"Job Description: {job_description}\n\nResume Text: {resume_text}"
    
    # Define the response schema
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "match_percentage": {"type": "INTEGER"},
            "matching_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
            "missing_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
            "strengths": {"type": "STRING"},
            "suggestions": {"type": "STRING"}
        },
        "required": ["match_percentage", "matching_skills", "missing_skills", "suggestions"]
    }

    # Exponential Backoff Implementation
    retries = 5
    for i in range(retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[user_query],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=response_schema
                )
            )
            return json.loads(response.text)
        except Exception as e:
            error_str = str(e)
            # If we hit a rate limit (429), check if it's the last retry
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if i < retries - 1:
                    # Wait longer for quota issues: 2s, 4s, 8s, 16s...
                    time.sleep(2 ** (i + 1))
                    continue
                else:
                    return {"error": "Gemini API Quota Exceeded. Please wait a minute before trying again or check your billing settings."}
            
            if i < retries - 1:
                time.sleep(1)
            else:
                return {"error": f"AI service error: {error_str}"}

# --- Routes ---

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    if not API_KEY:
        return jsonify({"error": "API Key not configured in app.py"}), 400

    jd = request.form.get('jd')
    resume_file = request.files.get('resume')

    if not jd or not resume_file:
        return jsonify({"error": "Missing Job Description or Resume file"}), 400

    # 1. Extract Text
    resume_text = extract_text_from_pdf(resume_file)
    
    if "Error extracting PDF" in resume_text:
        return jsonify({"error": resume_text}), 500

    # 2. Get AI Analysis
    analysis = get_gemini_analysis(resume_text, jd)
    
    return jsonify(analysis)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)