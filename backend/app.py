import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from video_processor import VideoProcessor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize video processor
processor = VideoProcessor()

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400
        
    try:
        # Create temporary directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save file and process
        from werkzeug.utils import secure_filename
        temp_path = os.path.join(temp_dir, secure_filename(file.filename))
        
        # Debug logging
        print(f"Saving file to: {temp_path}")
        file.save(temp_path)
        print(f"File saved successfully, processing video...")
        
        # Pass the file path directly instead of the file object
        job_id = processor.process_video(temp_path)
        print(f"Processing complete, job_id: {job_id}")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Temporary file removed: {temp_path}")
        
        if job_id:
            return jsonify({'job_id': job_id})
        else:
            print("Processing failed: No job_id returned")
            return jsonify({'error': 'Processing failed'}), 500
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error during upload/processing:\n{error_trace}")
        # Clean up on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Cleaned up temporary file: {temp_path}")
        
        # Check if error is related to CUDA/GPU
        if "CUDA" in str(e) or "GPU" in str(e):
            error_msg = "GPU error occurred. Ensure CUDA is properly configured on the instance."
        elif "memory" in str(e).lower():
            error_msg = "Out of memory error. Try reducing batch size or video resolution."
        else:
            error_msg = str(e)
            
        return jsonify({
            'error': error_msg,
            'details': error_trace
        }), 500

@app.route('/ask', methods=['POST'])
def ask():
    if not request.is_json:
        return jsonify({'error': 'No data provided'}), 400
        
    data = request.get_json()
    if not data or 'job_id' not in data or 'question' not in data:
        return jsonify({'error': 'Missing job_id or question'}), 400
        
    try:
        answer = processor.answer_question(data['job_id'], data['question'])
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5050)))
