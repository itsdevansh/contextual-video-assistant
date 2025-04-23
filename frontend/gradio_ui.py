import gradio as gr
import requests
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5050")

def check_backend():
    """Check if backend server is accessible"""
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        return response.status_code == 200
    except:
        return False

def process_upload(video):
    if video is None:
        return "", "Please upload a video file"
        
    try:
        files = {
            'file': (
                Path(video).name,
                open(video, 'rb'),
                'video/mp4'
            )
        }
        response = requests.post(f"{BACKEND_URL}/upload", files=files)
        response.raise_for_status()
        
        job_id = response.json()["job_id"]
        return job_id, "Video processed successfully!"
        
    except requests.exceptions.ConnectionError:
        return "", "Error: Cannot connect to backend server. Please ensure the backend is running."
    except Exception as e:
        return "", f"Error: {str(e)}"

def ask_question(job_id, question):
    if not job_id:
        return "Please upload and process a video first"
    if not question:
        return "Please enter a question"
        
    try:
        response = requests.post(
            f"{BACKEND_URL}/ask",
            json={"job_id": job_id, "question": question}
        )
        response.raise_for_status()
        
        return response.json()["answer"]
        
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to backend server"
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Video Question Answering System") as demo:
    gr.Markdown("# Video Question Answering System")
    gr.Markdown("Upload a video and ask questions about its content.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            upload_button = gr.Button("Process Video")
            job_id_output = gr.Textbox(label="Job ID", interactive=False)
            status_output = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask something about the video...",
                autofocus=True
            )
            ask_button = gr.Button("Ask Question")
            answer_output = gr.Textbox(label="Answer", interactive=False)
    
    upload_button.click(
        process_upload,
        inputs=[video_input],
        outputs=[job_id_output, status_output]
    )
    
    # Add both button click and Enter key handlers
    ask_button.click(
        ask_question,
        inputs=[job_id_output, question_input],
        outputs=answer_output
    )
    
    question_input.submit(
        ask_question,
        inputs=[job_id_output, question_input],
        outputs=answer_output
    )

if __name__ == "__main__":
    if not check_backend():
        print("WARNING: Backend server is not accessible. Please start the backend server first.")
        print(f"Expected backend URL: {BACKEND_URL}")
    demo.launch(server_name="0.0.0.0", share=True)
