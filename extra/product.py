import os
import uuid
import json
import torch
import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import faiss
from dotenv import load_dotenv
import tempfile
import requests
# Load environment variables
load_dotenv()

class VideoProcessor:
    def __init__(self):
        self.device = "mps" if torch.mps.is_available() else "cpu"
        
        # Initialize models
        print("Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        
        print("Loading Sentence Transformer...")
        self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("Initializing OpenAI client...")
        self.openai_client = OpenAI()
        
        # Create data directories
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Store processed jobs
        self.jobs = {}

    def extract_significant_segments(self, video_path, fps=1, ssim_threshold=0.90):
        """Extract significant video segments with captions"""
        from skimage.metrics import structural_similarity as ssim
        import cv2
        
        segments = []
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps))
        
        frame_idx = 0
        current_caption = None
        current_start = None
        last_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_idx / video_fps
            
            if frame_idx % frame_interval == 0:
                frame_resized = cv2.resize(frame, (224, 224))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                if last_frame is not None:
                    similarity = ssim(last_frame, frame_rgb, channel_axis=-1)
                    if similarity >= ssim_threshold:
                        frame_idx += 1
                        continue
                        
                last_frame = frame_rgb.copy()
                
                # Generate caption with BLIP
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = self.blip_processor(images=pil_image, return_tensors="pt").to(self.device)
                out = self.blip_model.generate(**inputs)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                
                if caption != current_caption:
                    if current_caption is not None:
                        segments.append({
                            "caption": current_caption,
                            "start_time": current_start,
                            "end_time": timestamp
                        })
                    current_caption = caption
                    current_start = timestamp
                    
            frame_idx += 1
            
        cap.release()
        
        if current_caption:
            segments.append({
                "caption": current_caption,
                "start_time": current_start,
                "end_time": timestamp
            })
            
        return segments

    def transcribe_audio(self, video_path):
        """Transcribe audio using OpenAI Whisper"""
        video = VideoFileClip(str(video_path))
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
            audio_path = tmp_audio.name
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            
        with open(audio_path, "rb") as audio_file:
            transcription = self.openai_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
            
        os.remove(audio_path)
        return transcription.segments

    def process_video(self, video_path):
        """Process video and create searchable index"""
        # Generate unique job ID
        job_id = uuid.uuid4().hex
        
        try:
            # Extract video segments
            segments = self.extract_significant_segments(video_path)
            
            # Get audio transcription
            transcription = self.transcribe_audio(video_path)
            
            # Align transcripts with segments
            for segment in segments:
                matched_text = []
                for trans in transcription:
                    if trans.end >= segment["start_time"] and trans.start <= segment["end_time"]:
                        matched_text.append(trans.text.strip())
                segment["transcript"] = " ".join(matched_text)
            
            # Create FAISS index
            combined_texts = [f"{seg['caption']} {seg['transcript']}" for seg in segments]
            embeddings = self.st_model.encode(combined_texts, convert_to_numpy=True, normalize_embeddings=True)
            
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            
            # Save job data
            self.jobs[job_id] = {
                "index": index,
                "segments": segments
            }
            
            return job_id
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return None

    def answer_question(self, job_id, question, top_k=5):
        """Answer question about the video"""
        if job_id not in self.jobs:
            return "Please upload and process a video first."
            
        try:
            # Get relevant segments
            job = self.jobs[job_id]
            query_vec = self.st_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
            
            D, I = job["index"].search(query_vec, top_k)
            relevant_segments = [job["segments"][i] for i in I[0]]
            
            # Build context
            context = "\n\n".join([
                f"[{seg['start_time']:.2f} - {seg['end_time']:.2f}s]\n"
                f"Caption: {seg['caption']}\n"
                f"Transcript: {seg['transcript']}"
                for seg in relevant_segments
            ])
            
            # Generate answer with GPT
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant answering questions about a video using provided captions and transcripts."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error answering question: {e}")
            return f"Error: {str(e)}"

# Initialize processor
processor = VideoProcessor()

# Define Gradio interface
def process_upload(video):
    if video is None:
        return "", "Please upload a video file"
    return processor.process_video(video), "Video processed successfully!"

def ask_question(job_id, question):
    if not job_id:
        return "Please upload and process a video first"
    if not question:
        return "Please enter a question"
    return processor.answer_question(job_id, question)

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
            question_input = gr.Textbox(label="Your Question", placeholder="Ask something about the video...")
            ask_button = gr.Button("Ask Question")
            answer_output = gr.Textbox(label="Answer", interactive=False)
    
    upload_button.click(
        process_upload,
        inputs=[video_input],
        outputs=[job_id_output, status_output]
    )
    
    ask_button.click(
        ask_question,
        inputs=[job_id_output, question_input],
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)