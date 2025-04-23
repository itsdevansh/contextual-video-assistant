import os
import uuid
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import faiss
import torch
from skimage.metrics import structural_similarity as ssim

# Initialize device
print("Checking available compute devices...")
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using CUDA GPU")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using Apple Silicon MPS")
else:
    DEVICE = "cpu"
    print("Using CPU")

# Initialize models at module level
print("Loading BLIP model...")
BLIP_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
BLIP_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

print("Loading Sentence Transformer...")
ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

class VideoProcessor:
    def __init__(self):
        self.device = DEVICE
        
        # Use global models
        self.blip_processor = BLIP_PROCESSOR
        self.blip_model = BLIP_MODEL
        self.st_model = ST_MODEL
        
        print("Initializing OpenAI client...")
        self.openai_client = OpenAI()
        
        # Create data directories
        self.data_dir = Path("data")
        self.videos_dir = self.data_dir / "videos"
        self.data_dir.mkdir(exist_ok=True)
        self.videos_dir.mkdir(exist_ok=True)
        
        # Store processed jobs
        self.jobs = {}

    def extract_significant_segments(self, video_path, fps=1, ssim_threshold=0.90):
        """Extract significant video segments with captions"""
        segments = []
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps))
        
        frame_idx = 0
        last_frame = None
        batch_size = 8  # Process frames in batches
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / video_fps
                
                if frame_idx % frame_interval == 0:
                    # Reduce memory usage by processing smaller frames
                    frame_resized = cv2.resize(frame, (160, 160))  # Even smaller resolution
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    
                    if last_frame is not None:
                        similarity = ssim(last_frame, frame_rgb, channel_axis=-1)
                        if similarity >= ssim_threshold:
                            frame_idx += 1
                            continue
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Generate caption with smaller batch size
                    with torch.no_grad():  # Reduce memory usage
                        inputs = self.blip_processor(pil_image, return_tensors="pt").to(self.device)
                        outputs = self.blip_model.generate(**inputs, max_length=30)  # Limit output length
                        caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
                    
                    segments.append({
                        'timestamp': timestamp,
                        'caption': caption,
                        'start_time': timestamp,
                        'end_time': timestamp + (1/fps)
                    })
                    
                    # Update last frame and clear memory
                    last_frame = frame_rgb.copy()
                    del frame_rgb
                    del pil_image
                    del inputs
                    del outputs
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Process in batches to avoid memory overflow
                    if len(segments) % batch_size == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                frame_idx += 1
                
        finally:
            cap.release()
            
        return segments

    def transcribe_audio(self, video_path):
        """Transcribe audio using OpenAI Whisper"""
        video = VideoFileClip(str(video_path))
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
            audio_path = tmp_audio.name
            video.audio.write_audiofile(audio_path)
            
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
        job_id = uuid.uuid4().hex
        
        try:
            # Log available memory
            if torch.cuda.is_available():
                print(f"GPU Memory before processing:")
                print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
                print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            
            # Extract video segments
            print(f"Starting video processing for {video_path}")
            segments = self.extract_significant_segments(video_path)
            print(f"Extracted {len(segments)} segments")
            
            # Get audio transcription
            print("Starting audio transcription")
            transcription = self.transcribe_audio(video_path)
            print(f"Transcription complete: {len(transcription)} segments")
            
            # Align transcripts with segments
            for segment in segments:
                matched_text = []
                for trans in transcription:
                    # Access start and end attributes correctly
                    trans_start = trans.start if hasattr(trans, 'start') else trans['start']
                    trans_end = trans.end if hasattr(trans, 'end') else trans['end']
                    trans_text = trans.text if hasattr(trans, 'text') else trans['text']
                    
                    if trans_end >= segment["start_time"] and trans_start <= segment["end_time"]:
                        matched_text.append(trans_text.strip())
                segment["transcript"] = " ".join(matched_text)
            
            # Create FAISS index
            print("Creating FAISS index")
            combined_texts = [f"{seg['caption']} {seg['transcript']}" for seg in segments]
            embeddings = self.st_model.encode(combined_texts, convert_to_numpy=True, normalize_embeddings=True)
            
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            
            # Save job data
            self.jobs[job_id] = {
                "index": index,
                "segments": segments
            }
            
            print(f"Processing complete. Job ID: {job_id}")
            return job_id
        
        except Exception as e:
            import traceback
            print(f"Error processing video:\n{traceback.format_exc()}")
            if torch.cuda.is_available():
                print(f"GPU Memory at error:")
                print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
            return None

    def answer_question(self, job_id, question, top_k=5):
        """Answer question about the video"""
        if job_id not in self.jobs:
            raise ValueError("Invalid job ID")
            
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
