from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import json
import requests

class FrameProcessor:
    def __init__(self, frames_dir="frames", fps=30):
        self.frames_dir = frames_dir
        self.fps = fps
        
        # Initialize BLIP-2
        print("Loading BLIP-2 model...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model.eval()
        
        # Process frames and generate captions
        self.frame_captions = self.process_frames()
        
    def process_frames(self):
        frame_captions = {}
        
        for frame_file in sorted(os.listdir(self.frames_dir)):
            if not frame_file.endswith(('.jpg', '.png')):
                continue
                
            image = Image.open(f"{self.frames_dir}/{frame_file}").convert('RGB')
            inputs = self.processor(image, return_tensors="pt")
            
            with torch.no_grad():
                out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            frame_number = int(frame_file.split('_')[1].split('.')[0])
            timestamp = frame_number / self.fps
            frame_captions[timestamp] = caption
            
        return frame_captions
        
    def save_captions(self, output_file="frame_captions.json"):
        with open(output_file, 'w') as f:
            json.dump(self.frame_captions, f, indent=2)
            
class OllamaClient:
    def __init__(self, model="gemma3:latest"):
        self.model = model
        self.api_url = "http://127.0.0.1:11434"
        
    def generate(self, prompt, context=None):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "context": context if context else [],
            "stream": False
        }
        
        try:
            response = requests.post(f"{self.api_url}/api/generate", json=payload)  # <- FIXED URL
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return None