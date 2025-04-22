import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from video_processor import VideoProcessor
import torch
import tempfile
import os

class TestVideoProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = VideoProcessor()
        
    def test_initialization(self):
        self.assertIsNotNone(self.processor.blip_processor)
        self.assertIsNotNone(self.processor.blip_model)
        self.assertIsNotNone(self.processor.st_model)
        self.assertIsNotNone(self.processor.openai_client)
        
    @patch('video_processor.VideoFileClip')
    @patch('video_processor.OpenAI')
    def test_transcribe_audio(self, mock_openai, mock_video_clip):
        # Mock video file and transcription
        mock_video = Mock()
        mock_video.audio.write_audiofile = Mock()
        mock_video_clip.return_value = mock_video
        
        mock_client = Mock()
        mock_transcription = Mock()
        mock_transcription.segments = [
            Mock(start=0, end=1, text="Test transcription")
        ]
        mock_client.audio.transcriptions.create.return_value = mock_transcription
        mock_openai.return_value = mock_client
        
        # Test transcription
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_video:
            result = self.processor.transcribe_audio(temp_video.name)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].text, "Test transcription")
            
    def test_answer_question_invalid_job(self):
        with self.assertRaises(ValueError):
            self.processor.answer_question("invalid_job_id", "test question")

if __name__ == '__main__':
    unittest.main()
