import unittest
from unittest.mock import patch, Mock
import sys
from pathlib import Path
import requests

# Add parent directory to Python path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradio_ui import process_upload, ask_question, check_backend

class TestGradioUI(unittest.TestCase):
    def setUp(self):
        self.test_video = "test_video.mp4"
        
    @patch('requests.post')
    def test_process_upload_success(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"job_id": "test123"}
        mock_post.return_value = mock_response
        
        with patch('builtins.open', Mock()):
            job_id, status = process_upload(self.test_video)
            self.assertEqual(job_id, "test123")
            self.assertEqual(status, "Video processed successfully!")
        
    def test_process_upload_no_video(self):
        job_id, status = process_upload(None)
        self.assertEqual(job_id, "")
        self.assertEqual(status, "Please upload a video file")
        
    @patch('requests.post')
    def test_process_upload_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError()
        
        with patch('builtins.open', Mock()):
            job_id, status = process_upload(self.test_video)
            self.assertEqual(job_id, "")
            self.assertTrue("Cannot connect to backend server" in status)
        
    @patch('requests.post')
    def test_ask_question_success(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"answer": "Test answer"}
        mock_post.return_value = mock_response
        
        result = ask_question("test123", "What happens in the video?")
        self.assertEqual(result, "Test answer")
        
    def test_ask_question_no_job_id(self):
        result = ask_question("", "Test question")
        self.assertEqual(result, "Please upload and process a video first")
        
    def test_ask_question_no_question(self):
        result = ask_question("test123", "")
        self.assertEqual(result, "Please enter a question")
        
    @patch('requests.post')
    def test_ask_question_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError()
        result = ask_question("test123", "Test question")
        self.assertTrue("Cannot connect to backend server" in result)
        
    @patch('requests.get')
    def test_backend_check_success(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        self.assertTrue(check_backend())
        
    @patch('requests.get')
    def test_backend_check_failure(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()
        self.assertFalse(check_backend())

if __name__ == '__main__':
    unittest.main()
