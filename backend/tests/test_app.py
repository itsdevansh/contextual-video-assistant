import unittest
import json
from pathlib import Path
from app import app
import io

class TestBackendAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        
    def test_upload_no_file(self):
        response = self.app.post('/upload')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'No file part')
        
    def test_upload_empty_file(self):
        data = {'file': (io.BytesIO(b''), 'test.mp4')}
        response = self.app.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Empty file')
        
    def test_ask_no_data(self):
        response = self.app.post('/ask')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'No data provided')

    def test_ask_missing_params(self):
        response = self.app.post('/ask', json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Missing job_id or question')

if __name__ == '__main__':
    unittest.main()
