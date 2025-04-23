# Contextual Video Assistant

This project allows users to upload a video and ask natural language questions about its content. The system processes the video by extracting key visual segments, generating captions, transcribing the audio, and then uses AI models to answer user queries based on the combined visual and auditory information.

## Features

*   **Video Upload:** Accepts various video formats.
*   **Significant Segment Extraction:** Identifies visually distinct segments using frame analysis (SSIM) and sampling.
*   **Image Captioning:** Generates descriptive captions for key video frames using the BLIP model.
*   **Audio Transcription:** Transcribes the video's audio track into text with timestamps using OpenAI's Whisper model.
*   **Semantic Search:** Creates embeddings from captions and transcripts using Sentence Transformers and builds a FAISS index for efficient similarity search.
*   **Question Answering:** Uses OpenAI's GPT-4 model to synthesize answers based on the user's question and relevant video segments retrieved via semantic search.
*   **Web Interface:** Provides a user-friendly Gradio interface for uploading videos and asking questions.
*   **API Backend:** Built with Flask, providing endpoints for video processing and Q&A.
*   **Dockerized:** Includes Dockerfiles and a Docker Compose setup for easy deployment.

## Architecture

The application consists of two main components:

1.  **Backend (Flask):**
    *   Handles video uploads and storage.
    *   Orchestrates the video processing pipeline ([`backend/video_processor.py`](backend/video_processor.py)):
        *   Extracts frames and generates captions (BLIP).
        *   Extracts audio and transcribes (Whisper API).
        *   Aligns captions and transcripts.
        *   Generates text embeddings (Sentence Transformer).
        *   Builds and queries a FAISS index.
    *   Provides API endpoints (`/upload`, `/ask`, `/health`) ([`backend/app.py`](backend/app.py)).
    *   Interacts with OpenAI API for transcription and final answer generation.
2.  **Frontend (Gradio):**
    *   Provides the user interface ([`frontend/gradio_ui.py`](frontend/gradio_ui.py)).
    *   Allows users to upload videos.
    *   Sends processing requests to the backend API.
    *   Displays the processing status (`job_id`).
    *   Allows users to input questions and displays the answers received from the backend.

## Technology Stack

*   **Backend:** Python, Flask, Flask-CORS
*   **Frontend:** Gradio
*   **AI Models:**
    *   Image Captioning: Salesforce/blip-image-captioning-base
    *   Audio Transcription: OpenAI Whisper API (whisper-1)
    *   Text Embedding: all-MiniLM-L6-v2 (Sentence Transformers)
    *   Vector Search: FAISS (IndexFlatIP)
    *   Question Answering: OpenAI GPT-4 API
*   **Core Libraries:** transformers, sentence-transformers, faiss-cpu/faiss-gpu, openai, torch, Pillow, moviepy, opencv-python, scikit-image, numpy
*   **Deployment:** Docker, Docker Compose, Gunicorn
*   **Environment:** Python 3.10+

## Setup and Installation

### Prerequisites

*   Python 3.10+ and `pip`
*   Git
*   Docker and Docker Compose (for containerized running/deployment)
*   An OpenAI API Key

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd contextual-video-assistant
    ```

2.  **Set up Environment Variable:**
    Create a `.env` file in the project root directory:
    ```bash
    // filepath: .env
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    *(Note: This file is listed in [.gitignore](http://_vscodecontentref_/0) and should not be committed.)*

3.  **Install Dependencies (Option 1: Local Environment):**
    It's recommended to use a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r backend/requirements.txt
    pip install -r frontend/requirements.txt # Assuming a separate reqs file exists or combine them
    # Install PyTorch based on your system (CPU/CUDA/MPS): https://pytorch.org/get-started/locally/
    # Install FAISS (CPU or GPU version): https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
    ```

4.  **Install Dependencies (Option 2: Docker - Recommended for consistency):**
    Docker handles dependencies based on the `Dockerfile`s. Ensure Docker Desktop or Docker Engine is installed and running.

## Running the Application

### 1. Running Locally (using Python Virtual Environment)

*   **Start the Backend:**
    ```bash
    cd backend
    python app.py
    ```
    The backend API will be running, typically on `http://127.0.0.1:5050`.

*   **Start the Frontend:**
    Open a *new terminal*, activate the virtual environment, and run:
    ```bash
    cd frontend
    python gradio_ui.py
    ```
    The Gradio UI will be accessible, usually at `http://127.0.0.1:7860`.

### 2. Running with Docker Compose (Recommended)

This method builds and runs both the backend and frontend containers.

1.  **Ensure [.env](http://_vscodecontentref_/1) file is created** in the project root as described in Setup.
2.  **Build and Run:**
    ```bash
    docker compose up --build -d
    ```
    *   `-d` runs the containers in detached mode.
    *   `--build` forces a rebuild of the images if the code or Dockerfiles have changed.

3.  **Access:**
    *   Frontend (Gradio): `http://localhost:7860`
    *   Backend (Flask API): `http://localhost:5050`

4.  **Stop:**
    ```bash
    docker compose down
    ```

## Deployment (AWS EC2)

For deploying to an AWS EC2 instance, follow the detailed steps in [How_to_run.md](http://_vscodecontentref_/2). The general process involves:

1.  Launching an EC2 instance (consider a GPU instance like `g4dn` or `g5` for better performance if using local models heavily, though this setup relies heavily on APIs).
2.  SSHing into the instance.
3.  Installing Docker and Docker Compose.
4.  Cloning the project repository.
5.  Creating the [.env](http://_vscodecontentref_/3) file with your `OPENAI_API_KEY`.
6.  Running `docker compose up --build -d`.
7.  Configuring EC2 Security Groups to allow traffic on ports 7860 (frontend) and 5050 (backend).
8.  Accessing the application via the EC2 instance's public IP address.

## Usage

1.  Access the Gradio UI (e.g., `http://localhost:7860` or `http://<your-ec2-ip>:7860`).
2.  Upload a video file using the "Upload Video" component.
3.  Click "Process Video". Wait for the processing to complete. A [job_id](http://_vscodecontentref_/4) will be displayed upon success.
4.  Enter your question about the video content in the "Ask a Question" field.
5.  Click "Ask Question".
6.  The answer generated by the AI will appear in the "Answer" output field.

## Project Structure

```
contextual-video-assistant/
├── .env                        # OpenAI API key configuration
├── docker-compose.yml          # Docker compose configuration
├── README.md                   # Project documentation
├── How_to_run.md              # Deployment guide
├── setup.py                   # Python package configuration
├── pytest.ini                 # Test configuration
│
├── backend/                   # Backend service
│   ├── __init__.py
│   ├── app.py                # Flask application
│   ├── Dockerfile            # Backend container configuration
│   ├── requirements.txt      # Backend dependencies
│   ├── video_processor.py    # Video processing logic
│   ├── frame_processor.py    # Frame analysis logic
│   ├── wsgi.py              # WSGI entry point
│   ├── data/                # Directory for video storage
│   │   └── .gitkeep
│   └── tests/               # Backend tests
│       ├── __init__.py
│       ├── test_app.py
│       └── test_video_processor.py
│
├── frontend/                 # Frontend service
│   ├── __init__.py
│   ├── Dockerfile           # Frontend container configuration
│   ├── requirements.txt     # Frontend dependencies
│   ├── gradio_ui.py        # Gradio interface
│   └── tests/              # Frontend tests
│       └── test_gradio_ui.py
│
└── extra/                   # Additional utilities
    ├── experiment.ipynb    # Experimental notebooks
    └── product.py         # Additional utilities
```

### Key Components

- **Backend Service (`/backend`)**: Flask-based API handling video processing and AI operations
  - `app.py`: Main Flask application
  - `video_processor.py`: Core video processing logic
  - `frame_processor.py`: Frame analysis and processing
  - `wsgi.py`: WSGI entry point for production
  - `data/`: Storage for uploaded videos and processed results

- **Frontend Service (`/frontend`)**: Gradio-based user interface
  - `gradio_ui.py`: Main Gradio interface implementation
  - `tests/`: Frontend test suite

- **Configuration Files**
  - `.env`: Environment variables (OpenAI API key)
  - `docker-compose.yml`: Docker services configuration
  - `setup.py`: Python package configuration
  - `pytest.ini`: Testing configuration

- **Additional Tools (`/extra`)**
  - `experiment.ipynb`: Jupyter notebook for experiments
  - `product.py`: Utility functions and helpers
