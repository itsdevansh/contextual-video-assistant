FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend code
COPY . .

# Expose port for Gradio
EXPOSE 7860

# Run the Gradio app
CMD ["python", "gradio_ui.py"]