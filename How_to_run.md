# 🚀 EC2 Deployment Guide for Flask + Gradio App (Docker + Docker Compose)

This guide walks you through setting up your Flask backend and Gradio frontend on a new EC2 instance using Docker and Docker Compose.

---

## 🧱 Prerequisites

- AWS EC2 instance (Ubuntu preferred)
- Open ports 7860 (Gradio) and 5050 (Flask) in the Security Group
- Your project repository hosted on GitHub (or similar)
- OpenAI API key

---

## 🛠️ Step-by-Step Setup

### 1. 🚀 SSH into Your EC2 Instance

```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

---

### 2. 🐳 Install Docker and Docker Compose

```bash
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common git
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Optional: Run Docker without sudo
sudo usermod -aG docker $USER
newgrp docker

# Check versions
docker --version
docker compose version
```

---

### 3. 📁 Clone Your Project

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

---

### 4. 🔐 Set Up Environment Variables

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Ensure your `docker-compose.yml` picks this up with:

```yaml
environment:
  - OPENAI_API_KEY=${OPENAI_API_KEY}
```

---

### 5. 🧱 Build and Run Containers

```bash
docker compose down
docker compose up -d --build
```

---

### 6. 🌐 Access Your App

- **Frontend (Gradio UI)**: `http://<your-ec2-ip>:7860`
- **Backend (Flask API)**: `http://<your-ec2-ip>:5050`

---

## 🧰 Helpful Commands

### 🔎 Check Running Containers

```bash
docker ps
```

### 🧾 View Logs

```bash
docker compose logs frontend
docker compose logs backend
```

### 🔁 Restart Services

```bash
docker compose down
docker compose up -d --build
```

### 🐚 Exec into a Container

```bash
docker exec -it <container_name> /bin/bash
```

### 🌐 Check Port Binding

```bash
sudo lsof -i -P -n | grep LISTEN
```

---

## 🧪 Gradio Binding Reminder

Ensure in your frontend code:

```python
gr.Interface(...).launch(server_name="0.0.0.0", server_port=7860)
```

Or via environment variables:

```yaml
environment:
  - GRADIO_SERVER_NAME=0.0.0.0
  - GRADIO_SERVER_PORT=7860
```

---

## 🔐 Final Notes

- Always use Elastic IPs to avoid losing your public IP.
- Restrict Security Group access if deploying publicly.
- Optional: Add an NGINX reverse proxy with SSL for production.

---

You're now ready to re-deploy your Flask + Gradio stack anytime 🚀

