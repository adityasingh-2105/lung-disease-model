# Deployment Guide

This guide explains how to deploy the Lung Disease Classifier application to various platforms.

## Option 1: Streamlit Cloud (Recommended - Free & Easy)

Streamlit Cloud is the easiest way to deploy Streamlit apps for free.

### Prerequisites
1. GitHub account
2. Streamlit Cloud account (free at https://streamlit.io/cloud)

### Steps

#### 1. Prepare Your Repository
- Make sure your code is in a GitHub repository
- Ensure `requirements.txt` is in the root directory
- Ensure `models/best_model.pth` is included (or uploaded separately)

#### 2. Upload Model to GitHub
**Important**: The model file is large (~196MB). You have two options:

**Option A: Use Git LFS (Recommended for large files)**
```bash
# Install Git LFS if not installed
git lfs install

# Track the model file
git lfs track "models/*.pth"

# Add and commit
git add .gitattributes
git add models/best_model.pth
git commit -m "Add model with Git LFS"
git push
```

**Option B: Use External Storage (Better for large files)**
- Upload model to Google Drive, Dropbox, or AWS S3
- Update `app.py` to download the model on first run
- See "Model Download Option" section below

#### 3. Deploy to Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Click "Sign in" and connect your GitHub account
3. Click "New app"
4. Select your repository
5. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.9 (or latest)
6. Click "Deploy"

#### 4. Configure Environment (if needed)
- Streamlit Cloud will automatically install packages from `requirements.txt`
- If you need system packages, add them to `packages.txt`

### Model Download Option (For Large Models)

If your model is too large for GitHub, you can download it at runtime:

1. Upload your model to a cloud storage (Google Drive, Dropbox, S3)
2. Update `app.py` to download the model:

```python
import streamlit as st
import os
import urllib.request

MODEL_URL = "https://your-storage-url/models/best_model.pth"
MODEL_PATH = "models/best_model.pth"

@st.cache_resource
def download_model():
    """Download model if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        st.info("Downloading model... This may take a few minutes.")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded successfully!")
    return MODEL_PATH

# In load_model function:
def load_model():
    download_model()  # Download if needed
    # ... rest of the code
```

---

## Option 2: Heroku

### Prerequisites
1. Heroku account
2. Heroku CLI installed

### Steps

1. **Create Procfile**:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. **Create setup.sh**:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml
```

3. **Update requirements.txt** (add if missing):
```
gunicorn
```

4. **Deploy**:
```bash
heroku create your-app-name
git push heroku main
heroku open
```

**Note**: Heroku has a 500MB slug size limit. If your model is too large, use external storage.

---

## Option 3: Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run
```bash
docker build -t lung-disease-classifier .
docker run -p 8501:8501 lung-disease-classifier
```

### Deploy to Docker Hub / AWS / GCP
```bash
docker tag lung-disease-classifier yourusername/lung-disease-classifier
docker push yourusername/lung-disease-classifier
```

---

## Option 4: Google Cloud Run

1. **Create Dockerfile** (same as above)

2. **Deploy**:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/lung-disease-classifier
gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/lung-disease-classifier --platform managed
```

---

## Option 5: AWS EC2 / Lightsail

1. **Launch an EC2 instance** (Ubuntu recommended)
2. **SSH into the instance**
3. **Install dependencies**:
```bash
sudo apt update
sudo apt install python3-pip -y
pip3 install -r requirements.txt
```

4. **Run the app**:
```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

5. **Configure security group** to allow port 8501
6. **Access**: `http://YOUR_EC2_IP:8501`

---

## Troubleshooting

### Issue: Model file too large for GitHub
**Solution**: Use Git LFS or external storage (see Model Download Option above)

### Issue: App crashes on deployment
**Solution**: 
- Check logs in deployment platform
- Ensure all dependencies are in `requirements.txt`
- Verify model path is correct

### Issue: Memory errors
**Solution**:
- Reduce batch size
- Use model quantization
- Upgrade to a platform with more memory

### Issue: Slow loading
**Solution**:
- Use `@st.cache_resource` for model loading (already implemented)
- Optimize model size
- Use CDN for static files

---

## Quick Deployment Checklist

- [ ] Code is in GitHub repository
- [ ] `requirements.txt` is up to date
- [ ] Model file is accessible (Git LFS or external storage)
- [ ] `app.py` is in the root directory
- [ ] All dependencies are listed in `requirements.txt`
- [ ] Tested locally before deployment
- [ ] Environment variables set (if any)
- [ ] Security settings configured

---

## Recommended: Streamlit Cloud

For easiest deployment, use **Streamlit Cloud**:
- Free tier available
- Automatic deployments from GitHub
- Easy to set up
- No server management needed

Visit: https://streamlit.io/cloud

