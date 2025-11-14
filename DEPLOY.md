# 🚀 Deployment Guide

This guide will help you deploy your Lung Disease Classifier app to Streamlit Cloud (recommended) or other platforms.

## 📋 Quick Deployment Checklist

- [ ] Code is ready and tested locally
- [ ] Model file is accessible (196MB - needs special handling)
- [ ] All dependencies in `requirements.txt`
- [ ] GitHub repository created
- [ ] Deployment platform account set up

---

## 🌟 Option 1: Streamlit Cloud (Easiest - Recommended)

**Best for**: Quick deployment, free tier available

### Step 1: Prepare Your GitHub Repository

1. **Create a GitHub repository** (if you don't have one)
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **Handle the Large Model File (196MB)**

   **Option A: Use Git LFS (Recommended)**
   ```bash
   # Install Git LFS
   brew install git-lfs  # macOS
   # or download from: https://git-lfs.github.com/
   
   # Initialize Git LFS
   git lfs install
   
   # Track model files
   git lfs track "*.pth"
   git lfs track "models/*.pth"
   
   # Add the .gitattributes file
   git add .gitattributes
   git add models/best_model.pth
   git commit -m "Add model with Git LFS"
   git push
   ```

   **Option B: Upload Model to Cloud Storage**
   - Upload `models/best_model.pth` to Google Drive, Dropbox, or AWS S3
   - Get a direct download link
   - Use `app_deploy.py` which supports downloading the model
   - Set `MODEL_URL` environment variable in Streamlit Cloud

3. **Update .gitignore** (remove models exclusion if using Git LFS)
   ```bash
   # Comment out these lines if using Git LFS:
   # models/
   # *.pth
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Fill in the details**:
   - **Repository**: Select your repository
   - **Branch**: `main` (or `master`)
   - **Main file path**: `app.py`
   - **Python version**: 3.9 or 3.10
5. **Click "Deploy"**

### Step 3: Configure Environment Variables (if using external model)

1. In Streamlit Cloud, go to **Settings** → **Secrets**
2. Add:
   ```toml
   MODEL_URL = "https://your-direct-download-link/models/best_model.pth"
   ```
3. Update `app.py` to use `app_deploy.py` or add model download logic

### Step 4: Wait for Deployment

- Streamlit Cloud will automatically install dependencies
- Your app will be available at: `https://YOUR_APP_NAME.streamlit.app`

---

## 🐳 Option 2: Docker Deployment

**Best for**: Full control, production environments

### Step 1: Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Build and Run

```bash
# Build image
docker build -t lung-disease-classifier .

# Run container
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  lung-disease-classifier
```

### Step 3: Deploy to Cloud

**AWS ECS/Fargate**:
```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin YOUR_ECR_URL
docker tag lung-disease-classifier:latest YOUR_ECR_URL/lung-disease-classifier:latest
docker push YOUR_ECR_URL/lung-disease-classifier:latest
```

**Google Cloud Run**:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/lung-disease-classifier
gcloud run deploy --image gcr.io/YOUR_PROJECT/lung-disease-classifier
```

---

## ☁️ Option 3: Heroku

**Best for**: Simple PaaS deployment

### Step 1: Install Heroku CLI

```bash
# macOS
brew install heroku/brew/heroku

# Or download from: https://devcenter.heroku.com/articles/heroku-cli
```

### Step 2: Prepare Files

1. **Procfile** (already created):
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **setup.sh** (already created):
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]\nheadless = true\nport = \$PORT\n" > ~/.streamlit/config.toml
   ```

3. **Update requirements.txt** (add gunicorn if needed):
   ```
   gunicorn
   ```

### Step 3: Deploy

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

**Note**: Heroku has a 500MB slug limit. If your model is too large, use external storage.

---

## 🖥️ Option 4: VPS/Cloud Server (AWS EC2, DigitalOcean, etc.)

**Best for**: Full control, cost-effective for long-term use

### Step 1: Set Up Server

```bash
# SSH into your server
ssh user@your-server-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install python3 python3-pip -y

# Install dependencies
pip3 install -r requirements.txt
```

### Step 2: Run App

```bash
# Option A: Direct run (for testing)
streamlit run app.py --server.port=8501 --server.address=0.0.0.0

# Option B: Using systemd (for production)
sudo nano /etc/systemd/system/streamlit-app.service
```

**systemd service file**:
```ini
[Unit]
Description=Streamlit Lung Disease Classifier
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/your-app
ExecStart=/usr/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

### Step 3: Configure Firewall

```bash
# Allow port 8501
sudo ufw allow 8501/tcp
sudo ufw enable
```

### Step 4: Set Up Reverse Proxy (Optional - for HTTPS)

Use Nginx or Apache to proxy requests and add SSL certificate.

---

## 🔧 Troubleshooting

### Issue: Model file too large for GitHub

**Solutions**:
1. Use Git LFS (recommended)
2. Upload to cloud storage and use `MODEL_URL` environment variable
3. Use `app_deploy.py` which supports model download

### Issue: App crashes on startup

**Check**:
1. All dependencies in `requirements.txt`
2. Model file exists and path is correct
3. Python version compatibility (3.9+)
4. Memory limits (model needs ~200MB RAM)

### Issue: Slow loading

**Solutions**:
1. Model is cached with `@st.cache_resource` (already implemented)
2. Use CDN for static files
3. Optimize model size (quantization)
4. Upgrade to faster server

### Issue: Port already in use

**Solution**:
```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 <PID>
```

---

## 📝 Deployment Checklist

- [ ] Code tested locally
- [ ] All dependencies in `requirements.txt`
- [ ] Model file accessible (Git LFS or external storage)
- [ ] Environment variables set (if needed)
- [ ] `.streamlit/config.toml` configured
- [ ] Security settings reviewed
- [ ] Domain/URL configured (if needed)
- [ ] SSL certificate installed (for production)
- [ ] Monitoring/logging set up (optional)

---

## 🎯 Recommended: Streamlit Cloud

**Why Streamlit Cloud?**
- ✅ Free tier available
- ✅ Automatic deployments
- ✅ Easy setup
- ✅ No server management
- ✅ Built-in SSL
- ✅ GitHub integration

**Get started**: https://streamlit.io/cloud

---

## 📚 Additional Resources

- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-community-cloud
- Docker Docs: https://docs.docker.com/
- Heroku Docs: https://devcenter.heroku.com/
- Git LFS Docs: https://git-lfs.github.com/

---

## 💡 Tips

1. **For production**: Use external model storage (S3, GCS) for faster deployments
2. **For testing**: Streamlit Cloud is perfect
3. **For scale**: Use Docker on cloud platforms (AWS, GCP, Azure)
4. **For cost**: VPS is cost-effective for long-term use

---

## 🆘 Need Help?

If you encounter issues:
1. Check the error logs in your deployment platform
2. Verify all dependencies are installed
3. Ensure model file is accessible
4. Check Python version compatibility
5. Review Streamlit documentation

Good luck with your deployment! 🚀

