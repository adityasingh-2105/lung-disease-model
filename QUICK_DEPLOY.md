# 🚀 Quick Deployment Guide

## Easiest Method: Streamlit Cloud (5 minutes)

### Step 1: Prepare GitHub Repository

1. **Initialize Git** (if not done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Handle the Model File (196MB)**

   **Option A: Use Git LFS (Recommended)**
   ```bash
   # Install Git LFS
   brew install git-lfs  # macOS
   # or: https://git-lfs.github.com/
   
   # Setup Git LFS
   git lfs install
   git lfs track "models/*.pth"
   git add .gitattributes
   git add models/best_model.pth
   git commit -m "Add model with Git LFS"
   ```

   **Option B: Temporarily Allow Model in Git**
   ```bash
   # Update .gitignore - comment out these lines:
   # models/
   # *.pth
   
   # Then add model
   git add models/best_model.pth
   git commit -m "Add model file"
   ```

3. **Create GitHub Repository**:
   ```bash
   # On GitHub, create a new repository
   # Then push:
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to**: https://share.streamlit.io/
2. **Sign in** with GitHub
3. **Click "New app"**
4. **Select**:
   - Repository: Your repository
   - Branch: `main`
   - Main file: `app.py`
   - Python version: `3.9` or `3.10`
5. **Click "Deploy"**
6. **Wait 2-3 minutes** for deployment
7. **Your app is live!** 🎉

### Step 3: Access Your App

Your app will be available at:
```
https://YOUR_APP_NAME.streamlit.app
```

---

## Troubleshooting

### Problem: Model file too large for GitHub

**Solution 1: Use Git LFS** (Recommended)
```bash
git lfs install
git lfs track "models/*.pth"
git add .gitattributes models/best_model.pth
git commit -m "Add model with Git LFS"
git push
```

**Solution 2: Upload Model to Cloud Storage**
1. Upload `models/best_model.pth` to Google Drive or Dropbox
2. Get a direct download link
3. Update `app.py` to download the model:
   ```python
   import urllib.request
   MODEL_URL = "YOUR_DOWNLOAD_LINK"
   if not os.path.exists("models/best_model.pth"):
       urllib.request.urlretrieve(MODEL_URL, "models/best_model.pth")
   ```

### Problem: Deployment fails

**Check**:
1. All files are committed to Git
2. `requirements.txt` includes all dependencies
3. Model file is accessible (if using Git LFS)
4. Python version is correct (3.9+)

### Problem: App runs but model not found

**Solution**:
1. Check if model file is in repository
2. Verify model path in `app.py`: `models/best_model.pth`
3. Check deployment logs in Streamlit Cloud

---

## Alternative: Manual Model Upload

If Git LFS doesn't work, you can:

1. **Deploy app without model** first
2. **Upload model manually** using Streamlit Cloud's file manager
3. **Or use external storage** and download at runtime

---

## Quick Checklist

- [ ] Code is in GitHub repository
- [ ] Model file is accessible (Git LFS or external storage)
- [ ] All dependencies in `requirements.txt`
- [ ] App tested locally
- [ ] Deployed to Streamlit Cloud
- [ ] App is working

---

## Need Help?

1. Check Streamlit Cloud logs
2. Verify all files are in repository
3. Ensure model file is accessible
4. Check Python version compatibility

**Deployment should take about 5 minutes!** 🚀

