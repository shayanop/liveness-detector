# 🚀 Hugging Face Spaces Deployment Instructions

## Files Required for Deployment

This folder contains all the necessary files to deploy the Advanced Video Liveness Detection System to Hugging Face Spaces:

### 📁 File Structure
```
hf_spaces_deployment/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore file
└── DEPLOYMENT_INSTRUCTIONS.md  # This file
```

## 🔧 Deployment Steps

### 1. Create Hugging Face Space

1. Go to [https://huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose a name for your space (e.g., "advanced-liveness-detection")
4. Select **Streamlit** as the SDK
5. Choose visibility (Public or Private)
6. Click "Create Space"

### 2. Upload Files

Upload all files from this folder to your Hugging Face Space:

- **app.py** → Root directory
- **requirements.txt** → Root directory  
- **README.md** → Root directory
- **.gitignore** → Root directory

### 3. Configuration

The space will automatically:
- Install dependencies from `requirements.txt`
- Run the Streamlit app from `app.py`
- Display the README.md as project documentation

## 📋 Requirements

The `requirements.txt` includes:
```
streamlit>=1.28.0
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
scipy>=1.11.0
Pillow>=10.0.0
```

## 🚀 Alternative Deployment Methods

### Using Git
```bash
git clone <your-hf-space-url>
cd <space-name>
cp /path/to/hf_spaces_deployment/* .
git add .
git commit -m "Deploy Advanced Liveness Detection System"
git push
```

### Using Hugging Face Hub
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="hf_spaces_deployment/",
    repo_id="your-username/your-space-name",
    repo_type="space"
)
```

## ✅ Testing the Deployment

Once deployed:
1. Wait for the space to build (usually 2-5 minutes)
2. The app should be accessible at: `https://huggingface.co/spaces/your-username/your-space-name`
3. Test with a sample video to ensure everything works

## 🔍 Troubleshooting

### Common Issues:

1. **Build Failures**: Check logs for dependency conflicts
2. **Import Errors**: Ensure all imports are in requirements.txt
3. **Memory Issues**: Large video files may exceed space limits
4. **Timeout**: Processing very long videos may timeout

### Solutions:
- Use `opencv-python-headless` if GUI issues occur
- Add `--upgrade pip` to requirements if needed
- Consider file size limits for uploaded videos

## 🛡️ Security Considerations

- Videos are processed in memory and not permanently stored
- No user data is collected or saved
- All processing happens on Hugging Face's infrastructure
- Consider adding rate limiting for production use

## 📊 Performance Notes

- Processing time depends on video length and resolution
- Recommended video specs: 640x480+, 15-60 FPS, 2-30 seconds
- Higher resolution provides better accuracy but slower processing

## 🔄 Updates and Maintenance

To update the deployment:
1. Modify files locally
2. Upload changed files to the space
3. Space will automatically rebuild and redeploy

---

**Ready to Deploy!** 🎉

All files are prepared and ready for Hugging Face Spaces deployment. Simply upload the contents of this folder to your new space and it should work flawlessly.