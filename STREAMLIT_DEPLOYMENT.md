# Streamlit Cloud Deployment Guide

## 🚀 Quick Start

### 1. Files Created for Deployment

- ✅ `runtime.txt` - Specifies Python 3.11
- ✅ `.streamlit/config.toml` - Streamlit server configuration
- ✅ `.streamlit/secrets.toml` - Template for API key (local only)
- ✅ `requirements.txt` - Updated with `openai==1.12.0`

### 2. Deploy to Streamlit Cloud

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit Cloud deployment config"
   git push origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Click "New app"
   - Connect your GitHub repository
   - Select branch: `main`
   - Main file path: `app.py`

3. **Add Secrets (IMPORTANT!)**
   - In Streamlit Cloud, go to: **Settings → Secrets**
   - Add your OpenAI API key:
   
   ```toml
   [openai]
   OPENAI_API_KEY = "sk-proj-your-actual-api-key-here"
   ```
   
   - Click "Save"

4. **Deploy!**
   - Click "Deploy"
   - Wait for deployment to complete

## 🔑 Configuration Structure

### API Key (Secrets)
- **Location**: Streamlit Cloud Secrets (or local `.streamlit/secrets.toml`)
- **Contains**: ONLY the OpenAI API key
- **Why**: Maximum security, never committed to git

### Other Settings (.env)
- **Location**: `.env` file in repository
- **Contains**: All other configuration (model, tokens, etc.)
- **Why**: Easy to change without exposing API key

## 🐛 Troubleshooting

### Error: "Client.__init__() got an unexpected keyword argument 'proxies'"

**Solution**: Already fixed!
- ✅ Updated `openai` version to 1.12.0 (stable)
- ✅ Added Python version pinning (3.11)
- ✅ Added error handling with helpful messages

### Error: "OpenAI API key not provided"

**Solution**:
1. Check Streamlit Cloud Settings → Secrets
2. Ensure the format is:
   ```toml
   [openai]
   OPENAI_API_KEY = "sk-proj-..."
   ```
3. Make sure there are no extra spaces or quotes

### App crashes on startup

**Solution**:
1. Check the Streamlit Cloud logs
2. Verify all dependencies in `requirements.txt` installed correctly
3. Check that data folders exist: `chatbot/data/entry/`, `exit/`, `target/`, `breadth/`

## 📊 Data Files

Your data structure is already set up:
```
chatbot/data/
├── entry/     (open positions)
├── exit/      (completed trades)
├── target/    (target achievements)
└── breadth/   (market breadth)
```

These folders will be included in your git repository and deployed with the app.

## ✅ Deployment Checklist

Before deploying:
- [ ] Secrets configured in Streamlit Cloud
- [ ] requirements.txt updated
- [ ] runtime.txt present
- [ ] .streamlit/config.toml present
- [ ] Data folders populated with CSV files
- [ ] .env file committed (no API key in it)

## 🎯 Post-Deployment

After successful deployment:
1. Test the chatbot with a simple query
2. Verify all 4 signal types work (Entry, Exit, Target, Breadth)
3. Check that API calls are successful
4. Monitor usage in OpenAI dashboard

## 📞 Support

If issues persist:
1. Check Streamlit Cloud logs
2. Verify API key is active and has credits
3. Ensure you're not hitting rate limits (30K TPM)
