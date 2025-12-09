# 🛠️ Running the API Locally
The prediction service is run using Uvicorn on the Fast_api.app:app module path.

### 1. Run the Server
Execute the following command from the project root (lunar-crater-age-classifier):

Bash
`# The '--reload' flag is used for local development.`
`# If your application file is 'Fast_api/app.py', use the following command:`
`uvicorn Fast_api.app:app --reload --port 8000`

### 2. Test the Health Endpoint
Once the server starts (it runs on http://127.0.0.1:8000), verify it is operational.

Bash
`curl http://127.0.0.1:8000/health`
`# Expected Output: {"status":"ok","message":"Service operational.","environment":"local"}`
