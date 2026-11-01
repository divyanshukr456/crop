# Farmer Helps v2 (React + TypeScript)

- `backend/` — your original FastAPI + ML model + JSON database (unchanged).
- `frontend/` — new mobile-first React + TypeScript UI (Vite).

## Run locally
```bash
# backend (Python 3.11/3.12), from this folder
pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# frontend
cd frontend && cp .env.example .env && npm install && npm run dev
```
## Free-tier deploy
- Frontend: Vercel / Netlify / Cloudflare Pages (set `VITE_API_URL` to your backend URL).
- Backend: TensorFlow needs ~1 GB RAM, so use Hugging Face Spaces (free CPU, Docker). Set `GEMINI_API_KEY` there for Ask AI.
