# Farmer Helps - FastAPI Gemini voice endpoint
#
# Install:
#   pip install fastapi uvicorn google-genai python-dotenv
#
# .env:
#   GEMINI_API_KEY=your_real_key_here
#
# IMPORTANT: Keep the Gemini API key on the backend.

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai


load_dotenv()

router = APIRouter()

# Accept GEMINI_API_KEY (recommended) and GOOGLE_API_KEY as a fallback.
api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()

client = genai.Client(api_key=api_key) if api_key else None
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class VoiceRequest(BaseModel):
    # "message" is the normal field. "text" is also accepted so the endpoint
    # works with frontends that send { "text": "..." }.
    message: str | None = Field(default=None)
    text: str | None = Field(default=None)
    language: str = "en-IN"
    context: str = "Farmer Helps agricultural assistant"

    def get_message(self) -> str:
        return (self.message or self.text or "").strip()


@router.post("/api/voice")
def voice_assistant(req: VoiceRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="Voice assistant is unavailable until GEMINI_API_KEY is configured.")
    user_message = req.get_message()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message/text is empty")

    language = (req.language or "en-IN").lower()

    if language.startswith("hi"):
        language_instruction = (
            "Answer in natural Hindi using Devanagari script. "
            "Keep technical terms simple and explain them in Hindi."
        )
    else:
        language_instruction = "Answer in clear, simple English."

    prompt = f"""
You are the Farmer Helps AI agricultural assistant.

{language_instruction}

Help farmers with practical, easy-to-understand information.
Topics can include weather, mandi rates, crop diseases, farming practices,
government schemes, and farmer helplines.

User question:
{user_message}

Give a concise, useful answer. Do not claim live data unless it was supplied
by the application's APIs. If the question requires live weather or mandi data,
say that the relevant live feature should be used rather than inventing values.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        answer = getattr(response, "text", None)

        if not answer or not answer.strip():
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response",
            )

        return {"answer": answer.strip()}

    except HTTPException:
        raise

    except Exception as exc:
        # Print the real error in the FastAPI terminal so debugging is possible,
        # while keeping the API response safe for the frontend.
        print(f"Gemini error ({type(exc).__name__}): {exc}")

        raise HTTPException(
            status_code=502,
            detail="Gemini request failed. Check the FastAPI terminal for the exact error.",
        ) from exc
