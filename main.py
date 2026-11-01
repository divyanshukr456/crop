"""FastAPI entry point for the crop disease detection system."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .database_service import disease_database
from .location import get_location, search_location
from .model_service import PredictionError, crop_model
from .weather import get_weather as fetch_weather
from .voice_fixed import router as voice_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    disease_database.load()
    crop_model.load()
    yield


app = FastAPI(title="Farmer Helps API", lifespan=lifespan)
app.include_router(voice_router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def api_error(error_type: str, message: str, status_code: int) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"success": False, "error_type": error_type, "message": message})


@app.get("/")
def read_root():
    return {"message": "Farmer Helps API is running"}


@app.get("/location")
def location(lat: float, lon: float):
    try:
        return {"location": get_location(lat, lon), "weather": fetch_weather(lat, lon)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Location or weather service is unavailable") from exc


@app.get("/weather")
def weather(lat: float | None = None, lon: float | None = None, city: str | None = None):
    try:
        if city and (lat is None or lon is None):
            location_data = search_location(city)
            lat, lon = location_data["latitude"], location_data["longitude"]
        elif lat is None or lon is None:
            raise HTTPException(status_code=400, detail="Provide lat/lon or city")
        else:
            location_data = get_location(lat, lon)
        return {"location": location_data, "weather": fetch_weather(lat, lon)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Weather service is unavailable") from exc


@app.get("/api/weather")
def api_weather(lat: float | None = None, lon: float | None = None, city: str | None = None):
    return weather(lat, lon, city)


@app.post("/predict")
async def predict_disease(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise api_error("unsupported_file", "Please upload a PNG, JPG, WEBP, or other image file.", 415)
    try:
        prediction = crop_model.predict(await image.read())
        if prediction["disease"] == "Healthy":
            return {"success": True, "crop": prediction["crop"], "disease": "Healthy", "confidence": prediction["confidence"], "severity": "Healthy", "explanation": "The model's closest supported class is healthy.", "recommended_action": "Continue regular crop care and monitor for changes.", "prevention": "Use clean planting material and inspect plants regularly.", "database_match": False}
        match = disease_database.find(prediction["crop"], prediction["disease"])
        if not match:
            raise api_error("disease_not_found", "The predicted class does not have verified information in the disease database.", 422)
        disease = match["disease"]
        return {"success": True, "crop": match["crop"]["crop_name"], "disease": disease["disease_name"], "confidence": prediction["confidence"], "severity": "Needs assessment", "explanation": " ".join(disease.get("symptoms", [])), "recommended_action": " ".join(disease.get("treatment", [])), "prevention": " ".join(disease.get("prevention", [])), "database_match": True}
    except PredictionError as exc:
        raise api_error(exc.error_type, exc.message, exc.status_code) from exc
    except HTTPException:
        raise
    except Exception as exc:
        print(f"Prediction error ({type(exc).__name__}): {exc}")
        raise api_error("server_error", "The disease detector is temporarily unavailable. Please try again.", 500) from exc
