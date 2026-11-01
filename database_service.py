"""Loads the supplied crop JSON files and finds disease information safely."""
from __future__ import annotations

import json
import re
from pathlib import Path


DATABASE_DIR = Path(__file__).resolve().parent / "database"


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


class DiseaseDatabase:
    def __init__(self) -> None:
        self.crops: dict[str, dict] = {}

    def load(self) -> None:
        self.crops.clear()
        for path in DATABASE_DIR.glob("*.json"):
            with path.open(encoding="utf-8") as source:
                data = json.load(source)
            crop = data.get("crop", {})
            crop_name = crop.get("crop_name")
            diseases = crop.get("diseases")
            if not crop_name or not isinstance(diseases, list):
                raise ValueError(f"Invalid disease database file: {path.name}")
            self.crops[_normalise(crop_name)] = crop
        if not self.crops:
            raise ValueError("No disease database files were found")

    def find(self, crop_name: str, disease_name: str) -> dict | None:
        crop = self.crops.get(_normalise(crop_name))
        if not crop:
            return None
        target = _normalise(disease_name)
        for disease in crop["diseases"]:
            if _normalise(disease.get("disease_name", "")) == target:
                return {"crop": crop, "disease": disease}
        return None


disease_database = DiseaseDatabase()
