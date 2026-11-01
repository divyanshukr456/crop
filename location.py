import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
SEARCH_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {
    "User-Agent": "FarmerHelps/1.0 (weather-location-app)"
}


def _format_location(data, lat=None, lon=None):
    address = data.get("address", {})

    return {
        "latitude": lat,
        "longitude": lon,
        "display_name": data.get("display_name", "Unknown location"),
        "village": address.get("village"),
        "town": address.get("town"),
        "city": address.get("city") or address.get("town") or address.get("village"),
        "district": address.get("state_district") or address.get("county"),
        "state": address.get("state"),
        "country": address.get("country"),
        "postcode": address.get("postcode"),
    }


def get_location(lat: float, lon: float):
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "zoom": 18,
        "addressdetails": 1,
    }

    response = requests.get(
        NOMINATIM_URL,
        params=params,
        headers=HEADERS,
        timeout=10,
    )
    response.raise_for_status()
    return _format_location(response.json(), lat, lon)


def search_location(city: str):
    params = {
        "q": city,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
    }

    response = requests.get(
        SEARCH_URL,
        params=params,
        headers=HEADERS,
        timeout=10,
    )
    response.raise_for_status()

    results = response.json()
    if not results:
        raise ValueError(f"Location not found: {city}")

    result = results[0]
    lat = float(result["lat"])
    lon = float(result["lon"])
    return _format_location(result, lat, lon)
