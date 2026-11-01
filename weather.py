import requests

WEATHER_URL = "https://www.7timer.info/bin/api.pl"


def _description(cloudcover, prec_type):
    if prec_type in {"rain", "shower"}:
        return "Rainy", "🌧️"
    if prec_type in {"snow", "ice"}:
        return "Snow", "❄️"

    if cloudcover is None:
        return "Unknown", "☁️"
    if cloudcover <= 2:
        return "Clear", "☀️"
    if cloudcover <= 5:
        return "Partly Cloudy", "⛅"
    if cloudcover <= 7:
        return "Cloudy", "☁️"
    return "Very Cloudy", "☁️"


def get_weather(lat: float, lon: float):
    params = {
        "lon": round(float(lon), 3),
        "lat": round(float(lat), 3),
        "product": "civil",
        "output": "json",
        "unit": "metric",
    }

    response = requests.get(WEATHER_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    series = data.get("dataseries") or []
    if not series:
        raise ValueError("7Timer returned no weather data")

    current = series[0]
    temp = current.get("temp2m")
    rh = current.get("rh2m")
    cloudcover = current.get("cloudcover")
    prec_type = current.get("prec_type", "none")
    wind = current.get("wind10m") or {}
    wind_speed = wind.get("speed")

    # 7Timer's rh2m is a categorical humidity scale (0-15), not a direct %.
    # Convert it to an approximate percentage for the UI.
    humidity = None
    if isinstance(rh, (int, float)):
        humidity = round(float(rh) / 15 * 100)

    description, icon = _description(cloudcover, prec_type)

    forecast = []
    # 7Timer 'civil' gives 8 readings per day (every 3 hours)
    for i in range(0, min(len(series), 40), 8):
        day_data = series[i:i+8]
        if not day_data:
            break
        
        temps = [d.get("temp2m") for d in day_data if d.get("temp2m") is not None]
        max_temp = max(temps) if temps else "--"
        min_temp = min(temps) if temps else "--"
        
        mid_day = day_data[len(day_data)//2]
        desc, f_icon = _description(mid_day.get("cloudcover"), mid_day.get("prec_type", "none"))
        
        forecast.append({
            "day_index": i // 8,
            "max_temp": max_temp,
            "min_temp": min_temp,
            "description": desc,
            "icon": f_icon
        })

    return {
        "temperature": f"{temp}°C" if temp is not None else "--°C",
        "description": description,
        "humidity": f"{humidity}%" if humidity is not None else "--%",
        "wind_speed": f"{wind_speed} km/h" if wind_speed is not None else "-- km/h",
        "rain_probability": "Rain possible" if prec_type != "none" else "0%",
        "precipitation_type": prec_type,
        "icon": icon,
        "timepoint": current.get("timepoint"),
        "forecast": forecast,
    }