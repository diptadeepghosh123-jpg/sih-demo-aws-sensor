import os
import requests
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("WINDY_API_KEY")

print("Key loaded:", bool(key))
print("Key length:", len(key) if key else 0)

url = "https://api.windy.com/api/point-forecast/v2"

payload = {
    "lat": 22.5726,
    "lon": 88.3639,
    "model": "gfs",
    "parameters": [
        "temp",
        "rh",
        "pressure"
    ],
    "levels": [
        "surface"
    ],
    "key": key
}

response = requests.post(
    url,
    json=payload,
    timeout=20
)

print("HTTP status:", response.status_code)
print("Response:")
print(response.text)