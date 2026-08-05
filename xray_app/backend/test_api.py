## Quick standalone test script to sanity-check the API without the UI
# save as backend/test_api.py, run with: py -3.11 test_api.py
import requests
import time

test_image_path = r"F:\KDU projects (extra)\Data odessey 2026-my proposals\XRAY\archive (1)\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg"

start = time.time()

with open(test_image_path, "rb") as f:
    files = {"file": ("test.jpg", f, "image/jpeg")}
    response = requests.post("http://127.0.0.1:8000/predict", files=files)

elapsed = time.time() - start

print(f"Status code: {response.status_code}")
print(f"Time taken: {elapsed:.2f} seconds")
print(response.json())