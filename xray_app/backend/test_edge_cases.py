import requests

API_URL = "http://127.0.0.1:8000/predict"

test_cases = [
    ("normal_xray.jpg", "Valid X-ray — should succeed"),
    ("random_photo.jpg", "Non-X-ray photo — should still process, worth reviewing risk output"),
    ("blurry_xray.jpg", "Blurry image — should be rejected with blur message"),
    ("tiny_image.jpg", "Very low-res image — should be rejected with resolution message"),
    ("corrupted_file.jpg", "Corrupted file — should be rejected gracefully"),
    ("document.pdf", "Wrong file type — should be rejected at extension check"),
]

for filename, description in test_cases:
    print(f"\n--- Testing: {description} ---")
    try:
        with open(filename, "rb") as f:
            files = {"file": (filename, f, "image/jpeg")}
            response = requests.post(API_URL, files=files, timeout=15)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except FileNotFoundError:
        print(f"Skipped — create a test file named '{filename}' to run this case")
    except Exception as e:
        print(f"Unexpected error: {e}")