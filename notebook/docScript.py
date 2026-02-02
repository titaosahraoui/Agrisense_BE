import base64
import json
import subprocess
import requests
import os

# ---------- CONFIG ----------
PDF_PATH = "./notebook/SERIE-B-2019.pdf"  # Path to your PDF
PROJECT_ID = "312473731223"
PROCESSOR_ID = "6354d81fff6f3b8d"
LOCATION = "eu"
OUTPUT_JSON = "output.json"
# -----------------------------

def get_access_token():
    """Get access token using gcloud SDK."""
    print("🔑 Getting access token from gcloud...")
    try:
        token = subprocess.check_output(
            [
                r"C:\Users\titao\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
                "auth",
                "application-default",
                "print-access-token"
            ],
            text=True
        ).strip()
        print("✅ Access token retrieved!")
        return token
    except FileNotFoundError:
        print("❌ Could not find gcloud command. Please check your SDK path.")
        exit(1)

def pdf_to_base64(pdf_path):
    """Convert the PDF to base64."""
    print("📄 Encoding PDF to base64...")
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_document_ai(pdf_base64, token):
    """Send the document to Google Document AI."""
    url = f"https://{LOCATION}-documentai.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}:process"
    print(f"🚀 Sending request to: {url}")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }

    # ✅ Here's the request body you were missing
    request_body = {
        "skipHumanReview": True,
        "rawDocument": {
            "mimeType": "application/pdf",
            "content": pdf_base64
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(request_body))
    if response.status_code != 200:
        print("❌ Error:", response.text)
        response.raise_for_status()

    print("✅ Document processed successfully!")
    return response.json()

def save_json(data, filename):
    """Save JSON response to a file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Output saved to {filename}")

def main():
    pdf_base64 = pdf_to_base64(PDF_PATH)
    token = get_access_token()
    response_json = call_document_ai(pdf_base64, token)
    save_json(response_json, OUTPUT_JSON)

if __name__ == "__main__":
    main()
