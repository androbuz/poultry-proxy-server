import os
import requests

from flask import Flask, request, jsonify

app = Flask(__name__)
# the env variables to set that will allow connection to the real server
HF_SPACE_URL = os.environ["HF_SPACE_URL"]
HF_TOKEN = os.environ["HF_TOKEN"]

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("audio")

    if not file:
        return jsonify({
            "success": False,
            "error": "No audio file provided"
        }), 400

    try:
        response = requests.post(
            f"{HF_SPACE_URL}/gradio_api/call/predict",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}"
            },
            files={
                "audio": (
                    file.filename or "recording.m4a",
                    file.stream,
                    file.content_type or "audio/mp4"
                )
            },
            timeout=120
        )

        if response.status_code != 200:
            return jsonify({
                "success": False,
                "error": "Prediction service returned an error",
                "details": response.text
            }), response.status_code

        return jsonify(response.json())

    except requests.RequestException as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 502


@app.route("/health")
def health():
    return jsonify({
        "success": True
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(
        host="0.0.0.0",
        port=port
    )
