import os
import tempfile

from flask import Flask, request, jsonify
from gradio_client import Client, handle_file

app = Flask(__name__)

# env variables that will setup a hf space connection
HF_SPACE = os.environ["HF_SPACE"]
HF_TOKEN = os.environ["HF_TOKEN"]

client = Client(
    HF_SPACE,
    token=HF_TOKEN,
    verbose=False
)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True
    })


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("audio")

    if not file:
        return jsonify({
            "success": False,
            "error": "No audio file provided"
        }), 400

    temp_path = None

    try:
        suffix = os.path.splitext(
            file.filename or "recording.m4a"
        )[1]

        if not suffix:
            suffix = ".m4a"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)

        result = client.predict(
            handle_file(temp_path)
            # api_name="/predict"
        )

        if isinstance(result, tuple):
            result = result[0]

        if isinstance(result, dict):
            return jsonify(result)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        print(f"Prediction error: {e}")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(
        host="0.0.0.0",
        port=port
    )
