"""
Flask application for the RAG-from-scratch demo.

Routes:
    GET  /            -> serves the frontend page
    POST /api/ask      -> runs the full with/without-retrieval comparison
    GET  /api/health    -> liveness check

Models are loaded lazily on first use per embedder backend (see
RagPipeline), so the server starts instantly and only pays the model
load cost the first time each backend is actually requested.
"""

import logging
import os

from flask import Flask, jsonify, render_template, request

from src.pipeline import RagPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
pipeline = RagPipeline()

VALID_EMBEDDERS = {"bert", "dpr"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "")
    embedder_name = payload.get("embedder", "bert")
    k = payload.get("top_k", 5)

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Please provide a non-empty 'question' string."}), 400

    if embedder_name not in VALID_EMBEDDERS:
        return jsonify({
            "error": f"'embedder' must be one of {sorted(VALID_EMBEDDERS)}, got '{embedder_name}'."
        }), 400

    try:
        k = int(k)
        if not (1 <= k <= 10):
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "'top_k' must be an integer between 1 and 10."}), 400

    try:
        result = pipeline.ask(question, embedder_name=embedder_name, k=k)
        return jsonify(result)
    except FileNotFoundError as e:
        logger.error("Corpus/model file missing: %s", e)
        return jsonify({"error": "Server data files are missing. Check corpus.txt exists."}), 500
    except Exception as e:
        # Broad catch is deliberate here: this is a user-facing demo endpoint,
        # and a stack trace leaking to the browser is worse than a clean
        # 500 message. The real exception is still logged server-side.
        logger.exception("Unexpected error handling /api/ask")
        return jsonify({"error": "Something went wrong processing that question. Please try again."}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
