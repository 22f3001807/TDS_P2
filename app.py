from flask import Flask, request, jsonify

from quiz_solver.config import QUIZ_SECRET
from quiz_solver.solver import run_quiz_session_sync

app = Flask(__name__)


@app.post("/quiz")
def quiz_endpoint():
    # 1. Parse JSON safely
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    # 2. Validate required fields
    required = {"email", "secret", "url"}
    if not required.issubset(payload.keys()):
        return jsonify({"error": "Missing required fields"}), 400

    email = payload.get("email")
    secret = payload.get("secret")
    start_url = payload.get("url")

    # 3. Check secret (403 if invalid)
    if secret != QUIZ_SECRET:
        return jsonify({"error": "Invalid secret"}), 403

    # 4. Run quiz session
    session_result = run_quiz_session_sync(start_url)

    # 5. Always 200 here if secret was valid
    return jsonify({
        "ok": True,
        "email": email,
        "start_url": start_url,
        "session_result": session_result,
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)