import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from groq import Groq

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "mp3", "wav", "m4a", "ogg", "flac", "mpeg", "mpga"}

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nome de arquivo vazio."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Formato não suportado."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(filepath)

    try:
        with open(filepath, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(filename, f),
                model="whisper-large-v3-turbo",
                language="pt",
                response_format="verbose_json",
            )

        segments = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in (result.segments or [])
        ]
        return jsonify({"transcript": result.text, "segments": segments})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
