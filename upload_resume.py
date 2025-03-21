import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
app = Flask(__name__)

UPLOAD_FOLDER = "uploads/resumes"
ALLOWED_EXTENSIONS = {"pdf"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok = True)

def allowed_file(filename):
    extension = '.' in filename and filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS, extension

@app.route("/upload_resume", methods = ["POST"])
def upload_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded. Please attach a resume."}), 400
    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No file selected. Please choose a resume to upload."}), 400
    check, ext = allowed_file(file.filename)
    if file and check:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        return jsonify({"message": "Resume uploaded successfully", "file_path": file_path}), 200
    return jsonify({"error": f"Invalid file format. You uploaded a {ext} file!. (Only PDF resumes are allowed)"}), 400

if __name__ == "__main__":
    app.run(debug=True)
