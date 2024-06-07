import argparse
import io
import os
import cv2
import time
from flask import Flask, render_template, request, Response, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    imgpath = None

    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            if f and f.filename:
                basepath = os.path.dirname(__file__)
                filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
                print("Upload folder is", filepath)
                f.save(filepath)
                imgpath = secure_filename(f.filename)

                file_extension = f.filename.rsplit('.', 1)[1].lower()

                if file_extension == 'jpg':
                    img = cv2.imread(filepath)
                    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                    # Perform the detection
                    yolo = YOLO('best.pt')
                    detections = yolo.predict(image, save=True)
                    return display(imgpath)

                elif file_extension == 'mp4':
                    video_path = filepath
                    cap = cv2.VideoCapture(video_path)

                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

                    model = YOLO('best.pt')

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        results = model(frame, save=True)
                        print(results)

                        res_plotted = results[0].plot()
                        cv2.imshow("result", res_plotted)

                        out.write(res_plotted)

                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()

                    return video_feed()

    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    lastest_subfolder = max(subfolders, default=None, key=lambda x: os.path.getatime(os.path.join(folder_path, x)))

    if imgpath and lastest_subfolder:
        image_path = os.path.join(folder_path, lastest_subfolder, imgpath)
        return render_template('index.html', image_path=image_path)
    else:
        return render_template('index.html')

@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    # Tambahkan penanganan untuk situasi di mana tidak ada subfolder
    if not subfolders:
        return "No subfolders found"

    lastest_subfolder = max(subfolders, default=None, key=lambda x: os.path.getatime(os.path.join(folder_path, x)))
    directory = os.path.join(folder_path, lastest_subfolder)

    # Pastikan folder ada
    if not os.path.exists(directory):
        return f"Directory '{directory}' not found"

    files = os.listdir(directory)

    # Tambahkan penanganan untuk situasi di mana tidak ada file
    if not files:
        return "No files found in the directory"

    # Ambil file terbaru
    latest_file = max(files, default=None, key=lambda x: os.path.getatime(os.path.join(directory, x)))

    if latest_file:
        file_path = os.path.join(directory, latest_file)
        file_extension = latest_file.rsplit('.', 1)[1].lower()

        # Tambahkan penanganan untuk format file yang tidak valid
        if file_extension == "jpg":
            return send_from_directory(directory, latest_file)
        else:
            return "Invalid file format"
    else:
        return "File not found"

def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)

@app.route("/video_feed")
def video_feed():
    print("function called")
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
