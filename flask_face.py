import cv2
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)

    img = cv2.imread(filename)
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    result_filename = 'result.png'
    result_path = os.path.join('static', result_filename)
    cv2.imwrite(result_path, img)

    return redirect(url_for('result', filename=result_filename))

@app.route('/result')
def result():
    filename = request.args.get('filename')
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
