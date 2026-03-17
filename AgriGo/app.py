from flask import Flask, render_template, request, send_from_directory, redirect, url_for, session, flash
import random
import os
import sqlite3
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from functions import (
    img_predict,
    get_diseases_classes,
    get_crop_recommendation,
    get_fertilizer_recommendation,
    soil_types,
    Crop_types,
    crop_list
)

app = Flask(__name__)
random.seed(0)

app.config['SECRET_KEY'] = 'agrigo_secret_key_123'

# ==========================
# 🗄️ Database Setup
# ==========================
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==========================
# Upload Folder
# ==========================
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ==========================
# Register
# ==========================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':

        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, password)
            )
            conn.commit()
            conn.close()

            flash("Registration Successful! Please login.")
            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            flash("Username or Email already exists.")
            return redirect(url_for('register'))

    return render_template('register.html')


# ==========================
# Login
# ==========================
@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):

            session['user_id'] = user[0]
            session['username'] = user[1]

            flash("Login Successful!")
            return redirect(url_for('index'))

        else:
            flash("Invalid Email or Password")
            return redirect(url_for('login'))

    return render_template('login.html')


# ==========================
# Logout
# ==========================
@app.route('/logout')
def logout():

    session.clear()
    flash("Logged out successfully.")

    return redirect(url_for('login'))


# ==========================
# Home Page
# ==========================
@app.route('/', methods=['GET', 'POST'])
def index():

    if 'user_id' not in session:
        return redirect(url_for('login'))

    return render_template('index.html')


# ==========================
# 🌱 Crop Recommendation
# ==========================
@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():

    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == "POST":

        season = request.form['season']
        soil_type = request.form['soil_type']

        result = get_crop_recommendation(season, soil_type)

        return render_template("recommend_result.html", result=result)

    else:
        return render_template('crop-recommend.html')


# ==========================
# 🌾 Fertilizer Recommendation
# ==========================
@app.route('/fertilizer-recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():

    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == "POST":

        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']

        result = get_fertilizer_recommendation(soil_type, crop_type)

        return render_template("recommend_result.html", result=result)

    else:
        return render_template(
            'fertilizer-recommend.html'
        )


# ==========================
# 🍅 Crop Disease Detection
# ==========================
@app.route('/crop-disease', methods=['POST', 'GET'])
def find_crop_disease():

    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == "GET":

        return render_template(
            'crop-disease.html',
            crops=crop_list
        )

    else:

        if "file" not in request.files:
            return "No file uploaded"

        file = request.files["file"]

        if file.filename == "":
            return "No selected file"

        crop = request.form["crop"]

        filename = secure_filename(file.filename)

        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        file.save(file_path)

        prediction = img_predict(file_path, crop)

        result = get_diseases_classes(
            crop,
            prediction
        )

        return render_template(
            'disease-prediction-result.html',
            image_file_name=filename,
            result=result
        )


# ==========================
# Serve Uploaded Images
# ==========================
@app.route('/uploads/<filename>')
def send_file(filename):

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )


# ==========================
# Run App
# ==========================
if __name__ == '__main__':

    app.run(debug=True)