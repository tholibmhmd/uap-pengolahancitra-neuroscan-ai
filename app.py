from flask import Flask, render_template, request, url_for
import os
import cv2
import pickle
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import json
from datetime import datetime

app = Flask(__name__)

# --- KONFIGURASI FOLDER ---
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- LOAD MODEL & SCALER ---
try:
    with open('model_knn_glcm.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler_glcm.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    print("WARNING: File model_knn_glcm.pkl atau scaler_glcm.pkl tidak ditemukan!")
    print("Pastikan kamu sudah menjalankan train_model.py terlebih dahulu.")

# --- FUNGSI EKSTRAKSI GLCM ---
def extract_features_for_prediction(image_path):
    # Baca gambar dan ubah ke grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize harus persis sama dengan saat training (128x128)
    img = cv2.resize(img, (128, 128))
    
    # Hitung matriks GLCM
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Ambil nilai fitur
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return [contrast, homogeneity, energy, correlation]


# ==========================================
# ROUTING HALAMAN WEB
# ==========================================

# 1. Halaman Utama (Dashboard)
@app.route('/', methods=['GET'])
def index():
    return render_template('dashboard.html', result_active=False)

# 2. Proses Prediksi (Dipanggil saat tombol form ditekan)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Tidak ada file yang diunggah"
    
    file = request.files['file']
    if file.filename == '':
        return "Nama file kosong"
        
    if file:
        # Simpan file yang diunggah ke folder static/uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Ekstrak 4 nilai GLCM mentah dari gambar
        raw_features = extract_features_for_prediction(filepath)
        
        # --- SCALE FITUR SEBELUM DIPREDIKSI ---
        features_scaled = scaler.transform([raw_features])
        
        # Prediksi menggunakan fitur yang sudah di-scale
        prediction = model.predict(features_scaled)[0]
        
        # Terjemahkan hasil (1 = Tumor, 0 = Normal)
        hasil = "Terdeteksi Tumor" if prediction == 1 else "Normal (Sehat)"
        
        # --- LOGIKA SIMPAN RIWAYAT KE JSON ---
        record = {
            "waktu": datetime.now().strftime("%d-%m-%Y %H:%M"),
            "file": file.filename,
            "hasil": hasil,
            "c": round(raw_features[0], 4),
            "h": round(raw_features[1], 4),
            "e": round(raw_features[2], 4),
            "cor": round(raw_features[3], 4)
        }
        
        history_file = 'riwayat.json'
        riwayat_data = []
        
        # Baca data lama jika file sudah ada
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    riwayat_data = json.load(f)
            except json.JSONDecodeError:
                # Jika file json kosong atau error, mulai dengan list kosong
                riwayat_data = []
                
        # Masukkan data baru ke urutan paling atas (index 0)
        riwayat_data.insert(0, record)
        
        # Simpan kembali ke file
        with open(history_file, 'w') as f:
            json.dump(riwayat_data, f, indent=4)
        # -------------------------------------
        
        # Kirim data kembali ke dashboard.html untuk ditampilkan
        return render_template('dashboard.html', 
                               result_active=True,
                               image_name=file.filename,
                               hasil_prediksi=hasil,
                               contrast=record['c'],
                               homogeneity=record['h'],
                               energy=record['e'],
                               correlation=record['cor'])

# 3. Halaman Riwayat
@app.route('/riwayat')
def riwayat():
    history_file = 'riwayat.json'
    riwayat_data = []
    
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                riwayat_data = json.load(f)
        except json.JSONDecodeError:
            pass
            
    return render_template('riwayat.html', data_riwayat=riwayat_data)

# 4. Halaman Metodologi
@app.route('/metodologi')
def metodologi():
    return render_template('metodologi.html')

# 5. Halaman Tim Riset
@app.route('/tim-riset')
def tim_riset():
    return render_template('tim-riset.html')


if __name__ == '__main__':
    app.run(debug=True)