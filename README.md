
# 🧠 NeuroScan AI - Deteksi Dini Tumor Otak

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey?style=flat-square&logo=flask)
![Machine Learning](https://img.shields.io/badge/ML-KNN%20%2B%20GLCM-success?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

NeuroScan AI adalah sistem cerdas berbasis web untuk mendeteksi indikasi keberadaan tumor otak melalui citra *Magnetic Resonance Imaging* (MRI). Sistem ini mengidentifikasi pola kelainan pada otak bukan dari bentuknya, melainkan dengan menganalisis teksturnya menggunakan kombinasi ekstraksi fitur **Gray Level Co-occurrence Matrix (GLCM)** dan algoritma klasifikasi **K-Nearest Neighbors (KNN)**.

Proyek ini dikembangkan untuk memenuhi Ujian Akhir Praktikum (UAP) Program Studi Ilmu Komputer, Universitas Pakuan.

## 👥 Tim Riset & Pengembang (Kelas GAB A)
* **Hafnasatul Annisa** (065123025) - *Data Processing*
* **Revieta Ramadhani** (065123026) - *UI/UX & Web Development*
* **Muhamad Tholib** (065123058) - *Model Training & Logic*

## ✨ Fitur Utama
* **Unggah Citra Medis Interaktif:** Mendukung format JPG/PNG dengan fitur *drag & drop*.
* **Ekstraksi Tekstur Real-time:** Menghitung 4 parameter matriks GLCM secara instan (*Contrast, Homogeneity, Energy, Correlation*).
* **Klasifikasi Cerdas:** Menggunakan model KNN teroptimasi ($K=3$) yang telah distandarisasi menggunakan *Z-Score Scaler*.
* **Penyimpanan Riwayat:** Sistem otomatis mencatat log hasil analisis ke dalam format JSON.
* **UI/UX Modern:** Dilengkapi antarmuka yang bersih, responsif, dan didukung fitur *Dark Mode*.

## 🛠️ Teknologi yang Digunakan
* **Backend:** Python, Flask
* **Machine Learning:** Scikit-Learn, Scikit-Image
* **Image Processing:** OpenCV, NumPy
* **Frontend:** HTML5, CSS3, Bootstrap 5, Vanilla JavaScript

## 📁 Struktur Direktori
```text
📦 UAP_PC_BRAINTUMOR
 ┣ 📂 dataset/           # Data latih gambar MRI (Folder kelas 'no' dan 'yes')
 ┣ 📂 static/
 ┃ ┗ 📂 uploads/         # Direktori penyimpanan citra sementara
 ┣ 📂 templates/
 ┃ ┣ 📜 dashboard.html   # Halaman utama & workspace analisis
 ┃ ┣ 📜 metodologi.html  # Halaman landasan teori & rumus
 ┃ ┣ 📜 riwayat.html     # Halaman tabel log analisis
 ┃ ┗ 📜 tim-riset.html   # Halaman profil kelompok
 ┣ 📜 app.py             # Script utama Flask server
 ┣ 📜 train_model.py     # Script ekstraksi fitur & pelatihan model ML
 ┣ 📜 riwayat.json       # Database log riwayat analisis
 ┣ 📜 requirements.txt   # Daftar dependensi library
 ┗ 📜 .gitignore         # File pengecualian upload repositori

```

## 🚀 Cara Instalasi & Menjalankan Aplikasi (Localhost)

**1. Clone Repository**

```bash
git clone [https://github.com/USERNAME_GITHUB_KAMU/neuroscan-ai.git](https://github.com/USERNAME_GITHUB_KAMU/neuroscan-ai.git)
cd neuroscan-ai

```

**2. Buat & Aktifkan Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate  # Untuk pengguna Windows

```

**3. Install Dependencies**

```bash
pip install -r requirements.txt

```

**4. Jalankan Server Web**

```bash
python app.py

```

Aplikasi dapat diakses secara lokal melalui browser pada alamat: `http://127.0.0.1:5000`

---

*Dibuat dengan ❤️ oleh Mahasiswa Ilmu Komputer Universitas Pakuan.*


```