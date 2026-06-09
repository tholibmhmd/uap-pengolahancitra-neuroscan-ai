import os
import cv2
import numpy as np
import pickle
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # Senjata rahasia untuk KNN

def extract_glcm_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    
    # Kembali ke parameter dasar yang lebih fokus
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return [contrast, homogeneity, energy, correlation]

def load_dataset(base_path):
    X = [] 
    y = [] 
    classes = {'no': 0, 'yes': 1}
    
    print("Mengekstrak fitur...")
    for class_name, label in classes.items():
        folder_path = os.path.join(base_path, class_name)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            try:
                features = extract_glcm_features(img_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                pass
    return np.array(X), np.array(y)

if __name__ == '__main__':
    dataset_path = 'dataset'
    X, y = load_dataset(dataset_path)
    print(f"Total data: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- PROSES SCALING DATA ---
    print("Menyamakan skala fitur...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Kita coba K=3 kali ini agar lebih sensitif
    print("Melatih KNN (K=3)...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi SETELAH Scaling: {accuracy * 100:.2f}%")
    
    # Simpan Model dan Scaler
    with open('model_knn_glcm.pkl', 'wb') as file:
        pickle.dump(knn, file)
    with open('scaler_glcm.pkl', 'wb') as file:
        pickle.dump(scaler, file)
        
    print("Model dan Scaler berhasil disimpan.")