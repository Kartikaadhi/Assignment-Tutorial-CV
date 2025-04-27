import os 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline 
from sklearn.svm import SVC 
from sklearn.decomposition import PCA 
from sklearn.metrics import classification_report 
import pickle

# Fungsi untuk memuat gambar
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print('Error: Could not load image.')
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

# Fungsi untuk mengubah ukuran dan meratakan gambar
def resize_and_flatten(face, size=(100, 100)):
    face_resized = cv2.resize(face, size)
    return face_resized.flatten()

# Memuat dataset
print("Memuat dataset...")
dataset_dir = 'C:\\Users\\Asus\\Documents\\Tutorial CV\\dataset (1)'  # Ganti dengan path ke folder dataset Anda
images = []
labels = []
for root, dirs, files in os.walk(dataset_dir):
    if len(files) == 0:
        continue
    for f in files:
        print(f"Mengolah file: {f}")
        _, image = load_image(os.path.join(root, f))
        if image is None:
            continue
        images.append(image)
        labels.append(root.split('/')[-1])  # Mengambil nama folder sebagai label
print("Dataset dimuat.")

# Deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 

def detect_faces(image_gray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)): 
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size) 
    return faces

# Memotong wajah
def crop_faces(image_gray, faces, return_all=False): 
    cropped_faces = [] 
    selected_faces = [] 
    if len(faces) > 0: 
        if return_all: 
            for x, y, w, h in faces: 
                selected_faces.append((x, y, w, h)) 
                cropped_faces.append(image_gray[y:y+h, x:x+w]) 
        else: 
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3]) 
            selected_faces.append((x, y, w, h)) 
            cropped_faces.append(image_gray[y:y+h, x:x+w]) 
    return cropped_faces, selected_faces

face_size = (128, 128) 
 
def resize_and_flatten(face): 
    face_resized = cv2.resize(face, face_size) 
    face_flattened = face_resized.flatten() 
    return face_flattened

# Menyiapkan data pelatihan dan pengujian
X = [] 
y = [] 

for image, label in zip(images, labels): 
    faces = detect_faces(image) 
    cropped_faces, _ = crop_faces(image, faces) 
    if len(cropped_faces) > 0: 
        face_flattened = resize_and_flatten(cropped_faces[0]) 
        X.append(face_flattened) 
        y.append(label) 

X = np.array(X) 
y = np.array(y)

# Membagi dataset menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=177, stratify=y)

# Mean Centering
class MeanCentering(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None): 
        self.mean_face = np.mean(X, axis=0) 
        return self 

    def transform(self, X): 
        return X - self.mean_face
    
# Membangun pipeline untuk ekstraksi Eigenfaces dan klasifikasi
pipe = Pipeline([
     ('centering', MeanCentering()), 
     ('pca', PCA(svd_solver='randomized', whiten=True, random_state=177)), 
     ('svc', SVC(kernel='linear', random_state=177)) 
])
    
# Melatih model
print("Melatih model...")
pipe.fit(X_train, y_train) 
print("Model dilatih.")

# Menguji model 
print("Menguji model...")
y_pred = pipe.predict(X_test) 
print(classification_report(y_test, y_pred))

# Visualisasi Eigenfaces
n_components = len(pipe[1].components_)  # Jumlah komponen yang dihasilkan oleh PCA
ncol = 4 
nrow = (n_components + ncol - 1) // ncol 
fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.5*nrow), subplot_kw={'xticks':[], 'yticks':[]}) 

eigenfaces = pipe[1].components_.reshape((n_components, X_train.shape[1])) 
face_size = (128, 128)  # Ukuran yang sama dengan yang digunakan saat meratakan gambar

for i, ax in enumerate(axes.flat): 
    if i < n_components:  # Pastikan tidak melebihi jumlah komponen
        ax.imshow(eigenfaces[i].reshape(face_size), cmap='gray')  # Ganti dengan ukuran yang sesuai
        ax.set_title(f'Eigenface {i+1}') 
    else:
        ax.axis('off')  # Matikan sumbu jika tidak ada eigenface yang ditampilkan

plt.tight_layout()
plt.show()

# Menyimpan model pipeline
with open('eigenface_pipeline.pkl', 'wb') as f: 
    pickle.dump(pipe, f)

# Fungsi untuk mendapatkan skor eigenface
def get_eigenface_score(X): 
    X_pca = pipe[:2].transform(X) 
    eigenface_scores = np.max(pipe[2].decision_function(X_pca), axis=1) 
    return eigenface_scores 

# Fungsi untuk prediksi eigenface
def eigenface_prediction(image_gray): 
    faces = detect_faces(image_gray) 
    cropped_faces, selected_faces = crop_faces(image_gray, faces) 
     
    if len(cropped_faces) == 0: 
        return 'No face detected.' 
     
    X_face = [] 
    for face in cropped_faces: 
        face_flattened = resize_and_flatten(face) 
        X_face.append(face_flattened) 
    X_face = np.array(X_face) 
    labels = pipe.predict(X_face) 
    scores = get_eigenface_score(X_face) 
     
    return scores, labels, selected_faces   

# Fungsi untuk menggambar teks pada gambar
def draw_text(image, label, score, 
              font=cv2.FONT_HERSHEY_SIMPLEX, 
              pos=(0, 0), 
              font_scale=0.6, 
              font_thickness=2, 
              text_color=(0, 0, 0), 
              text_color_bg=(0, 255, 0)): 

    x, y = pos 
    score_text = f'Score: {score:.2f}' 
    (w1, h1), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness) 
    (w2, h2), _ = cv2.getTextSize(label, font, font_scale, font_thickness) 
    cv2.rectangle(image, (x, y-h1-h2-25), (x + max(w1, w2)+20, y), text_color_bg, -1) 
    cv2.putText(image, label, (x+10, y-10), font, font_scale, text_color, font_thickness) 
    cv2.putText(image, score_text, (x+10, y-h2-15), font, font_scale, text_color, font_thickness) 

# Fungsi untuk menggambar hasil deteksi
def draw_result(image, scores, labels, coords): 
    result_image = image.copy() 
    for (x, y, w, h), label, score in zip(coords, labels, scores): 
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        draw_text(result_image, label, score, pos=(x, y)) 
    return result_image

def real_time_recognition():
    cap = cv2.VideoCapture(0)  # Menggunakan webcam

    if not cap.isOpened():
        print("Error: Kamera tidak dapat diakses.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Atur lebar frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Atur tinggi frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame dari kamera.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray_frame)

        if len(faces) > 0:
            cropped_faces, coords = crop_faces(gray_frame, faces)
            scores = []
            labels = []
            for face in cropped_faces:
                face_flattened = resize_and_flatten(face)
                label = pipe.predict([face_flattened])[0]  # Ambil label pertama
                score = get_eigenface_score([face_flattened])[0]  # Ambil skor pertama
                scores.append(score)
                labels.append(label)

            # Gambar hasil deteksi
            frame = draw_result(frame, scores, labels, coords)

        cv2.imshow('Real-Time Face Recognition', frame)

        # Tampilkan gambar grayscale untuk debugging
        cv2.imshow('Gray Frame', gray_frame)

        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Keluar dari loop jika 'q' ditekan

    # Melepaskan kamera dan menutup semua jendela
    cap.release()
    cv2.destroyAllWindows()

# Menjalankan pengenalan wajah secara real-time
if __name__ == "__main__":
    real_time_recognition()