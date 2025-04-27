# Assignment-Tutorial-CV
Nama : Kartika Adhi Ning Wulan Satunggal
NIM  : 23/519940/PA/22325

# Pengenalan Wajah

## Deskripsi
Proyek ini adalah sistem pengenalan wajah yang menggunakan OpenCV dan scikit-learn untuk mendeteksi dan mengenali wajah dalam gambar dan video secara real-time. Sistem ini dilatih menggunakan dataset yang terdiri dari gambar wajah dari beberapa individu, termasuk gambar wajah pengguna sendiri. Dengan menggunakan teknik pengenalan wajah berbasis Eigenfaces, proyek ini bertujuan untuk memberikan akurasi yang baik dalam mengenali wajah meskipun dalam kondisi pencahayaan dan ekspresi yang bervariasi.

## Fitur
- Deteksi wajah secara real-time menggunakan webcam.
- Pengenalan wajah dengan menampilkan nama individu yang dikenali.
- Visualisasi hasil deteksi dengan kotak pembatas dan label.
- Kemampuan untuk menangani variasi dalam pencahayaan dan ekspresi wajah.

## Persyaratan
Sebelum menjalankan proyek ini, pastikan Anda memenuhi persyaratan berikut:
- Pastikan Python terinstal di sistem. 
- Instal pustaka berikut dengan menggunakan pip:
  ```bash
  pip install opencv-python numpy scikit-learn matplotlib

# Dataset

Dataset yang digunakan dalam proyek ini terdiri dari gambar wajah dari beberapa individu. Setiap individu memiliki folder terpisah yang berisi gambar-gambar wajah mereka. Pastikan setiap individu memiliki setidaknya 10 gambar dengan variasi pencahayaan dan ekspresi untuk meningkatkan akurasi pengenalan.

# Cara Menjalankan

1. Siapkan Dataset:
Kumpulkan gambar wajah dari beberapa individu dan simpan dalam folder terpisah sesuai dengan struktur yang dijelaskan di atas.
2. Jalankan Kode:
Setelah menyiapkan dataset, buka terminal atau command prompt dan navigasikan ke direktori proyek.

Jalankan skrip dengan perintah berikut:
python face_recognition.py

3. Penggunaan:
Setelah menjalankan skrip, jendela akan terbuka yang menampilkan video dari webcam. Sistem akan mendeteksi wajah dan menampilkan nama individu yang dikenali beserta skor kepercayaan.
Tekan 'q' untuk keluar dari aplikasi saat jendela pengenalan wajah terbuka.

# Visualisasi

Proyek ini juga menyertakan visualisasi Eigenfaces yang dihasilkan selama pelatihan model. Eigenfaces adalah representasi wajah yang digunakan untuk pengenalan wajah. Visualisasi ini akan ditampilkan setelah model dilatih.

#Catatan

- Pastikan webcam berfungsi dengan baik sebelum menjalankan aplikasi.
- Jika mengalami masalah dengan deteksi wajah, pastikan pencahayaan di sekitar cukup baik.
- Dapat menyesuaikan parameter dalam kode untuk meningkatkan akurasi deteksi dan pengenalan wajah.
