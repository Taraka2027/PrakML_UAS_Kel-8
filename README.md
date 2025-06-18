# Proyek Sistem Rekomendasi Buku - UAS Praktikum Pembelajaran Mesin
Ini adalah repository untuk proyek Ujian Akhir Semester (UAS) Praktikum Pembelajaran Mesin dengan topik Sistem Rekomendasi Buku.  Proyek ini dibangun untuk merekomendasikan buku kepada pengguna berdasarkan kemiripan pola rating dari pengguna lain.

## Anggota Kelompok 8
* 187231008 | Sellen Seselia
* 187231042 | Muhammad Herjuna Taraka
* 187231056 | Bunga Calista Nabila Syaman
* 187231104 | Raesutha Arya Cakrashena
* 187231107 | Thalita Putri Kaylaluna	
Struktur Proyek
Repository ini berisi:

PrakML_UAS.py: Script utama Python yang berisi semua logika, mulai dari preprocessing hingga model rekomendasi interaktif.

UAS_Kelompok8.pdf: Dokumen soal UAS.

dataset: Books.csv, Users.csv, Book-Ratings.csv.

Teknologi yang Digunakan
* Python 3
* Pandas: Untuk manipulasi dan analisis data.
* NumPy: Untuk operasi numerik.
* Scikit-learn: Untuk menghitung similaritas (Euclidean, Cosine) dan normalisasi (MinMaxScaler).
* Matplotlib & Seaborn: Untuk visualisasi data dalam tahap eksplorasi.

## Tahapan Pengerjaan Proyek
Proyek ini dikerjakan sesuai dengan alur kerja standar machine learning dan arahan tugas. 

### 1. Input Dataset
Memuat tiga dataset yang disediakan: Books.csv, Users.csv, dan Book-Ratings.csv. 

### 2. Preprocessing Data
Membersihkan dan mempersiapkan data agar siap untuk pemodelan. 
* Pengecekan Missing Value & Outlier: Mengidentifikasi data yang hilang dan data yang tidak wajar (misalnya, umur 244 tahun atau tahun terbit 0).
* Penanganan Missing Value & Outlier: Mengisi data numerik yang hilang dengan median dan menghapus baris dengan data kategorikal yang hilang. Outlier diubah menjadi nilai NaN lalu diimputasi.
* Encoding: Melakukan feature engineering pada kolom Location untuk membuat kolom Country, kemudian menerapkan One-Hot Encoding untuk mengubah data kategorikal negara menjadi format numerik.
* Normalisasi: Menerapkan Min-Max Scaling pada kolom Age untuk mengubah skalanya menjadi rentang 0-1. 

### 3. Analisis Similaritas
Membangun "mesin" rekomendasi dengan menghitung kemiripan antar buku menggunakan tiga metode berbeda. 
* Euclidean Distance: Mengukur jarak geometris lurus antar buku.
* Cosine Similarity: Mengukur kemiripan pola rating berdasarkan sudut antar vektor.
* Pearson Correlation Coefficient: Varian dari Cosine yang menyesuaikan dengan bias rating dari setiap pengguna.

### 4. Implementasi Model Collaborative Filtering
Membungkus semua logika (pembuatan matriks interaksi dan perhitungan similaritas) ke dalam sebuah class Python ItemItemRecommender. Pendekatan yang digunakan adalah Item-Item Collaborative Filtering.  Model ini memiliki metode .fit() untuk persiapan data dan .predict() untuk menghasilkan rekomendasi.

### 5. Evaluasi Model
Mengukur performa model dengan metrik hit rate sederhana pada 20 sampel pengguna. Proses ini mensimulasikan skenario di mana model merekomendasikan 10 buku, lalu dihitung berapa banyak dari rekomendasi tersebut yang relevan (pernah dibaca) oleh pengguna.

## Cara Menjalankan Program
* Pastikan Anda memiliki Python 3 ter-install.
* Clone repository ini atau unduh file .py dan dataset .csv
* Install semua library yang dibutuhkan melalui terminal/command prompt: Bash
"pip install pandas numpy scikit-learn matplotlib seaborn"
* Tempatkan file Books.csv, Users.csv, dan Book-Ratings.csv dalam satu folder dengan script Python, atau pastikan script dapat mengunduhnya secara otomatis.
* Jalankan script dari terminal: Bash
"python PrakML_UAS.py"
* Program akan berjalan secara interaktif dan meminta Anda memasukkan judul buku untuk mendapatkan rekomendasi.
