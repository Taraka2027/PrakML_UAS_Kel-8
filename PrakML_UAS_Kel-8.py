# ==============================================================================
# UAS PRAKTIKUM PEMBELAJARAN MESIN - SISTEM REKOMENDASI BUKU
# ==============================================================================

# ==============================================================================
# 1. SETUP AWAL: IMPOR LIBRARY
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')

print("Library yang dibutuhkan telah diimpor.")

# ==============================================================================
# 2. INPUT DATASET: MEMUAT FILE CSV
# ==============================================================================
print("\nMemuat dataset...")
try:
    # Catatan: Dataset ini memiliki beberapa keunikan saat dibaca
    # sep=';'          -> pemisah antar kolom adalah titik koma, bukan koma
    # encoding='latin-1' -> menggunakan encoding latin-1 untuk membaca karakter khusus
    # on_bad_lines='skip' -> melewati baris yang memiliki format error

    books = pd.read_csv('Books.csv', sep=',', on_bad_lines='skip', encoding='latin-1')
    users = pd.read_csv('Users.csv', sep=',', on_bad_lines='skip', encoding='latin-1')
    ratings = pd.read_csv('Books-Ratings.csv', sep=',', on_bad_lines='skip', encoding='latin-1')

    print("Dataset berhasil dimuat.")

    # Menampilkan 5 baris pertama dari setiap dataframe untuk verifikasi
    print("\nContoh Data Books:")
    print(books.head())

    print("\nContoh Data Users:")
    print(users.head())

    print("\nContoh Data Ratings:")
    print(ratings.head())

except FileNotFoundError as e:
    print(f"\nError: {e}")
    print("Pastikan file CSV berada di folder yang sama dengan script Python Anda.")

# ==============================================================================
# 3. PREPROCESSING DATA
# ==============================================================================

# ==============================================================================
# A. Cek Missing Value dan Outlier
# ==============================================================================

# ------------------------------------------------------------------------------
# a. Cek Missing Value
# ------------------------------------------------------------------------------
print("\n--- Mengecek Missing Value ---")

# Menghitung jumlah missing value di setiap kolom untuk setiap dataframe
print("\nMissing values di dataframe Books:")
print(books.isnull().sum())

print("\nMissing values di dataframe Users:")
print(users.isnull().sum())

print("\nMissing values di dataframe Ratings:")
print(ratings.isnull().sum())

# ------------------------------------------------------------------------------
# 3.b. Diagnosa Tipe Data : Cek Tipe Data di Semua Kolom Numerik
# ------------------------------------------------------------------------------

print("\n--- Memulai Diagnosis Menyeluruh untuk Kolom Numerik ---")

# 1. Definisikan dataframe dan kolom-kolom yang kita harapkan numerik
datasets = {
    "Books": {
        "df": books,
        "numeric_cols": ['Year-Of-Publication']
    },
    "Users": {
        "df": users,
        "numeric_cols": ['User-ID', 'Age']
    },
    "Ratings": {
        "df": ratings,
        "numeric_cols": ['User-ID', 'Book-Rating']
    }
}

# 2. Lakukan looping untuk setiap dataset dan kolom
for name, data in datasets.items():
    print(f"\n--- Menganalisis Dataframe: {name} ---")
    df = data['df']
    numeric_cols = data['numeric_cols']

    for col in numeric_cols:
        # Cek tipe data saat ini
        current_dtype = df[col].dtype
        print(f"Kolom '{col}' | Tipe data saat ini: {current_dtype}")

        # Jika tipe datanya bukan angka (object), maka perlu investigasi mendalam
        if current_dtype == 'object':
            # Gunakan teknik yang sama untuk menemukan baris yang bermasalah
            is_numeric = pd.to_numeric(df[col], errors='coerce').notnull()
            problematic_rows = df[~is_numeric]

            if not problematic_rows.empty:
                print(f"  [!] Ditemukan {len(problematic_rows)} baris dengan nilai non-numerik di kolom '{col}'.")
                # Tampilkan beberapa contoh baris yang bermasalah
                print("  Contoh baris bermasalah:")
                print(problematic_rows.head())
            else:
                print(f"  [OK] Kolom '{col}' bertipe 'object' tapi semua nilainya bisa dikonversi ke numerik.")
        else:
            # Jika sudah numerik, anggap aman
            print(f"  [OK] Kolom '{col}' sudah memiliki tipe data numerik yang valid.")

print("\n--- Diagnosis Menyeluruh Selesai ---")

print("\n\n--- Membersihkan dan Memperbaiki Tipe Data ---")

# --- 1. Membersihkan Kolom 'Year-Of-Publication' ---
print(f"Tipe data 'Year-Of-Publication' sebelum diubah: {books['Year-Of-Publication'].dtype}")
books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
print(f"Tipe data 'Year-Of-Publication' setelah diubah: {books['Year-Of-Publication'].dtype}")

print("\nPembersihan data selesai. Nilai-nilai yang salah telah diubah menjadi NaN.")

# ------------------------------------------------------------------------------
# 3.c. Cek Outlier
# ------------------------------------------------------------------------------
print("\n\n--- Mengecek Outlier ---")

# --- 1. Menggunakan Metode IQR ---
print("\n--- Menghitung Jumlah Outlier Secara Langsung dengan Metode IQR ---")

def hitung_outlier(dataframe, column_name):
    """
    Fungsi untuk menghitung jumlah outlier dalam sebuah kolom menggunakan metode IQR.
    """
    # Pastikan kolom ada di dataframe
    if column_name not in dataframe.columns:
        return f"Error: Kolom '{column_name}' tidak ditemukan."

    # Ambil data kolom dan hapus nilai NaN untuk perhitungan
    data = dataframe[column_name].dropna()

    # Hitung Q1, Q3, dan IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Tentukan batas bawah dan batas atas
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR

    # Hitung jumlah nilai yang berada di luar batas
    jumlah_outlier = ((data < batas_bawah) | (data > batas_atas)).sum()

    return jumlah_outlier

# --- Terapkan fungsi pada kolom yang relevan ---

# Menghitung outlier untuk 'Year-Of-Publication'
jumlah_outlier_tahun = hitung_outlier(books, 'Year-Of-Publication')
print(f"Jumlah outlier di kolom 'Year-Of-Publication': {jumlah_outlier_tahun}")

# Menghitung outlier untuk 'Age'
jumlah_outlier_umur = hitung_outlier(users, 'Age')
print(f"Jumlah outlier di kolom 'Age': {jumlah_outlier_umur}")

# --- 2. Menggunakan Visualisasi untuk melihat sebaran data ---
print("\nMembuat visualisasi untuk deteksi outlier...")

# Box plot untuk 'Age'
plt.figure(figsize=(8, 6))
sns.boxplot(x=users['Age'])
plt.title('Box Plot untuk Kolom Age')
plt.xlabel('Age')
plt.grid(True)
plt.savefig('age_boxplot.png') # Menyimpan plot sebagai file gambar
print("Box plot untuk Age disimpan sebagai 'age_boxplot.png'")

# Box plot untuk 'Year-Of-Publication'
plt.figure(figsize=(8, 6))
sns.boxplot(x=books['Year-Of-Publication'])
plt.title('Box Plot untuk Kolom Year-Of-Publication')
plt.xlabel('Year-Of-Publication')
plt.grid(True)
plt.savefig('year_boxplot.png') # Menyimpan plot sebagai file gambar
print("Box plot untuk Year-Of-Publication disimpan sebagai 'year_boxplot.png'")

# Count plot untuk 'Book-Rating' untuk melihat distribusi rating
plt.figure(figsize=(10, 6))
sns.countplot(x='Book-Rating', data=ratings)
plt.title('Distribusi Book-Rating')
plt.xlabel('Rating')
plt.ylabel('Jumlah')
plt.grid(True)
plt.savefig('ratings_distribution.png') # Menyimpan plot sebagai file gambar
print("Distribusi rating disimpan sebagai 'ratings_distribution.png'")

# ------------------------------------------------------------------------------
# 3.d. Menangani Missing Value (Imputation & Dropping)
# ------------------------------------------------------------------------------

print("--- Menangani Semua Sisa Missing Value ---")

# --- 1. Mengisi NaN pada Kolom 'Age' dengan Median ---
print(f"Jumlah missing value di 'Age' sebelum diisi: {users['Age'].isnull().sum()}")
median_age = users['Age'].median()
users['Age'] = users['Age'].fillna(median_age)
print(f"Missing values di 'Age' telah diisi dengan median: {median_age}")


# --- 2. Menghapus Baris dengan NaN pada Kolom Lainnya di 'Books' ---
print(f"\nJumlah baris di 'Books' sebelum dihapus: {len(books)}")
# Menghapus baris jika salah satu dari kolom ini memiliki nilai NaN
books.dropna(subset=['Book-Author', 'Publisher', 'Image-URL-L'], inplace=True)
print(f"Jumlah baris di 'Books' setelah baris dengan NaN dihapus: {len(books)}")


# --- Verifikasi Akhir ---
print("\n--- Verifikasi Ulang Setelah Semua Missing Value Ditangani ---")
print("\nMissing values di dataframe Books:")
print(books.isnull().sum())
print("\nMissing values di dataframe Users:")
print(users.isnull().sum())

# ==============================================================================
# B. Encoding
# ==============================================================================

print("\n--- Memulai Proses Feature Engineering & Encoding ---")

# --- 1. Feature Engineering: Membuat kolom 'Country' dari 'Location' ---
users['Country'] = users['Location'].str.split(',').str[-1].str.strip()

users.loc[users['Country'].isin(['', 'o', 'z', 'ß', 'far away...', 'the world', 'under the sea running away']), 'Country'] = 'other'

# --- Perbaikan di baris ini ---
users['Country'] = users['Country'].fillna('other')

print("Kolom 'Country' berhasil dibuat dari 'Location'.")


# --- 2. One-Hot Encoding pada Kolom 'Country' ---
top_10_countries = users['Country'].value_counts().head(10).index
users['Country'] = users['Country'].where(users['Country'].isin(top_10_countries), 'other')

country_dummies = pd.get_dummies(users['Country'], prefix='country', dtype=int)
users = pd.concat([users, country_dummies], axis=1)
users.drop(['Location', 'Country'], axis=1, inplace=True)

print("\nProses Encoding selesai.")
print("Contoh dataframe 'users' setelah encoding:")
print(users.head())

# ==============================================================================
# C. Normalisasi
# ==============================================================================

print("\n--- Memulai Proses Normalisasi ---")

# 1. Inisialisasi Scaler
scaler = MinMaxScaler()

# 2. Melakukan normalisasi pada kolom 'Age'
#    fit_transform memerlukan input 2D, jadi kita gunakan [[]] untuk memilih kolom
users['Age_normalized'] = scaler.fit_transform(users[['Age']])

print("Kolom 'Age' telah dinormalisasi dan disimpan di 'Age_normalized'.")

# 3. Tampilkan hasil untuk verifikasi
print("\nContoh dataframe 'users' setelah normalisasi:")
print(users.head())

print("\nStatistik deskriptif untuk kolom 'Age_normalized':")
print(users['Age_normalized'].describe())

# ==============================================================================
# 4. SIMILARITY: Membuat Matriks Interaksi User-Item
# ==============================================================================

print("\n--- Membuat Matriks Interaksi User-Item ---")

# --- 1. Menggabungkan Dataframes ---
# Kita gabungkan ratings dengan books untuk mendapatkan judul buku
# Ini juga akan secara otomatis menyaring rating untuk buku-buku yang ada di dataframe 'books' kita
df = pd.merge(ratings, books, on='ISBN')


# --- 2. Menghitung Jumlah Rating ---
# Menghitung berapa kali setiap pengguna memberikan rating
user_rating_count = df.groupby('User-ID').count()['Book-Rating']

# Menghitung berapa kali setiap buku menerima rating
book_rating_count = df.groupby('Book-Title').count()['Book-Rating']


# --- 3. Filtering Pengguna dan Buku ---
# Ambil pengguna yang telah memberikan lebih dari 200 rating
active_users = user_rating_count[user_rating_count >= 200].index

# Ambil buku yang telah menerima lebih dari 100 rating
popular_books = book_rating_count[book_rating_count >= 100].index

# Membuat dataframe baru yang hanya berisi pengguna aktif dan buku populer
filtered_df = df[df['User-ID'].isin(active_users) & df['Book-Title'].isin(popular_books)]

print(f"\nUkuran dataframe setelah difilter: {filtered_df.shape}")


# --- 4. Membuat Matriks User-Item ---
# Menggunakan pivot_table untuk mengubah data menjadi matriks
# Index: Judul Buku, Kolom: User ID, Nilai: Rating
user_item_matrix = filtered_df.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')

# Mengisi nilai NaN (buku yang tidak diberi rating oleh pengguna) dengan 0
user_item_matrix.fillna(0, inplace=True)


# --- 5. Tampilkan Hasil ---
print("Dimensi matriks User-Item:", user_item_matrix.shape)
print("\nContoh matriks User-Item:")
print(user_item_matrix.head())

# ==============================================================================
# A. Menghitung Similarity dengan Euclidean Distance
# ==============================================================================

print("\n--- Menghitung Euclidean Distance Antar Buku ---")

# Menghitung jarak euclidean antar setiap baris (buku) di matriks
euclidean_dist = euclidean_distances(user_item_matrix)

# Membuat dataframe dari hasil perhitungan jarak
# Index dan kolomnya adalah judul buku
euclidean_dist_df = pd.DataFrame(euclidean_dist, index=user_item_matrix.index, columns=user_item_matrix.index)

print("Matriks Jarak Euclidean berhasil dibuat.")
print("Dimensi matriks:", euclidean_dist_df.shape)

print("\nContoh Matriks Jarak Euclidean (5x5):")
# Tampilkan 5 buku pertama dan jaraknya terhadap 5 buku lainnya
print(euclidean_dist_df.iloc[0:5, 0:5])

# ------------------------------------------------------------------------------
# Membuat Fungsi Rekomendasi Euclidean
# ------------------------------------------------------------------------------

def recommend_euclidean(book_title, distance_matrix, num_recommendations=5):
    """
    Memberikan rekomendasi buku berdasarkan jarak Euclidean terdekat.

    Args:
    - book_title (str): Judul buku yang ingin dicari rekomendasinya.
    - distance_matrix (pd.DataFrame): Matriks jarak Euclidean yang sudah dihitung.
    - num_recommendations (int): Jumlah buku yang ingin direkomendasikan.

    Returns:
    - list: Daftar judul buku yang direkomendasikan.
    """
    print(f"\n--- Rekomendasi untuk buku: '{book_title}' ---")

    try:
        # 1. Ambil jarak buku yang dipilih terhadap semua buku lain
        book_distances = distance_matrix[book_title]

        # 2. Urutkan jarak dari yang terkecil (paling mirip) ke terbesar
        # Kita ambil num_recommendations + 1 karena yang pertama adalah buku itu sendiri (jarak 0)
        similar_books = book_distances.sort_values(ascending=True)[1:num_recommendations+1]

        print("Buku-buku berikut direkomendasikan:")
        for i, book in enumerate(similar_books.index):
            # Ambil nilai jaraknya untuk ditampilkan
            dist = similar_books.iloc[i]
            print(f"{i+1}. {book} (Jarak: {dist:.2f})")

        return similar_books.index.tolist()

    except KeyError:
        return f"Error: Buku dengan judul '{book_title}' tidak ditemukan dalam matriks. Pastikan judulnya benar."


# --- Contoh Penggunaan Fungsi ---
# Kita coba cari rekomendasi untuk buku '1984'
# Pastikan buku ini ada di dalam matriks kita
if '1984' in euclidean_dist_df.index:
    recommend_euclidean('1984', euclidean_dist_df)
else:
    # Jika '1984' tidak ada, coba dengan buku pertama di matriks
    sample_book = euclidean_dist_df.index[0]
    recommend_euclidean(sample_book, euclidean_dist_df)

# ==============================================================================
# B. Menghitung Similarity dengan Cosine Similarity
# ==============================================================================

print("\n--- Menghitung Cosine Similarity Antar Buku ---")

# Menghitung cosine similarity antar setiap baris (buku) di matriks
# Matriks ini sudah memiliki nilai 0 untuk rating yang tidak ada, yang cocok untuk cosine similarity
cosine_sim = cosine_similarity(user_item_matrix)

# Membuat dataframe dari hasil perhitungan similaritas
# Index dan kolomnya adalah judul buku
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

print("Matriks Cosine Similarity berhasil dibuat.")
print("Dimensi matriks:", cosine_sim_df.shape)

print("\nContoh Matriks Cosine Similarity (5x5):")
# Tampilkan 5 buku pertama dan similaritasnya terhadap 5 buku lainnya
print(cosine_sim_df.iloc[0:5, 0:5])

# ------------------------------------------------------------------------------
# Membuat Fungsi Rekomendasi Cosine Similarity
# ------------------------------------------------------------------------------

def recommend_cosine(book_title, similarity_matrix, num_recommendations=5):
    """
    Memberikan rekomendasi buku berdasarkan Cosine Similarity tertinggi.

    Args:
    - book_title (str): Judul buku yang ingin dicari rekomendasinya.
    - similarity_matrix (pd.DataFrame): Matriks Cosine Similarity yang sudah dihitung.
    - num_recommendations (int): Jumlah buku yang ingin direkomendasikan.

    Returns:
    - list: Daftar judul buku yang direkomendasikan.
    """
    print(f"\n--- Rekomendasi untuk buku: '{book_title}' ---")

    try:
        # 1. Ambil similaritas buku yang dipilih terhadap semua buku lain
        book_similarities = similarity_matrix[book_title]

        # 2. Urutkan similaritas dari yang terbesar (paling mirip) ke terkecil
        # Kita ambil num_recommendations + 1 karena yang pertama adalah buku itu sendiri (similaritas 1.0)
        similar_books = book_similarities.sort_values(ascending=False)[1:num_recommendations+1]

        print("Buku-buku berikut direkomendasikan:")
        for i, book in enumerate(similar_books.index):
            # Ambil nilai similaritasnya untuk ditampilkan
            sim_score = similar_books.iloc[i]
            print(f"{i+1}. {book} (Similaritas: {sim_score:.2f})")

        return similar_books.index.tolist()

    except KeyError:
        return f"Error: Buku dengan judul '{book_title}' tidak ditemukan dalam matriks. Pastikan judulnya benar."


# --- Contoh Penggunaan Fungsi ---
# Kita coba cari rekomendasi untuk buku '1984' untuk perbandingan
if '1984' in cosine_sim_df.index:
    recommend_cosine('1984', cosine_sim_df)
else:
    # Jika '1984' tidak ada, coba dengan buku pertama di matriks
    sample_book = cosine_sim_df.index[0]
    recommend_cosine(sample_book, cosine_sim_df)

# ==============================================================================
# C. Menghitung Similarity dengan Pearson Correlation
# ==============================================================================

print("\n--- Menghitung Pearson Correlation Antar Buku ---")

# Menghitung korelasi pearson antar setiap buku.
# Kita transpose (.T) matriksnya agar buku menjadi kolom, karena .corr() bekerja pada kolom.
pearson_corr_df = user_item_matrix.T.corr(method='pearson')

print("Matriks Korelasi Pearson berhasil dibuat.")
print("Dimensi matriks:", pearson_corr_df.shape)

print("\nContoh Matriks Korelasi Pearson (5x5):")
# Tampilkan 5 buku pertama dan korelasinya terhadap 5 buku lainnya
print(pearson_corr_df.iloc[0:5, 0:5])

# ------------------------------------------------------------------------------
# Membuat Fungsi Rekomendasi Pearson Correlation
# ------------------------------------------------------------------------------

def recommend_pearson(book_title, correlation_matrix, num_recommendations=5):
    """
    Memberikan rekomendasi buku berdasarkan Korelasi Pearson tertinggi.

    Args:
    - book_title (str): Judul buku yang ingin dicari rekomendasinya.
    - correlation_matrix (pd.DataFrame): Matriks Korelasi Pearson yang sudah dihitung.
    - num_recommendations (int): Jumlah buku yang ingin direkomendasikan.

    Returns:
    - list: Daftar judul buku yang direkomendasikan.
    """
    print(f"\n--- Rekomendasi untuk buku: '{book_title}' ---")

    try:
        # 1. Ambil korelasi buku yang dipilih terhadap semua buku lain
        book_correlations = correlation_matrix[book_title]

        # 2. Urutkan korelasi dari yang terbesar (paling mirip) ke terkecil
        similar_books = book_correlations.sort_values(ascending=False)[1:num_recommendations+1]

        print("Buku-buku berikut direkomendasikan:")
        for i, book in enumerate(similar_books.index):
            # Ambil nilai korelasinya untuk ditampilkan
            corr_score = similar_books.iloc[i]
            print(f"{i+1}. {book} (Korelasi: {corr_score:.2f})")

        return similar_books.index.tolist()

    except KeyError:
        return f"Error: Buku dengan judul '{book_title}' tidak ditemukan dalam matriks. Pastikan judulnya benar."


# --- Contoh Penggunaan Fungsi ---
# Kita coba cari rekomendasi untuk buku '1984' untuk perbandingan ketiga
if '1984' in pearson_corr_df.index:
    recommend_pearson('1984', pearson_corr_df)
else:
    # Jika '1984' tidak ada, coba dengan buku pertama di matriks
    sample_book = pearson_corr_df.index[0]
    recommend_pearson(sample_book, pearson_corr_df)

# ==============================================================================
# 5. IMPLEMENTASI MODEL COLLABORATIVE FILTERING
# ==============================================================================

class ItemItemRecommender:
    """
    Sebuah class untuk sistem rekomendasi Item-Item Collaborative Filtering.
    
    Metodologi:
    1.  Fit: Menerima data, memfilter pengguna aktif & buku populer, lalu
        membangun matriks interaksi dan matriks similaritas.
    2.  Predict: Menerima judul buku, lalu memberikan rekomendasi berdasarkan
        matriks similaritas yang telah dibuat.
    """
    def __init__(self):
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.df = None

    def fit(self, ratings_df, books_df, min_user_ratings=200, min_book_ratings=100, similarity_method='pearson'):
        """
        Mempersiapkan model dengan data yang ada.
        """
        print("--- (Fit) Memulai persiapan model ---")
        
        # 1. Gabungkan & Filter Data
        df = pd.merge(ratings_df, books_df, on='ISBN')
        user_rating_count = df.groupby('User-ID').count()['Book-Rating']
        book_rating_count = df.groupby('Book-Title').count()['Book-Rating']
        
        active_users = user_rating_count[user_rating_count >= min_user_ratings].index
        popular_books = book_rating_count[book_rating_count >= min_book_ratings].index
        
        self.df = df[df['User-ID'].isin(active_users) & df['Book-Title'].isin(popular_books)]
        print(f"(Fit) Data berhasil difilter. Ukuran data: {self.df.shape}")

        # 2. Buat Matriks User-Item
        self.user_item_matrix = self.df.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)
        print(f"(Fit) Matriks User-Item dibuat dengan dimensi: {self.user_item_matrix.shape}")

        # 3. Hitung Matriks Similaritas
        if similarity_method == 'pearson':
            self.similarity_matrix = self.user_item_matrix.T.corr(method='pearson')
        elif similarity_method == 'cosine':
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
            self.similarity_matrix = pd.DataFrame(self.similarity_matrix, index=self.user_item_matrix.index, columns=self.user_item_matrix.index)
        else:
            raise ValueError("Metode similaritas tidak valid. Pilih 'pearson' atau 'cosine'.")
            
        print(f"(Fit) Matriks Similaritas '{similarity_method}' berhasil dibuat.")
        print("--- (Fit) Model siap digunakan ---")

    def predict(self, book_title, num_recommendations=5):
        """
        Memberikan rekomendasi untuk sebuah buku.
        """
        if self.similarity_matrix is None:
            print("Error: Model belum di-fit. Jalankan metode .fit() terlebih dahulu.")
            return

        print(f"\n--- (Predict) Rekomendasi untuk buku: '{book_title}' ---")
        
        if book_title not in self.similarity_matrix:
            print(f"Error: Buku '{book_title}' tidak ditemukan.")
            return

        # Ambil similaritas dan urutkan
        similar_books = self.similarity_matrix[book_title].sort_values(ascending=False)[1:num_recommendations+1]
        
        print("Buku-buku berikut direkomendasikan:")
        for i, book in enumerate(similar_books.index):
            score = similar_books.iloc[i]
            print(f"{i+1}. {book} (Skor Similaritas: {score:.2f})")
            
        return similar_books.index.tolist()
    
# ------------------------------------------------------------------------------
# Contoh Penggunaan Class Collaborative Filtering
# ------------------------------------------------------------------------------

# 1. Buat instance dari model
recommender = ItemItemRecommender()

# 2. "Train" model dengan data yang kita punya (menggunakan metode Pearson)
#    (Gunakan dataframe `ratings` dan `books` yang sudah bersih dari tahap preprocessing)
recommender.fit(ratings, books, similarity_method='pearson')

# 3. Dapatkan rekomendasi untuk sebuah buku
recommender.predict('1984')

# ==============================================================================
# 6. EVALUASI MODEL
# ==============================================================================

def evaluate_model(recommender_model, num_users_to_test=20, top_n_recommendations=10, min_rating=9):
    """
    Mengevaluasi model recommender dengan strategi hit rate sederhana.
    """
    print("\n--- Memulai Evaluasi Model ---")
    
    # Ambil daftar pengguna dari matriks yang sudah ada di model
    all_users = recommender_model.user_item_matrix.columns
    # Pilih pengguna secara acak untuk diuji
    test_users = np.random.choice(all_users, size=num_users_to_test, replace=False)
    
    total_hits = 0
    
    for i, user_id in enumerate(test_users):
        print(f"\n({i+1}/{num_users_to_test}) Mengevaluasi Pengguna ID: {user_id}...")
        
        # 1. Dapatkan semua buku yang pernah diberi rating oleh pengguna ini dari dataframe asli yang difilter
        user_rated_books = recommender_model.df[recommender_model.df['User-ID'] == user_id]
        
        # 2. Cari buku yang sangat disukai pengguna (rating >= min_rating) sebagai 'test case'
        highly_rated = user_rated_books[user_rated_books['Book-Rating'] >= min_rating]
        
        if highly_rated.empty:
            print("  -> Pengguna ini tidak memiliki buku dengan rating tinggi. Dilewati.")
            continue
            
        # Ambil satu buku sebagai 'pemicu' rekomendasi
        test_book_title = highly_rated.iloc[0]['Book-Title']
        
        # 3. Dapatkan top N rekomendasi untuk buku ini
        # Kita panggil fungsi predict dari class recommender kita
        recommendations = recommender_model.predict(test_book_title, num_recommendations=top_n_recommendations)
        
        if recommendations is None:
            continue
        
        # 4. Cek berapa banyak rekomendasi yang 'kena' (ada di daftar buku yang sudah dibaca pengguna)
        # Ambil semua judul buku yang pernah dibaca pengguna
        all_user_read_titles = user_rated_books['Book-Title'].tolist()
        
        # Hitung berapa banyak dari rekomendasi yang ada di daftar bacaan pengguna
        hits = 0
        for book in recommendations:
            if book in all_user_read_titles:
                hits += 1
                print(f"  -> HIT! Buku '{book}' ada di daftar bacaan pengguna.")
        
        print(f"  -> Pengguna {user_id} mendapatkan {hits} hit dari {top_n_recommendations} rekomendasi.")
        total_hits += hits

    print("\n--- Hasil Evaluasi Selesai ---")
    print(f"Dari {num_users_to_test} pengguna yang diuji, ditemukan total {total_hits} rekomendasi yang relevan (hits).")
    
    return total_hits

# --- Menjalankan Evaluasi ---
# Pastikan Anda sudah membuat dan me-fit() objek `recommender` dari tahap sebelumnya
evaluate_model(recommender)