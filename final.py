# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.options.display.float_format = '{:,.2f}'.format


print("Daftar file di folder project:")
print(os.listdir("."))


print("IDENTITAS DATASET")
print("=================")
print("Nama Dataset       : Used Car Listings in Indonesia")
print("Sumber Dataset     : Kaggle")
print("Link Kaggle        : https://www.kaggle.com/datasets/indraputra21/used-car-listings-in-indonesia/data")
print("Jenis data         : Data listing mobil bekas di Indonesia")
print("Studi kasus        : Prediksi harga mobil bekas di Indonesia")
print("Metode             : Regresi Linear Berganda")
print("Variabel target Y  : price_rp / harga mobil")
print("Variabel X         : year, mileage, brand, model, transmission, fuel_type, dan fitur mobil")


# Mengambil file CSV dari folder yang sama dengan file Python
csv_files = [file for file in os.listdir(".") if file.endswith(".csv")]

if len(csv_files) == 0:
    raise FileNotFoundError("File CSV belum ditemukan. Simpan file dataset CSV di folder yang sama dengan final.py.")

file_path = csv_files[0]


# Membaca dataset dengan aman
try:
    df = pd.read_csv(file_path)

    if df.shape[1] == 1:
        df = pd.read_csv(file_path, sep=";")

except:
    try:
        df = pd.read_csv(file_path, sep=";")
    except:
        df = pd.read_csv(file_path, sep=None, engine="python")


print("\nDataset berhasil dibaca.")
print("File yang digunakan:", file_path)
print("Jumlah baris dan kolom:", df.shape)


print("\n5 data teratas:")
print(df.head())

print("\n5 data terbawah:")
print(df.tail())


jumlah_baris, jumlah_kolom = df.shape

print("\nJumlah data/baris :", jumlah_baris)
print("Jumlah kolom      :", jumlah_kolom)

print("\nNama-nama kolom dalam dataset:")
print(df.columns.tolist())


print("\nInformasi dataset:")
df.info()


print("\nJumlah data kosong pada setiap kolom:")
print(df.isnull().sum())

print("\nTotal seluruh data kosong:", df.isnull().sum().sum())


print("\nStatistik deskriptif dataset:")
print(df.describe())


df_clean = df.copy()

df_clean.columns = (
    df_clean.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("/", "_", regex=False)
    .str.replace("-", "_", regex=False)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("|", "_", regex=False)
    .str.replace("%", "percent", regex=False)
)

print("\nNama kolom setelah dibersihkan:")
print(df_clean.columns.tolist())


def cari_kolom(kata_kunci):
    for kolom in df_clean.columns:
        for kata in kata_kunci:
            if kata in kolom:
                return kolom
    return None


kolom_price = cari_kolom(["price_rp", "price", "harga"])
kolom_year = cari_kolom(["year", "tahun"])
kolom_mileage = cari_kolom(["mileage_km", "mileage", "kilometer", "km"])
kolom_brand = cari_kolom(["brand", "merk", "merek"])
kolom_model = cari_kolom(["model"])
kolom_transmission = cari_kolom(["transmission", "transmisi"])
kolom_fuel = cari_kolom(["fuel", "bahan_bakar"])


print("\nKolom harga/Y     :", kolom_price)
print("Kolom tahun       :", kolom_year)
print("Kolom mileage     :", kolom_mileage)
print("Kolom brand       :", kolom_brand)
print("Kolom model       :", kolom_model)
print("Kolom transmisi   :", kolom_transmission)
print("Kolom fuel type   :", kolom_fuel)


def ubah_ke_angka(series):
    series = series.astype(str)
    series = series.str.replace("Rp", "", regex=False)
    series = series.str.replace("rp", "", regex=False)
    series = series.str.replace("KM", "", regex=False)
    series = series.str.replace("km", "", regex=False)
    series = series.str.replace(".", "", regex=False)
    series = series.str.replace(",", "", regex=False)
    series = series.str.strip()

    return pd.to_numeric(series, errors="coerce")


kolom_angka = [kolom_year, kolom_mileage, kolom_price]

for kolom in kolom_angka:
    if kolom is not None:
        df_clean[kolom] = ubah_ke_angka(df_clean[kolom])


print("\nTipe data setelah kolom angka dibersihkan:")
print(df_clean[kolom_angka].dtypes)

print("\nContoh data kolom angka:")
print(df_clean[kolom_angka].head())


kolom_fitur = [
    "rear_camera",
    "sun_roof",
    "auto_retract_mirror",
    "electric_parking_brake",
    "map_navigator",
    "vehicle_stability_control",
    "keyless_push_start"
]

# Hanya ambil fitur yang memang ada di dataset
kolom_fitur_ada = [kolom for kolom in kolom_fitur if kolom in df_clean.columns]


kolom_dipakai = [
    kolom_year,
    kolom_mileage,
    kolom_brand,
    kolom_model,
    kolom_transmission,
    kolom_fuel
] + kolom_fitur_ada + [kolom_price]


# Hapus kolom None jika ada yang tidak ditemukan
kolom_dipakai = [kolom for kolom in kolom_dipakai if kolom is not None]

data_regresi = df_clean[kolom_dipakai].copy()


print("\nKolom yang digunakan untuk regresi:")
print(kolom_dipakai)

print("\nData yang digunakan untuk regresi:")
print("Jumlah baris dan kolom:", data_regresi.shape)

print("\nContoh data regresi:")
print(data_regresi.head())


print("\nData kosong sebelum dibersihkan:")
print(data_regresi.isnull().sum())


# Menghapus data kosong
data_regresi = data_regresi.dropna()

# Menghapus harga mobil yang 0 atau negatif
data_regresi = data_regresi[data_regresi[kolom_price] > 0]

# Menghapus mileage negatif jika ada
data_regresi = data_regresi[data_regresi[kolom_mileage] >= 0]


print("\nData kosong setelah dibersihkan:")
print(data_regresi.isnull().sum())

print("\nJumlah data setelah dibersihkan:", data_regresi.shape)

print("\nContoh data setelah dibersihkan:")
print(data_regresi.head())


print("\nVARIABEL PENELITIAN")
print("===================")
print("Y  =", kolom_price, ": Harga mobil bekas yang akan diprediksi")
print("X1 =", kolom_year, ": Tahun produksi mobil")
print("X2 =", kolom_mileage, ": Jarak tempuh mobil dalam kilometer")
print("X3 =", kolom_brand, ": Brand / merek mobil")
print("X4 =", kolom_model, ": Model / tipe mobil")
print("X5 =", kolom_transmission, ": Jenis transmisi mobil")
print("X6 =", kolom_fuel, ": Jenis bahan bakar mobil")


print("\nFitur tambahan yang digunakan:")
for fitur in kolom_fitur_ada:
    print("-", fitur)


kolom_kategori = []

for kolom in [kolom_brand, kolom_model, kolom_transmission, kolom_fuel]:
    if kolom is not None and kolom in data_regresi.columns:
        kolom_kategori.append(kolom)


data_regresi_encoded = pd.get_dummies(
    data_regresi,
    columns=kolom_kategori,
    drop_first=True
)


print("\nKolom kategori yang diubah menjadi angka:")
print(kolom_kategori)

print("\nJumlah kolom sebelum encoding:", data_regresi.shape[1])
print("Jumlah kolom setelah encoding :", data_regresi_encoded.shape[1])

print("\nContoh data setelah encoding:")
print(data_regresi_encoded.head())


korelasi = data_regresi_encoded.corr(numeric_only=True)

print("\nKorelasi setiap variabel terhadap harga mobil:")
print(korelasi[kolom_price].sort_values(ascending=False).head(15))


korelasi_harga = korelasi[kolom_price].drop(kolom_price)

# Ambil 10 variabel dengan hubungan terkuat
korelasi_terkuat = korelasi_harga.abs().sort_values(ascending=False).head(10)
nama_kolom_terkuat = korelasi_terkuat.index


plt.figure(figsize=(9, 5))
plt.barh(nama_kolom_terkuat, korelasi_harga[nama_kolom_terkuat])
plt.xlabel("Nilai Korelasi terhadap Harga Mobil")
plt.ylabel("Parameter")
plt.title("10 Parameter yang Paling Berhubungan dengan Harga Mobil")
plt.tight_layout()
plt.show()


X = data_regresi_encoded.drop(columns=[kolom_price])
y = data_regresi_encoded[kolom_price]


print("\nJumlah kolom X setelah brand/model/transmission/fuel_type dimasukkan:", X.shape[1])

print("\nContoh variabel X:")
print(X.head())

print("\nVariabel Y:")
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nJumlah data training:", X_train.shape[0])
print("Jumlah data testing :", X_test.shape[0])


model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Regresi Linear Berganda berhasil dibuat.")

print("Konstanta / Intercept:", round(model.intercept_, 4))


koefisien = pd.DataFrame({
    "Parameter": X.columns,
    "Koefisien": model.coef_
})

koefisien["Koefisien"] = koefisien["Koefisien"].round(4)

print("\n20 Koefisien terbesar:")
print(koefisien.sort_values(by="Koefisien", ascending=False).head(20))


y_pred = model.predict(X_test)


hasil_prediksi = pd.DataFrame({
    "Harga Asli Mobil": y_test.values,
    "Harga Prediksi Mobil": y_pred
})

hasil_prediksi["Harga Prediksi Mobil"] = hasil_prediksi["Harga Prediksi Mobil"].round(2)


print("\nHasil prediksi awal:")
print(hasil_prediksi.head(20))


def format_rupiah(nilai):
    return "Rp {:,.0f}".format(float(nilai)).replace(",", ".")


hasil_prediksi_tampil = hasil_prediksi.head(20).copy()

hasil_prediksi_tampil["Harga Asli Mobil"] = hasil_prediksi_tampil["Harga Asli Mobil"].apply(format_rupiah)
hasil_prediksi_tampil["Harga Prediksi Mobil"] = hasil_prediksi_tampil["Harga Prediksi Mobil"].apply(format_rupiah)


print("\nHasil prediksi dalam format Rupiah:")
print(hasil_prediksi_tampil)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("\nHASIL EVALUASI MODEL")
print("====================")
print("Mean Absolute Error (MAE):", format_rupiah(mae))
print("Mean Squared Error (MSE) :", round(mse, 4))
print("Root Mean Squared Error  :", format_rupiah(rmse))
print("R-Squared (R2)           :", round(r2, 4))


persamaan = f"{kolom_price} = {model.intercept_:.4f}"

for kolom, coef in zip(X.columns, model.coef_):
    persamaan += f" + ({coef:.4f} × {kolom})"


print("\nPersamaan Regresi Linear Berganda:")
print(persamaan[:3000])

if len(persamaan) > 3000:
    print("\nPersamaan terlalu panjang, hanya ditampilkan sebagian.")


plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Harga Mobil Asli")
plt.ylabel("Harga Mobil Prediksi")
plt.title("Perbandingan Harga Mobil Asli dan Prediksi")
plt.tight_layout()
plt.show()


cek_prediksi = X_test.copy()

cek_prediksi["Harga Asli Mobil"] = y_test.values
cek_prediksi["Harga Prediksi Mobil"] = y_pred.round(2)


print("\nCek prediksi:")
print(cek_prediksi.head(20))


# Membuat data baru kosong dengan semua kolom X bernilai 0
data_baru = pd.DataFrame(0, index=[0], columns=X.columns)

# Mengisi data numerik
data_baru[kolom_year] = 2020
data_baru[kolom_mileage] = 45000


# Mengisi contoh kategori jika kolomnya ada
# Contoh: Toyota Avanza Automatic Petrol
for kolom in data_baru.columns:
    if "brand_toyota" in kolom.lower():
        data_baru[kolom] = 1
    if "model_avanza" in kolom.lower():
        data_baru[kolom] = 1
    if "transmission_automatic" in kolom.lower():
        data_baru[kolom] = 1
    if "fuel" in kolom.lower() and "petrol" in kolom.lower():
        data_baru[kolom] = 1


# Mengisi fitur jika kolomnya ada
for fitur in kolom_fitur_ada:
    if fitur in data_baru.columns:
        data_baru[fitur] = 1


prediksi_baru = model.predict(data_baru)


print("\nData mobil baru:")
print(data_baru)

print("\nPrediksi harga mobil:", format_rupiah(prediksi_baru[0]))


print("\nKESIMPULAN ANALISIS")
print("===================")

print("Dataset yang digunakan adalah Used Car Listings in Indonesia.")
print("Dataset berasal dari Kaggle.")
print("Link Kaggle: https://www.kaggle.com/datasets/indraputra21/used-car-listings-in-indonesia/data")
print(f"Jumlah data awal adalah {df.shape[0]} baris dan {df.shape[1]} kolom.")
print(f"Jumlah data setelah dibersihkan adalah {data_regresi.shape[0]} baris.")
print("Metode yang digunakan adalah Regresi Linear Berganda.")
print(f"Variabel target / Y adalah {kolom_price}, yaitu harga mobil bekas dalam Rupiah.")


print("\nParameter utama / X yang digunakan adalah:")
print("-", kolom_year)
print("-", kolom_mileage)
print("-", kolom_brand)
print("-", kolom_model)
print("-", kolom_transmission)
print("-", kolom_fuel)


print("\nFitur tambahan yang digunakan:")
for fitur in kolom_fitur_ada:
    print("-", fitur)


print("\nHasil evaluasi model:")
print("MAE  :", format_rupiah(mae))
print("RMSE :", format_rupiah(rmse))
print("R2   :", round(r2, 4))


print("\nKesimpulan akhir:")
print("Harga mobil bekas dapat diprediksi menggunakan tahun mobil, jarak tempuh, brand, model, transmisi, jenis bahan bakar, dan fitur mobil.")
print("Penambahan brand, model, transmission, dan fuel_type membuat model lebih memahami perbedaan harga antar jenis mobil.")
print("Dengan begitu, prediksi dapat menjadi lebih masuk akal dibandingkan hanya memakai tahun, kilometer, dan fitur saja.")


ringkasan = f"""
RINGKASAN LAPORAN

Judul:
Analisis Regresi Linear Berganda untuk Memprediksi Harga Mobil Bekas di Indonesia Berdasarkan Tahun Kendaraan, Jarak Tempuh, Brand, Model, Transmisi, Jenis Bahan Bakar, dan Fitur Mobil

Dataset:
Used Car Listings in Indonesia

Sumber:
Kaggle

Link:
https://www.kaggle.com/datasets/indraputra21/used-car-listings-in-indonesia/data

Jumlah Data Awal:
{df.shape[0]} data mobil

Jumlah Data Setelah Dibersihkan:
{data_regresi.shape[0]} data mobil

Jumlah Kolom Dataset:
{df.shape[1]} kolom

Metode:
Regresi Linear Berganda

Variabel Target / Y:
{kolom_price}

Parameter / X:
{kolom_year}, {kolom_mileage}, {kolom_brand}, {kolom_model}, {kolom_transmission}, {kolom_fuel}, dan fitur-fitur mobil.

Alasan Memilih Dataset:
Dataset ini dipilih karena berasal dari Kaggle, berisi data mobil bekas di Indonesia, memiliki target numerik berupa harga mobil, dan cocok digunakan untuk analisis regresi linear berganda.

Kesimpulan:
Model regresi linear berganda digunakan untuk memprediksi harga mobil bekas berdasarkan tahun kendaraan, jarak tempuh, brand, model, transmisi, jenis bahan bakar, dan fitur mobil. Penambahan kolom brand dan model membuat model lebih memahami perbedaan harga antar jenis mobil.
"""

print(ringkasan)