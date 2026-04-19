import pandas as pd
import os

def clean_slang_csv(input_file, output_file):
    # 1. Tentukan path root folder secara dinamis
    # __file__ adalah lokasi script ini (src/data/clean_csv.py)
    # Kita naik 2 level ke atas untuk sampai ke root (sentimen_analysis)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir))
    
    input_path = os.path.join(root_dir, input_file)
    output_path = os.path.join(root_dir, output_file)

    print(f"🔍 Mencari file di: {input_path}")

    if not os.path.exists(input_path):
        print(f"❌ Error: File '{input_path}' tidak ditemukan di root folder!")
        return

    try:
        # 2. Memuat file CSV (menggunakan semicolon sesuai request sebelumnya)
        df = pd.read_csv(input_path, sep=';')

        initial_count = len(df)

        # 3. Hapus duplikat total
        df = df.drop_duplicates()

        # 4. Hapus duplikat berdasarkan key 'singkatan' (biar unik)
        df = df.drop_duplicates(subset=['singkatan'], keep='first')

        # 5. Sort biar rapi
        df = df.sort_values(by='singkatan')

        # 6. Simpan ke root
        df.to_csv(output_path, index=False, sep=';')

        print(f"✅ Berhasil dibersihkan!")
        print(f"📊 Data awal: {initial_count} | Sekarang: {len(df)}")
        print(f"💾 File tersimpan di: {output_path}")

    except Exception as e:
        print(f"❌ Terjadi kesalahan saat memproses: {e}")

if __name__ == "__main__":
    clean_slang_csv('data\\slang_dirty.csv', 'data\\slang_clean.csv')