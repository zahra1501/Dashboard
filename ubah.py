import geopandas as gpd
import os

os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

# Baca file GeoJSON asli (pastikan path-nya benar)
gdf = gpd.read_file("batas_wilayah_indonesia.geojson")

# Simpan sebagai Parquet (format lebih ringan dan cepat)
gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)
gdf.to_parquet("batas_wilayah_simplified.parquet")

print("Konversi selesai. File disimpan sebagai Parquet.")
