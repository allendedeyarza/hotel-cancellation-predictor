# src/etl.py
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "hotel_booking.csv"

def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """
    Carga los datos de reservas de hotel y hace una limpieza básica.
    """
    df = pd.read_csv(path)

    # 🔽 AQUÍ pegas la parte de limpieza de tu notebook 🔽
    # Ejemplo (ajusta según tu notebook):
    if "stays_in_weekend_nights" in df.columns and "stays_in_week_nights" in df.columns:
        df["total_nights"] = (
            df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        )

    # Quitar filas completamente vacías
    df = df.dropna(how="all")

    return df
