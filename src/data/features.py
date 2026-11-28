"""Variables: hora, día semana, festivo, lags 24h/168h, temperatura."""
"""Ingeniería de características sobre data/processed/demanda_limpia.csv.

Uso (desde la raíz del proyecto):
    python -m src.data.features
"""
import numpy as np
import pandas as pd

from src.data.limpieza import DATA_PROCESSED, RUTA_LIMPIA

RUTA_FEATURES = DATA_PROCESSED / "dataset_features.csv"


def generar_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Añade variables de calendario, térmicas y autoregresivas (lags)."""
    if df is None:
        df = pd.read_csv(RUTA_LIMPIA)

    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True)
    df = df.sort_values("dt_utc").reset_index(drop=True)
    dt_madrid = df["dt_utc"].dt.tz_convert("Europe/Madrid")

    # 1. Variables de calendario
    df["hora"] = dt_madrid.dt.hour
    df["dia_semana"] = dt_madrid.dt.day_of_week          # 0=Lunes ... 6=Domingo
    df["dia_anio"] = dt_madrid.dt.day_of_year
    df["mes"] = dt_madrid.dt.month
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["es_laborable"] = ((df["es_fin_semana"] == 0) & (df["es_festivo"] == 0)).astype(int)

    # Codificación cíclica (preserva continuidad 23h -> 00h y Dic -> Ene)
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

    # 2. Variables térmicas (relación no lineal en U entre temperatura y demanda)
    # Base estándar 18°C: demanda de calefacción (HDD) y refrigeración (CDD)
    df["hdd_18"] = np.maximum(0.0, 18.0 - df["temp_media"])
    df["cdd_22"] = np.maximum(0.0, df["temp_media"] - 22.0)
    df["temp_media_24h"] = df["temp_media"].rolling(window=24, min_periods=1).mean()

    # 3. Retardos (lags) y estadísticos móviles de demanda
    for lag in (1, 24, 48, 168):  # 1h, 1 día, 2 días, 1 semana
        df[f"demanda_lag_{lag}h"] = df["demanda_real"].shift(lag)

    df["demanda_media_24h"] = df["demanda_real"].shift(1).rolling(window=24).mean()
    df["demanda_std_24h"] = df["demanda_real"].shift(1).rolling(window=24).std()
    df["demanda_media_168h"] = df["demanda_real"].shift(1).rolling(window=168).mean()

    # Eliminar filas iniciales sin historial completo (primeros 7 días = 168h)
    df_final = df.dropna().reset_index(drop=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(RUTA_FEATURES, index=False)
    print(f"Guardado {RUTA_FEATURES}: {len(df_final)} filas x {df_final.shape[1]} columnas.")
    return df_final


if __name__ == "__main__":
    generar_features()
