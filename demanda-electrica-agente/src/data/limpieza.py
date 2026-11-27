"""Limpieza y unificación de las fuentes en un único dataset."""
"""Limpieza y cruce de fuentes brutas a data/processed/demanda_limpia.csv.

Uso (desde la raíz del proyecto):
    python -m src.data.limpieza
"""
from pathlib import Path
import pandas as pd

from src.config import CIUDADES, DATA_RAW

DATA_PROCESSED = DATA_RAW.parent / "processed"
RUTA_LIMPIA = DATA_PROCESSED / "demanda_limpia.csv"


def _cargar_demanda() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "demanda_horaria.csv")
    df["dt_utc"] = pd.to_datetime(df["datetime"], utc=True)
    return (
        df[["dt_utc", "demanda_real"]]
        .drop_duplicates("dt_utc")
        .sort_values("dt_utc")
    )


def _cargar_temperaturas() -> pd.DataFrame:
    dfs = []
    for ciudad in CIUDADES:
        ruta = DATA_RAW / "openmeteo" / f"temperatura_{ciudad}.csv"
        df_c = pd.read_csv(ruta)
        dt_naive = pd.to_datetime(df_c["datetime"])
        # True para la primera aparición (horario verano +02:00), False si hubiera duplicado (+01:00)
        es_dst = ~dt_naive.duplicated(keep="first")
        dt_madrid = dt_naive.dt.tz_localize(
            "Europe/Madrid",
            ambiguous=es_dst,
            nonexistent="shift_forward",
        )
        df_c["dt_utc"] = dt_madrid.dt.tz_convert("UTC")
        df_c = (
            df_c[["dt_utc", f"temp_{ciudad}"]]
            .drop_duplicates("dt_utc")
            .sort_values("dt_utc")
        )
        dfs.append(df_c.set_index("dt_utc"))

    df_meteo = pd.concat(dfs, axis=1).reset_index()
    cols_temp = [f"temp_{c}" for c in CIUDADES]
    df_meteo["temp_media"] = df_meteo[cols_temp].mean(axis=1)
    return df_meteo

def _cargar_festivos() -> set[str]:
    df_f = pd.read_csv(DATA_RAW / "festivos_nacionales.csv")
    return set(df_f["fecha"].astype(str))


def limpiar_datos() -> pd.DataFrame:
    """Cruza demanda, temperaturas y festivos resolviendo cambios de hora (DST)."""
    df_dem = _cargar_demanda()
    df_meteo = _cargar_temperaturas()
    festivos = _cargar_festivos()

    df = pd.merge(df_dem, df_meteo, on="dt_utc", how="inner")
    df = df.sort_values("dt_utc").reset_index(drop=True)

    # Interpolación lineal por si existiera algún hueco puntual en sensores/API
    cols_num = ["demanda_real", "temp_media"] + [f"temp_{c}" for c in CIUDADES]
    df[cols_num] = df[cols_num].interpolate(method="linear", limit=3)

    # Convertir a hora oficial peninsular para calendario y festivos
    dt_madrid = df["dt_utc"].dt.tz_convert("Europe/Madrid")
    df["datetime"] = dt_madrid
    df["fecha"] = dt_madrid.dt.strftime("%Y-%m-%d")
    df["es_festivo"] = df["fecha"].isin(festivos).astype(int)

    cols_finales = (
        ["datetime", "dt_utc", "demanda_real", "temp_media"]
        + [f"temp_{c}" for c in CIUDADES]
        + ["es_festivo"]
    )
    df = df[cols_finales]

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUTA_LIMPIA, index=False)
    print(f"Guardado {RUTA_LIMPIA}: {len(df)} filas x {df.shape[1]} columnas "
          f"(nulos: {df.isna().sum().sum()}).")
    return df


if __name__ == "__main__":
    limpiar_datos()
