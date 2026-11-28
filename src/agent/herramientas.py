"""Herramientas: predecir_demanda, clasificar_picos, buscar_normativa."""
"""Herramientas locales del agente (CSV procesados, modelos ML y RAG Qdrant).

Regla: NUNCA se realizan llamadas a APIs externas desde este módulo.
"""
import joblib
import pandas as pd

from src.config import DATA_PROCESSED, MODELS_DIR, TOP_K
from src.rag.buscar import buscar_normativa

RUTA_LIMPIA = DATA_PROCESSED / "demanda_limpia.csv"
RUTA_FEATURES = DATA_PROCESSED / "dataset_features.csv"
RUTA_REG = MODELS_DIR / "modelo_regresion.joblib"
RUTA_CLF = MODELS_DIR / "modelo_clasificacion.joblib"

# Caché en memoria para no releer archivos en cada turno del agente
_CACHE: dict = {}


def _get_df_limpia() -> pd.DataFrame:
    if "limpia" not in _CACHE:
        df = pd.read_csv(RUTA_LIMPIA)
        df["dt_madrid"] = pd.to_datetime(df["dt_utc"], utc=True).dt.tz_convert("Europe/Madrid")
        _CACHE["limpia"] = df
    return _CACHE["limpia"]


def _get_df_features() -> pd.DataFrame:
    if "features" not in _CACHE:
        df = pd.read_csv(RUTA_FEATURES)
        df["dt_madrid"] = pd.to_datetime(df["dt_utc"], utc=True).dt.tz_convert("Europe/Madrid")
        _CACHE["features"] = df
    return _CACHE["features"]


def _get_modelo(clave: str, ruta):
    if clave not in _CACHE:
        _CACHE[clave] = joblib.load(ruta)
    return _CACHE[clave]


def consultar_historico(fecha: str | None = None, anio: int | None = None, mes: int | None = None) -> dict:
    """Consulta estadísticas históricas de demanda real y temperatura en datos locales.

    Args:
        fecha: Fecha exacta 'YYYY-MM-DD', o parcial 'YYYY-MM' / 'YYYY' (opcional).
        anio: Año entre 2021 y 2024 (opcional).
        mes: Mes entre 1 y 12 (opcional).
    """
    df = _get_df_limpia()
    filtro = df.copy()
    filtro["fecha"] = filtro["dt_madrid"].astype(str).str.slice(0, 10)

    if fecha:
        f_str = str(fecha).strip()
        if len(f_str) == 4 and f_str.isdigit():          # '2024'
            filtro = filtro[filtro["dt_madrid"].dt.year == int(f_str)]
        elif len(f_str) == 7:                            # '2024-01'
            filtro = filtro[filtro["fecha"].str.startswith(f_str)]
        else:                                            # '2024-01-15'
            filtro = filtro[filtro["fecha"] == f_str]
    else:
        if anio is not None:
            filtro = filtro[filtro["dt_madrid"].dt.year == int(anio)]
        if mes is not None:
            filtro = filtro[filtro["dt_madrid"].dt.month == int(mes)]

    if filtro.empty:
        return {"error": "No hay registros históricos para los filtros indicados (rango válido: 2021-01-01 a 2024-12-31)."}

    idx_max = filtro["demanda_real"].idxmax()
    fila_max = filtro.loc[idx_max]

    return {
        "registros_horas": int(len(filtro)),
        "demanda_media_MW": round(float(filtro["demanda_real"].mean()), 2),
        "demanda_max_MW": round(float(filtro["demanda_real"].max()), 2),
        "momento_pico": str(fila_max["datetime"]),
        "demanda_min_MW": round(float(filtro["demanda_real"].min()), 2),
        "temp_media_C": round(float(filtro["temp_media"].mean()), 2),
        "horas_festivo": int(filtro["es_festivo"].sum()),
    }


def predecir_demanda(fecha_hora: str) -> dict:
    """Predice la demanda eléctrica (MW) con el modelo de regresión para una fecha y hora.

    Args:
        fecha_hora: Timestamp en formato 'YYYY-MM-DD HH:00' (ej. '2024-07-15 14:00').
    """
    df = _get_df_features()
    ts = pd.to_datetime(fecha_hora, errors="coerce")
    if pd.isna(ts):
        return {"error": f"Formato de fecha_hora inválido: '{fecha_hora}'. Usa 'YYYY-MM-DD HH:00'."}

    if ts.tzinfo is None:
        ts = ts.tz_localize("Europe/Madrid", ambiguous=True, nonexistent="shift_forward")

    fila = df[df["dt_madrid"] == ts]
    if fila.empty:
        return {
            "error": (
                f"No hay features disponibles para '{fecha_hora}'. "
                f"Rango válido: {df['dt_madrid'].min()} a {df['dt_madrid'].max()}."
            )
        }

    bundle = _get_modelo("reg", RUTA_REG)
    X = fila[bundle["features"]]
    pred = float(bundle["modelo"].predict(X)[0])
    real = float(fila["demanda_real"].iloc[0])

    return {
        "fecha_hora": str(fila["datetime"].iloc[0]),
        "modelo": bundle["nombre"],
        "demanda_predicha_MW": round(pred, 2),
        "demanda_real_registrada_MW": round(real, 2),
        "error_absoluto_MW": round(abs(pred - real), 2),
        "temp_media_C": round(float(fila["temp_media"].iloc[0]), 2),
        "es_festivo": int(fila["es_festivo"].iloc[0]),
    }


def clasificar_hora_pico(fecha_hora: str) -> dict:
    """Clasifica si una fecha y hora corresponde a una hora pico crítica (percentil 95).

    Args:
        fecha_hora: Timestamp en formato 'YYYY-MM-DD HH:00' (ej. '2024-01-18 20:00').
    """
    df = _get_df_features()
    ts = pd.to_datetime(fecha_hora, errors="coerce")
    if pd.isna(ts):
        return {"error": f"Formato de fecha_hora inválido: '{fecha_hora}'. Usa 'YYYY-MM-DD HH:00'."}

    if ts.tzinfo is None:
        ts = ts.tz_localize("Europe/Madrid", ambiguous=True, nonexistent="shift_forward")

    fila = df[df["dt_madrid"] == ts]
    if fila.empty:
        return {"error": f"No hay datos disponibles para '{fecha_hora}'."}

    bundle = _get_modelo("clf", RUTA_CLF)
    X = fila[bundle["features"]]
    clase = int(bundle["modelo"].predict(X)[0])
    prob = float(bundle["modelo"].predict_proba(X)[0][1]) if hasattr(bundle["modelo"], "predict_proba") else None

    return {
        "fecha_hora": str(fila["datetime"].iloc[0]),
        "modelo": bundle["nombre"],
        "es_hora_pico_critica": bool(clase),
        "probabilidad_pico": round(prob, 4) if prob is not None else None,
        "umbral_critico_MW": round(float(bundle["umbral_pico_MW"]), 2),
        "demanda_real_MW": round(float(fila["demanda_real"].iloc[0]), 2),
    }


def consultar_normativa(consulta: str, top_k: int = TOP_K) -> dict:
    """Busca información normativa sobre periodos horarios, tarifas y operación de REE en el RAG."""
    hits = buscar_normativa(consulta, top_k=top_k)
    return {
        "consulta": consulta,
        "resultados": [
            {"fuente": h["fuente"], "score": h["score"], "fragmento": h["texto"]}
            for h in hits
        ],
    }


CATALOGO_HERRAMIENTAS = {
    "consultar_historico": consultar_historico,
    "predecir_demanda": predecir_demanda,
    "clasificar_hora_pico": clasificar_hora_pico,
    "consultar_normativa": consultar_normativa,
}
