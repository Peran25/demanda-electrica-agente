"""Modelos baseline temporales (sin entrenamiento) para regresión y clasificación.

Uso (desde la raíz del proyecto):
    python -m src.ml.baselines
"""
import json
import pandas as pd

from eval.metricas_ml import evaluar_clasificacion, evaluar_regresion
from src.config import DATA_PROCESSED, PERCENTIL_PICO, ROOT

RUTA_FEATURES = DATA_PROCESSED / "dataset_features.csv"
RUTA_RESULTADOS = ROOT / "eval" / "resultados" / "baselines.json"

# Columnas que no son features de entrada (objetivos o identificadores temporales)
COLS_NO_FEATURES = {"datetime", "dt_utc", "fecha", "demanda_real"}


def cargar_particion_temporal():
    """Divide dataset_features.csv en Train (2021-2023) y Test (2024)."""
    df = pd.read_csv(RUTA_FEATURES)
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True)
    df = df.sort_values("dt_utc").reset_index(drop=True)

    anio = df["dt_utc"].dt.tz_convert("Europe/Madrid").dt.year
    train = df[anio < 2024].copy()
    test = df[anio == 2024].copy()

    # Umbral de hora pico crítica calculado SOLO con datos de Train (evita data leakage)
    umbral_pico = float(train["demanda_real"].quantile(PERCENTIL_PICO))
    train["es_pico"] = (train["demanda_real"] >= umbral_pico).astype(int)
    test["es_pico"] = (test["demanda_real"] >= umbral_pico).astype(int)

    feature_cols = [c for c in df.columns if c not in COLS_NO_FEATURES]
    return train, test, feature_cols, umbral_pico


def ejecutar_baselines() -> dict:
    train, test, _, umbral_pico = cargar_particion_temporal()
    y_test_reg = test["demanda_real"]
    y_test_clf = test["es_pico"]

    resultados = {
        "particion": {"train_filas": len(train), "test_filas": len(test)},
        "umbral_pico_MW": round(umbral_pico, 2),
        "regresion": {
            "persistencia_24h": evaluar_regresion(y_test_reg, test["demanda_lag_24h"]),
            "persistencia_168h": evaluar_regresion(y_test_reg, test["demanda_lag_168h"]),
            "media_movil_24h": evaluar_regresion(y_test_reg, test["demanda_media_24h"]),
        },
        "clasificacion": {
            "siempre_no": evaluar_clasificacion(
                y_test_clf, pd.Series(0, index=y_test_clf.index, dtype=int)
            ),
            "persistencia_24h": evaluar_clasificacion(
                y_test_clf, (test["demanda_lag_24h"] >= umbral_pico).astype(int)
            ),
            "persistencia_168h": evaluar_clasificacion(
                y_test_clf, (test["demanda_lag_168h"] >= umbral_pico).astype(int)
            ),
        },
    }

    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADOS.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultados, indent=2, ensure_ascii=False))
    return resultados


if __name__ == "__main__":
    ejecutar_baselines()