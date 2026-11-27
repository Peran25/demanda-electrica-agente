"""Tests unitarios y de integración para datos, ML, herramientas y validador."""
import pandas as pd
import pytest

from eval.metricas_ml import evaluar_clasificacion, evaluar_regresion
from src.agent.herramientas import (
    clasificar_hora_pico,
    consultar_historico,
    predecir_demanda,
)
from src.agent.validador import validar_respuesta_final, validar_salida_llm
from src.config import DATA_PROCESSED, MODELS_DIR


# ----------------------------------------------------------------------------
# 1. Integridad de datasets procesados
# ----------------------------------------------------------------------------
def test_datasets_procesados_existen_y_sin_nulos():
    ruta_limpia = DATA_PROCESSED / "demanda_limpia.csv"
    ruta_features = DATA_PROCESSED / "dataset_features.csv"

    assert ruta_limpia.exists(), "Falta demanda_limpia.csv"
    assert ruta_features.exists(), "Falta dataset_features.csv"

    df_limpia = pd.read_csv(ruta_limpia)
    df_feat = pd.read_csv(ruta_features)

    assert len(df_limpia) > 35000
    assert df_limpia.isna().sum().sum() == 0
    assert df_feat.isna().sum().sum() == 0
    assert {"demanda_real", "temp_media", "es_festivo"}.issubset(df_limpia.columns)
    assert {"demanda_lag_24h", "demanda_lag_168h", "hdd_18", "cdd_22"}.issubset(df_feat.columns)


# ----------------------------------------------------------------------------
# 2. Métricas y artefactos ML
# ----------------------------------------------------------------------------
def test_metricas_ml():
    reg = evaluar_regresion([100.0, 200.0], [110.0, 190.0])
    assert reg["MSE"] == 100.0
    assert reg["RMSE"] == 10.0
    assert reg["R2"] == 0.96

    clf = evaluar_clasificacion([0, 1, 1, 0], [0, 1, 0, 0])
    assert clf["accuracy"] == 0.75
    assert clf["precision"] == 1.0
    assert clf["recall"] == 0.5
    assert clf["matriz_confusion"] == [[2, 0], [1, 1]]


def test_modelos_serializados_existen():
    assert (MODELS_DIR / "modelo_regresion.joblib").exists()
    assert (MODELS_DIR / "modelo_clasificacion.joblib").exists()


# ----------------------------------------------------------------------------
# 3. Herramientas locales del agente
# ----------------------------------------------------------------------------
def test_herramientas_locales():
    hist = consultar_historico(anio=2024, mes=1)
    assert "error" not in hist
    assert hist["registros_horas"] == 744
    assert hist["demanda_max_MW"] > hist["demanda_media_MW"] > hist["demanda_min_MW"]

    pred = predecir_demanda("2024-07-15 14:00")
    assert "error" not in pred
    assert pred["demanda_predicha_MW"] > 0

    clf = clasificar_hora_pico("2024-01-11 20:00")
    assert "error" not in clf
    assert isinstance(clf["es_hora_pico_critica"], bool)
    assert 0.0 <= clf["probabilidad_pico"] <= 1.0


def test_herramienta_fuera_de_rango():
    hist = consultar_historico(fecha="2026-03-01")
    assert "error" in hist


# ----------------------------------------------------------------------------
# 4. Validador JSON y respuestas del agente
# ----------------------------------------------------------------------------
def test_validador_salida_llm():
    # JSON envuelto en bloque markdown
    crudo_tool = '```json\n{"accion": "consultar_historico", "argumentos": {"anio": 2024}}\n```'
    res = validar_salida_llm(crudo_tool)
    assert res.es_valida is True
    assert res.accion == "consultar_historico"
    assert res.argumentos == {"anio": 2024}

    # Herramienta inexistente
    res_bad = validar_salida_llm('{"accion": "inventada", "argumentos": {}}')
    assert res_bad.es_valida is False

    # Respuesta final válida
    res_fin = validar_salida_llm('{"accion": "responder", "respuesta": "La demanda media fue de 338426 MW."}')
    assert res_fin.es_valida is True
    ok, _ = validar_respuesta_final(res_fin.respuesta, ["consultar_historico"])
    assert ok is True