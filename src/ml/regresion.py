"""Regresión de demanda: lineal, polinómica, árbol, MLP (split cronológico)."""
"""Entrenamiento y evaluación de modelos de regresión para demanda horaria (MW).

Uso (desde la raíz del proyecto):
    python -m src.ml.regresion
"""
import json
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval.metricas_ml import evaluar_regresion
from src.config import MODELS_DIR, ROOT, SEED
from src.ml.baselines import cargar_particion_temporal

RUTA_MODELO = MODELS_DIR / "modelo_regresion.joblib"
RUTA_RESULTADOS = ROOT / "eval" / "resultados" / "regresion.json"


def entrenar_regresion() -> dict:
    train, test, feature_cols, _ = cargar_particion_temporal()
    X_train, y_train = train[feature_cols], train["demanda_real"]
    X_test, y_test = test[feature_cols], test["demanda_real"]

    candidatos = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=SEED)),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=SEED
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=SEED
        ),
    }

    resultados = {}
    mejor_nombre, mejor_modelo, mejor_rmse = None, None, float("inf")

    for nombre, modelo in candidatos.items():
        print(f"Entrenando regresión: {nombre} ...")
        modelo.fit(X_train, y_train)
        preds = modelo.predict(X_test)
        metricas = evaluar_regresion(y_test, preds)
        resultados[nombre] = metricas
        print(f"  -> {metricas}")

        if metricas["RMSE"] < mejor_rmse:
            mejor_rmse = metricas["RMSE"]
            mejor_nombre = nombre
            mejor_modelo = modelo

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "nombre": mejor_nombre,
            "modelo": mejor_modelo,
            "features": feature_cols,
            "metricas_test": resultados[mejor_nombre],
        },
        RUTA_MODELO,
    )

    salida = {"mejor_modelo": mejor_nombre, "modelos": resultados}
    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADOS.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMejor modelo guardado en {RUTA_MODELO}: {mejor_nombre} (RMSE={mejor_rmse} MW)")
    return salida


if __name__ == "__main__":
    entrenar_regresion()