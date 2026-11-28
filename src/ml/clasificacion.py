"""Clasificación de hora pico crítica: logística, árbol, class_weight/umbral."""
"""Entrenamiento y evaluación de clasificación de hora pico crítica (percentil 95).

Uso (desde la raíz del proyecto):
    python -m src.ml.clasificacion
"""
import json
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval.metricas_ml import evaluar_clasificacion
from src.config import MODELS_DIR, PERCENTIL_PICO, ROOT, SEED
from src.ml.baselines import cargar_particion_temporal

RUTA_MODELO = MODELS_DIR / "modelo_clasificacion.joblib"
RUTA_RESULTADOS = ROOT / "eval" / "resultados" / "clasificacion.json"


def entrenar_clasificacion() -> dict:
    train, test, feature_cols, umbral_pico = cargar_particion_temporal()
    X_train, y_train = train[feature_cols], train["es_pico"]
    X_test, y_test = test[feature_cols], test["es_pico"]

    candidatos = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=SEED,
        ),
    }

    resultados = {}
    mejor_nombre, mejor_modelo, mejor_f1 = None, None, -1.0

    for nombre, modelo in candidatos.items():
        print(f"Entrenando clasificación: {nombre} ...")
        modelo.fit(X_train, y_train)
        preds = modelo.predict(X_test)
        metricas = evaluar_clasificacion(y_test, preds)
        resultados[nombre] = metricas
        print(f"  -> {metricas}")

        if metricas["f1"] > mejor_f1:
            mejor_f1 = metricas["f1"]
            mejor_nombre = nombre
            mejor_modelo = modelo

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "nombre": mejor_nombre,
            "modelo": mejor_modelo,
            "features": feature_cols,
            "percentil": PERCENTIL_PICO,
            "umbral_pico_MW": umbral_pico,
            "metricas_test": resultados[mejor_nombre],
        },
        RUTA_MODELO,
    )

    salida = {
        "mejor_modelo": mejor_nombre,
        "percentil": PERCENTIL_PICO,
        "umbral_pico_MW": round(umbral_pico, 2),
        "modelos": resultados,
    }
    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADOS.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMejor clasificador guardado en {RUTA_MODELO}: {mejor_nombre} (F1={mejor_f1})")
    return salida


if __name__ == "__main__":
    entrenar_clasificacion()