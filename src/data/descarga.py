"""Descarga de datos a data/raw/ (con caché, por tramos y con metadatos).

Uso (desde la raíz del proyecto):
    python -m src.data.descarga --fuente prueba          # 1 día de REData, para validar
    python -m src.data.descarga --fuente redata          # demanda REData (tramos de 15 días + media horaria)
    python -m src.data.descarga --fuente openmeteo       # temperatura horaria por ciudad
    python -m src.data.descarga --fuente festivos        # festivos nacionales
    python -m src.data.descarga --fuente todo            # las tres anteriores
    python -m src.data.descarga --fuente esios-buscar --texto "demanda real"
    python -m src.data.descarga --fuente esios --indicador <ID>

Reglas de uso responsable (REE): nunca se repite una petición cuyo archivo ya existe
en data/raw/, se descarga por tramos y se hace una pausa entre peticiones.
El agente NO debe llamar nunca a estas APIs: solo lee los CSV locales.
"""
import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    CIUDADES,
    DATA_RAW,
    ESIOS_TOKEN,
    FECHA_FIN,
    FECHA_INICIO,
)

PAUSA_S = 1.0          # pausa entre peticiones
REINTENTOS = 3
TIMEOUT_S = 60

REDATA_URL = "https://apidatos.ree.es/es/datos/demanda/demanda-tiempo-real"
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
NAGER_URL = "https://date.nager.at/api/v3/PublicHolidays/{year}/ES"
ESIOS_BASE = "https://api.esios.ree.es"

DIR_REDATA = DATA_RAW / "redata"
DIR_METEO = DATA_RAW / "openmeteo"
DIR_FESTIVOS = DATA_RAW / "festivos"
DIR_ESIOS = DATA_RAW / "esios"
META_PATH = DATA_RAW / "metadata_descarga.json"


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def _get(url, params=None, headers=None):
    """GET con reintentos y pausa. Lanza excepción si falla tras los reintentos."""
    ultimo_error = None
    for intento in range(1, REINTENTOS + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT_S)
            r.raise_for_status()
            time.sleep(PAUSA_S)
            return r
        except requests.HTTPError as e:
            cuerpo = e.response.text[:300] if e.response is not None else ""
            if e.response is not None and 400 <= e.response.status_code < 500:
                raise RuntimeError(f"Petición rechazada ({e.response.status_code}): "
                                   f"{cuerpo}\nURL: {url} params={params}") from e
            ultimo_error = e
            print(f"  intento {intento}/{REINTENTOS} falló: {e}")
            time.sleep(PAUSA_S * 3 * intento)
        except requests.RequestException as e:
            ultimo_error = e
            print(f"  intento {intento}/{REINTENTOS} falló: {e}")
            time.sleep(PAUSA_S * 3 * intento)
    raise RuntimeError(f"No se pudo descargar {url}: {ultimo_error}")


def _guardar_json(ruta: Path, datos):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")


def _registrar_metadatos(fuente: str, info: dict):
    """Guarda fecha de descarga y parámetros (para el documento de medición)."""
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    info["fecha_descarga"] = datetime.now().isoformat(timespec="seconds")
    meta[fuente] = info
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _meses(inicio: str, fin: str):
    """Genera (primer_dia, ultimo_dia) de cada mes entre dos fechas."""
    d = date.fromisoformat(inicio).replace(day=1)
    fin_d = date.fromisoformat(fin)
    while d <= fin_d:
        siguiente = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
        ultimo = min(siguiente - timedelta(days=1), fin_d)
        yield d, ultimo
        d = siguiente


def _tramos_dias(inicio: str, fin: str, dias: int = 15):
    """Genera tramos de N días para no exceder el límite de puntos de REData."""
    d = date.fromisoformat(inicio)
    fin_d = date.fromisoformat(fin)
    while d <= fin_d:
        ultimo = min(d + timedelta(days=dias - 1), fin_d)
        yield d, ultimo
        d = ultimo + timedelta(days=1)


# ----------------------------------------------------------------------------
# REData: demanda en tiempo real agregada a nivel horario (sin token)
# ----------------------------------------------------------------------------
def _redata_tramo(ini: date, fin: date, trunc: str = "hour") -> dict:
    params = {
        "start_date": f"{ini.isoformat()}T00:00",
        "end_date": f"{fin.isoformat()}T23:59",
        "time_trunc": trunc,
        "geo_limit": "peninsular",
    }
    return _get(REDATA_URL, params=params).json()


def descargar_prueba():
    """Una sola petición de 1 día: valida la conexión y muestra la estructura."""
    ini = date.fromisoformat(FECHA_INICIO)
    datos = None
    for trunc in ("hour", "day"):
        try:
            datos = _redata_tramo(ini, ini, trunc)
            print(f"REData acepta time_trunc={trunc} en demanda/demanda-tiempo-real.")
            break
        except RuntimeError as e:
            print(f"REData NO acepta time_trunc={trunc}:\n{e}\n")
    if datos is None:
        raise SystemExit("REData rechazó ambas granularidades.")
    _guardar_json(DATA_RAW / "prueba_redata.json", datos)
    print("Claves de primer nivel:", list(datos.keys()))
    for item in datos.get("included", []):
        attrs = item.get("attributes", {})
        vals = attrs.get("values", [])
        print(f"- {attrs.get('title') or item.get('type')}: {len(vals)} valores")
        if vals:
            print("  ejemplo:", vals[0])
    print("\nRevisa que 'Demanda real' aparezca (~288 valores cada 5 min). "
          "Archivo guardado en data/raw/prueba_redata.json")


def descargar_redata():
    DIR_REDATA.mkdir(parents=True, exist_ok=True)
    tramos = list(_tramos_dias(FECHA_INICIO, FECHA_FIN, dias=15))
    nuevos = 0
    for ini, fin in tramos:
        ruta = DIR_REDATA / f"demanda_{ini:%Y-%m-%d}_{fin:%Y-%m-%d}.json"
        if ruta.exists():
            continue  # caché: no se repite lo ya descargado
        print(f"REData {ini:%Y-%m-%d} -> {fin:%Y-%m-%d} ...")
        _guardar_json(ruta, _redata_tramo(ini, fin))
        nuevos += 1
    print(f"REData: {nuevos} tramos nuevos, {len(tramos) - nuevos} ya en caché.")
    consolidar_redata()
    _registrar_metadatos("redata", {
        "url": REDATA_URL, "geo_limit": "peninsular", "time_trunc": "hour (agregado)",
        "inicio": FECHA_INICIO, "fin": FECHA_FIN,
    })


def consolidar_redata():
    """Une los JSON de REData y los agrega por hora en data/raw/demanda_horaria.csv."""
    filas = []
    for ruta in sorted(DIR_REDATA.glob("demanda_*.json")):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        for item in datos.get("included", []):
            attrs = item.get("attributes", {})
            nombre = (attrs.get("title") or item.get("type") or "").lower()
            if "real" in nombre:
                for v in attrs.get("values", []):
                    filas.append({"datetime": v["datetime"], "demanda_real": v["value"]})
    if not filas:
        print("AVISO: no se encontró la serie 'Demanda real'. Ejecuta --fuente prueba "
              "y revisa los nombres de las series.")
        return
    df = pd.DataFrame(filas)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
    df["datetime"] = df["datetime"].dt.floor("h")
    df = (
        df.groupby("datetime", as_index=False)["demanda_real"]
        .mean()
        .sort_values("datetime")
    )
    salida = DATA_RAW / "demanda_horaria.csv"
    df.to_csv(salida, index=False)
    print(f"Guardado {salida} con {len(df)} filas "
          f"({df['datetime'].min()} -> {df['datetime'].max()}).")


# ----------------------------------------------------------------------------
# Open-Meteo: temperatura horaria por ciudad (sin clave)
# ----------------------------------------------------------------------------
def descargar_openmeteo():
    DIR_METEO.mkdir(parents=True, exist_ok=True)
    for ciudad, (lat, lon) in CIUDADES.items():
        ruta = DIR_METEO / f"temperatura_{ciudad}.csv"
        if ruta.exists():
            print(f"Open-Meteo {ciudad}: ya en caché.")
            continue
        print(f"Open-Meteo {ciudad} ...")
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": FECHA_INICIO, "end_date": FECHA_FIN,
            "hourly": "temperature_2m", "timezone": "Europe/Madrid",
        }
        h = _get(OPENMETEO_URL, params=params).json()["hourly"]
        pd.DataFrame({"datetime": h["time"], f"temp_{ciudad}": h["temperature_2m"]}) \
            .to_csv(ruta, index=False)
    _registrar_metadatos("openmeteo", {
        "url": OPENMETEO_URL, "variable": "temperature_2m (observación/reanálisis)",
        "ciudades": CIUDADES, "inicio": FECHA_INICIO, "fin": FECHA_FIN,
    })


# ----------------------------------------------------------------------------
# Festivos nacionales (Nager.Date, sin clave)
# ----------------------------------------------------------------------------
def descargar_festivos():
    DIR_FESTIVOS.mkdir(parents=True, exist_ok=True)
    anios = range(date.fromisoformat(FECHA_INICIO).year, date.fromisoformat(FECHA_FIN).year + 1)
    filas = []
    for anio in anios:
        ruta = DIR_FESTIVOS / f"festivos_{anio}.json"
        if not ruta.exists():
            print(f"Festivos {anio} ...")
            _guardar_json(ruta, _get(NAGER_URL.format(year=anio)).json())
        for f in json.loads(ruta.read_text(encoding="utf-8")):
            if f.get("global"):  # solo festivos de ámbito nacional
                filas.append({"fecha": f["date"], "nombre": f["localName"]})
    pd.DataFrame(filas).drop_duplicates("fecha").sort_values("fecha") \
        .to_csv(DATA_RAW / "festivos_nacionales.csv", index=False)
    print(f"Guardado data/raw/festivos_nacionales.csv ({len(filas)} festivos).")
    _registrar_metadatos("festivos", {"url": NAGER_URL, "ambito": "nacional"})


# ----------------------------------------------------------------------------
# ESIOS (opcional, con token): para contrastar la demanda de REData
# ----------------------------------------------------------------------------
def _headers_esios():
    if not ESIOS_TOKEN:
        raise SystemExit("Falta ESIOS_TOKEN en el archivo .env")
    return {
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "Content-Type": "application/json",
        "x-api-key": ESIOS_TOKEN,
    }


def esios_buscar(texto: str):
    """Busca indicadores por texto en el catálogo. Úsalo ANTES de pedir datos,
    para no consultar indicadores inexistentes."""
    r = _get(f"{ESIOS_BASE}/indicators", params={"text": texto}, headers=_headers_esios())
    for ind in r.json().get("indicators", []):
        print(f"{ind.get('id'):>6}  {ind.get('name')}")


def descargar_esios(indicador: int):
    DIR_ESIOS.mkdir(parents=True, exist_ok=True)
    for ini, fin in _meses(FECHA_INICIO, FECHA_FIN):
        ruta = DIR_ESIOS / f"indicador_{indicador}_{ini:%Y-%m}.json"
        if ruta.exists():
            continue
        print(f"ESIOS {indicador} {ini:%Y-%m} ...")
        params = {
            "start_date": f"{ini.isoformat()}T00:00:00+01:00",
            "end_date": f"{fin.isoformat()}T23:59:59+01:00",
            "time_trunc": "hour",
            "time_agg": "average",
        }
        r = _get(f"{ESIOS_BASE}/indicators/{indicador}", params=params, headers=_headers_esios())
        _guardar_json(ruta, r.json())
    consolidar_esios(indicador)
    _registrar_metadatos("esios", {"indicador": indicador, "inicio": FECHA_INICIO, "fin": FECHA_FIN})


def consolidar_esios(indicador: int):
    """Une los JSON mensuales de ESIOS en data/raw/demanda_horaria.csv."""
    filas = []
    for ruta in sorted(DIR_ESIOS.glob(f"indicador_{indicador}_*.json")):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        filas.extend(datos.get("indicator", {}).get("values", []))
    if not filas:
        print("AVISO: no hay valores en los JSON de ESIOS.")
        return
    df = pd.DataFrame(filas)
    # Si el indicador viene desglosado por zona, conservar la península (geo_id 8741)
    if "geo_id" in df.columns and 8741 in set(df["geo_id"]):
        df = df[df["geo_id"] == 8741]
    df["datetime"] = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_convert("Europe/Madrid")
    df = df.rename(columns={"value": "demanda_real"})[["datetime", "demanda_real"]]
    df = df.drop_duplicates("datetime").sort_values("datetime")
    df.to_csv(DATA_RAW / "demanda_horaria.csv", index=False)
    print(f"Guardado data/raw/demanda_horaria.csv con {len(df)} filas "
          f"({df['datetime'].min()} -> {df['datetime'].max()}).")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fuente", required=True,
                   choices=["prueba", "redata", "openmeteo", "festivos", "todo",
                            "esios-buscar", "esios"])
    p.add_argument("--texto", help="texto a buscar en el catálogo de ESIOS")
    p.add_argument("--indicador", type=int, help="ID del indicador de ESIOS")
    a = p.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if a.fuente == "prueba":
        descargar_prueba()
    elif a.fuente == "redata":
        descargar_redata()
    elif a.fuente == "openmeteo":
        descargar_openmeteo()
    elif a.fuente == "festivos":
        descargar_festivos()
    elif a.fuente == "todo":
        descargar_redata()
        descargar_openmeteo()
        descargar_festivos()
    elif a.fuente == "esios-buscar":
        esios_buscar(a.texto or "demanda")
    elif a.fuente == "esios":
        if not a.indicador:
            raise SystemExit("Indica --indicador <ID> (búscalo antes con esios-buscar)")
        descargar_esios(a.indicador)


if __name__ == "__main__":
    main()