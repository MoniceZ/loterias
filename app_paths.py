from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path

APP_DIR_NAME = "PrevisorLoteriasInteligente"


def _slugify(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto or "")
    sem_acentos = normalizado.encode("ascii", "ignore").decode("ascii")
    return sem_acentos.lower().strip().replace(" ", "_").replace("+", "mais")


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def bundle_dir() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return _project_root()


def data_dir() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        destino = base / APP_DIR_NAME
        destino.mkdir(parents=True, exist_ok=True)
        return destino
    return _project_root()


def icon_path() -> Path:
    return bundle_dir() / "loteria.ico"


def writable_modelos_dir() -> Path:
    destino = data_dir() / "modelos"
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def bundled_modelos_dir() -> Path:
    return bundle_dir() / "modelos"


def historico_modelos_path() -> Path:
    return data_dir() / "historico_modelos.csv"


def bundled_historico_modelos_path() -> Path:
    return bundle_dir() / "historico_modelos.csv"


def resultados_csv_path(loteria: str) -> Path:
    return data_dir() / f"{_slugify(loteria)}_resultados.csv"


def bundled_resultados_csv_path(loteria: str) -> Path:
    return bundle_dir() / f"{_slugify(loteria)}_resultados.csv"


def resolve_resultados_csv_path(loteria: str) -> Path:
    caminho_usuario = resultados_csv_path(loteria)
    if caminho_usuario.exists():
        return caminho_usuario

    caminho_bundle = bundled_resultados_csv_path(loteria)
    if caminho_bundle.exists():
        return caminho_bundle

    return caminho_usuario
