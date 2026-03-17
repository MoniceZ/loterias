# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


project_root = Path(SPECPATH).parent
datas = []

for arquivo in project_root.glob("*_resultados.csv"):
    datas.append((str(arquivo), "."))

for arquivo in (
    project_root / "historico_modelos.csv",
    project_root / "loteria.ico",
):
    if arquivo.exists():
        datas.append((str(arquivo), "."))

modelos_dir = project_root / "modelos"
if modelos_dir.exists():
    datas.append((str(modelos_dir), "modelos"))

hiddenimports = collect_submodules("sklearn")


a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PrevisorLoteriasInteligente",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "loteria.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PrevisorLoteriasInteligente",
)
