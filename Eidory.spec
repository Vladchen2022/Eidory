# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


datas = collect_data_files("eidory", includes=["db/*.sql"])
hiddenimports = []
hiddenimports += collect_submodules("transformers.models.metaclip_2")

a = Analysis(
    ["src/eidory/main.py"],
    pathex=["."],
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
    name="Eidory",
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Eidory",
)
app = BUNDLE(
    coll,
    name="Eidory.app",
    icon=None,
    bundle_identifier="local.eidory",
    info_plist={
        "CFBundleName": "Eidory",
        "CFBundleDisplayName": "Eidory",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": "True",
    },
)
