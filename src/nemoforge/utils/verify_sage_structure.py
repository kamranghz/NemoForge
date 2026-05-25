"""Quick check of local SAGE-10k layout (zip scenes or folders). Run with any Python."""
from pathlib import Path
import json
import random
import zipfile
import importlib.util as _ilu
_PATHS_PY = Path(__file__).resolve().parent / "paths.py"
_spec = _ilu.spec_from_file_location("nemoforge.utils.paths", _PATHS_PY)
_pm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_pm)  # type: ignore[union-attr]
data_root = _pm.SAGE_10K_ROOT

print(f"Data root exists: {data_root.exists()} ({data_root})")
scenes_dir = data_root / "scenes"
print(f"Scenes dir exists: {scenes_dir.exists()}")

if scenes_dir.exists():
    zips = list(scenes_dir.glob("*.zip"))
    dirs = [d for d in scenes_dir.iterdir() if d.is_dir()]
    print(f"Found {len(zips)} zip archives, {len(dirs)} scene folders")

    if zips:
        z = random.choice(zips)
        print(f"\nSample zip: {z.name}")
        try:
            with zipfile.ZipFile(z, "r") as zf:
                names = zf.namelist()[:15]
                print("First entries:", names)
        except zipfile.BadZipFile as e:
            print(f"Bad zip: {e}")

    if dirs:
        sample_id = random.choice(dirs).name
        sample_dir = scenes_dir / sample_id
        print(f"\nSample folder scene: {sample_id}")
        layout_files = list(sample_dir.glob("*layout*.json"))
        if layout_files:
            print(f"Layout file: {layout_files[0].name}")
            with open(layout_files[0], encoding="utf-8") as f:
                layout = json.load(f)
            print("Layout keys:", list(layout.keys())[:10])
        else:
            print("No *layout*.json in sample folder")
else:
    print("No 'scenes' folder — check your download path!")

# Also check for kits/ if present
kits_dir = data_root / "kits"
print(f"\nKits directory present: {kits_dir.exists()}")
