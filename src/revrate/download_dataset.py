import os
import sys
import shutil
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
data_dir = project_root / "data"

kaggle_dir = Path.home() / ".kaggle"
kaggle_json = kaggle_dir / "kaggle.json"

if not kaggle_json.exists():
    local_kaggle = project_root / "kaggle.json"
    if not local_kaggle.exists():
        shutil.copy(project_root / "kaggle.example.json", local_kaggle)
        print("UWAGA: Utworzono kaggle.json z szablonu. Wypelnij go swoimi danymi logowania do Kaggle.")
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(local_kaggle), str(kaggle_json))
    kaggle_json.chmod(0o600)
    print(f"Skopiowano {local_kaggle} -> {kaggle_json}")

try:
    import kaggle.api

    kaggle.api.authenticate()

    data_dir.mkdir(parents=True, exist_ok=True)

    print("Pobieranie datasetu...")
    kaggle.api.dataset_download_files(
        "bartoszpieniak/poland-cars-for-sale-dataset",
        path=str(data_dir),
        unzip=True,
    )
    print(f"Pobrano do {data_dir}/")

    print("\nPliki w data/:")
    for file in os.listdir(str(data_dir)):
        print(f"  - {file}")

except ImportError:
    print("Brak pakietu kaggle. Zainstaluj: pip install kaggle")
    sys.exit(1)
except Exception as e:
    print(f"Blad: {e}")
    print("Upewnij sie ze masz:")
    print("  1. pip install kaggle")
    print("  2. kaggle.json w ~/.kaggle/")
    sys.exit(1)
