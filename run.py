"""Entry point for running the Car Price Prediction application."""

# pylint: disable=import-error
import os
import subprocess  # noqa: C0411
import sys
import time
import webbrowser  # noqa: C0411
from pathlib import Path  # noqa: C0411
from multiprocessing import Process  # noqa: C0411

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT)


def run_api():
    """Start the prediction API server on port 8000."""
    import uvicorn  # pylint: disable=import-outside-toplevel

    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000)


def run_middleware():
    """Start the middleware server on port 8001."""
    import uvicorn  # pylint: disable=import-outside-toplevel

    uvicorn.run("backend.middleware.main:app", host="0.0.0.0", port=8001)


def run_streamlit():
    """Start the Streamlit frontend on port 8501."""
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(ROOT / "frontend" / "web_app.py"),
            "--server.port", "8501",
            "--server.headless", "true",
        ],
        check=True,
    )


def generate_options():
    """Generate car_options.json if it does not already exist."""
    options_path = ROOT / "frontend" / "car_options.json"
    if not options_path.exists():
        print("Generowanie car_options.json...")
        subprocess.run(
            [sys.executable, str(ROOT / "backend" / "tools" / "generate_car_options.py")],
            check=True,
        )
    else:
        print("car_options.json juz istnieje, pomijam generowanie.")


def check_models():
    """Verify that the required model files exist."""
    models_dir = ROOT / "models"
    required = ["custom_model.pkl"]
    missing = [f for f in required if not (models_dir / f).exists()]
    if missing:
        print(f"Blad: Brak plikow modelu w {models_dir}:")
        for f in missing:
            print(f"  - {f}")
        print("Przeprowadz trening w notebooku przed uruchomieniem.")
        sys.exit(1)


if __name__ == "__main__":
    check_models()
    generate_options()

    print("\nUruchamianie API (port 8000), Middleware (port 8001) i Streamlit (port 8501)...")
    print("Frontend: http://localhost:8501\n")

    api_proc = Process(target=run_api, daemon=True)
    mid_proc = Process(target=run_middleware, daemon=True)
    streamlit_proc = Process(target=run_streamlit, daemon=True)

    api_proc.start()
    mid_proc.start()
    streamlit_proc.start()

    time.sleep(3)
    webbrowser.open("http://localhost:8501")

    try:
        streamlit_proc.join()
    except KeyboardInterrupt:
        print("\nZamykanie serwerow...")
        api_proc.terminate()
        mid_proc.terminate()
        streamlit_proc.terminate()
        api_proc.join()
        mid_proc.join()
        streamlit_proc.join()
        print("Zamknieto.")
