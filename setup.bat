@echo off

echo === RevRate Setup ===

echo Tworzenie venv...
if exist ".venv\" (
    echo .venv juz istnieje.
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo BLAD: Nie udalo sie utworzyc venv
        exit /b 1
    )
)

echo Aktywacja venv...
call .venv\Scripts\activate.bat
set "PATH=.venv\Scripts;%PATH%"

echo Instalacja zaleznosci...
pip install -r requirements.txt
if errorlevel 1 (
    echo BLAD: Instalacja nieudana
    exit /b 1
)

echo Pobieranie datasetu...
python ./backend/download_dataset.py
if errorlevel 1 (
    echo UWAGA: Nie udalo sie pobrac datasetu.
) else (
    echo Dataset pobrany.
)

echo === Gotowe ===
echo Aby aktywowac venv: .venv\Scripts\activate
pause
