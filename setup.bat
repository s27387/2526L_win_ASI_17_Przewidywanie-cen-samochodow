@echo off
setlocal

set "VENV_DIR=venv"

if not exist "%VENV_DIR%" (
    echo Tworzenie wirtualnego srodowiska...
    py -m venv "%VENV_DIR%"
)

echo Instalowanie zaleznosci...
call "%VENV_DIR%\Scripts\activate.bat"
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

echo.
echo Setup zakonczony. Aby aktywowac srodowisko, uruchom:
echo     %VENV_DIR%\Scripts\activate
echo.

endlocal
