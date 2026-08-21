@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM --- Hugging Face cache (local weights) ---
if not defined HF_HUB_CACHE set "HF_HUB_CACHE=E:\Backup\huggingface\hub"
if not defined HF_HOME set "HF_HOME=E:\Backup\huggingface"
set "HF_HUB_OFFLINE=0"
set "TRANSFORMERS_OFFLINE=0"
set "HF_DATASETS_OFFLINE=0"

REM Model: HF id or full snapshot path.
REM Uncomment to skip the Hub and load a specific snapshot:
REM set "ZIMAGE_MODEL=E:\Backup\huggingface\hub\models--Tongyi-MAI--Z-Image-Turbo\snapshots\0e36c2b379e66fa531d01cc531c44919e5f1c6fd"
if not defined ZIMAGE_MODEL set "ZIMAGE_MODEL=Tongyi-MAI/Z-Image-Turbo"

REM Optional LoRA directory (scanned for .safetensors / .pt):
REM if not defined ZIMAGE_LORA_DIR set "ZIMAGE_LORA_DIR="

set "ZIMAGE_DEVICE=auto"
set "ZIMAGE_PORT=43127"
set "PYTHONUTF8=1"

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo.
echo Z-Image-Turbo Studio
echo   model : %ZIMAGE_MODEL%
echo   cache : %HF_HUB_CACHE%
echo   url   : http://127.0.0.1:%ZIMAGE_PORT%
echo.

"%PYTHON%" app.py --host 127.0.0.1 --port %ZIMAGE_PORT%
if errorlevel 1 pause
endlocal
