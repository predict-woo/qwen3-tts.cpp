@echo off
rem download_model.bat - fetch the Qwen3-TTS 0.6B Q8_0 GGUF files from HuggingFace.
rem
rem Usage: download_model.bat [dest_dir]
rem   dest_dir defaults to .\models
rem
rem Downloads two files into dest_dir (both are required by the CLI):
rem   - qwen3-tts-0.6b-q8-0.gguf            (~986 MB, TTS model weights)
rem   - qwen3-tts-tokenizer-0.6b-q8-0.gguf  (~250 MB, tokenizer + vocoder)
rem
rem Filenames are kept verbatim from HuggingFace; pass them explicitly to the
rem CLI via --tts-model / --tokenizer-model (auto-detection uses different
rem filenames).

setlocal enabledelayedexpansion

set "DEST_DIR=%~1"
if "%DEST_DIR%"=="" set "DEST_DIR=.\models"

set "BASE_URL=https://huggingface.co/OpenVoiceOS/qwen3-tts-0.6b-q8-0/resolve/main"
set "FILE1=qwen3-tts-0.6b-q8-0.gguf"
set "SHA1=11486718b76d5d7fa1d1c6b03c3afac03f7736c53c3acc3d2903db6925f9ca71"
set "FILE2=qwen3-tts-tokenizer-0.6b-q8-0.gguf"
set "SHA2=f03d490a06d038212d99464ae9a6c2fc3327e34c8a4b840d9e8e6ca67bccaf72"

if not exist "%DEST_DIR%" (
    echo [mkdir] %DEST_DIR%
    mkdir "%DEST_DIR%"
    if errorlevel 1 (
        echo error: failed to create directory "%DEST_DIR%" 1>&2
        endlocal
        exit /b 1
    )
)

call :download_one "%BASE_URL%/%FILE1%" "%DEST_DIR%\%FILE1%" "%SHA1%"
if errorlevel 1 goto :fail

call :download_one "%BASE_URL%/%FILE2%" "%DEST_DIR%\%FILE2%" "%SHA2%"
if errorlevel 1 goto :fail

echo.
echo done. files in %DEST_DIR%:
echo   %DEST_DIR%\%FILE1%
echo   %DEST_DIR%\%FILE2%

endlocal
exit /b 0

:fail
echo.
echo error: download failed 1>&2
rem Pause so users who double-clicked from Explorer can read the error before
rem the cmd window closes. Skipped when CI=1 is set or stdin is not a tty.
if not defined CI pause
endlocal
exit /b 1

rem ---------------------------------------------------------------------------
rem :download_one URL OUT EXPECTED_SHA256
rem   Downloads URL to OUT via PowerShell's Invoke-WebRequest, then verifies
rem   sha256 against EXPECTED_SHA256. Skips download if OUT already exists but
rem   still verifies the checksum. Writes to OUT.part, renames on success,
rem   removes OUT.part on failure so re-runs don't silently skip a corrupt result.
rem ---------------------------------------------------------------------------
:download_one
set "URL=%~1"
set "OUT=%~2"
set "EXPECTED_SHA=%~3"

if exist "%OUT%" (
    echo [skip] %OUT% already exists
    call :verify_sha "%OUT%" "%EXPECTED_SHA%"
    exit /b %errorlevel%
)

echo [download] %URL%
echo        --^> %OUT%

rem Clean any stale .part from a previous failed run.
if exist "%OUT%.part" del /f /q "%OUT%.part"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='Continue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%URL%' -OutFile '%OUT%.part' -UseBasicParsing } catch { Write-Error $_; exit 1 }"
if errorlevel 1 (
    echo error: download failed for %URL% 1>&2
    if exist "%OUT%.part" del /f /q "%OUT%.part"
    exit /b 1
)

if not exist "%OUT%.part" (
    echo error: expected output file "%OUT%.part" not found after download 1>&2
    exit /b 1
)

ren "%OUT%.part" "%~nx2"
if errorlevel 1 (
    echo error: failed to rename "%OUT%.part" to "%OUT%" 1>&2
    if exist "%OUT%.part" del /f /q "%OUT%.part"
    exit /b 1
)

echo [ok] %OUT%

call :verify_sha "%OUT%" "%EXPECTED_SHA%"
if errorlevel 1 (
    echo error: checksum mismatch, removing corrupt download 1>&2
    del /f /q "%OUT%"
    exit /b 1
)
exit /b 0

rem ---------------------------------------------------------------------------
rem :verify_sha FILE EXPECTED_SHA256
rem   Computes sha256 via PowerShell Get-FileHash and compares to EXPECTED_SHA256.
rem ---------------------------------------------------------------------------
:verify_sha
set "V_FILE=%~1"
set "V_EXPECTED=%~2"
for /f "delims=" %%H in ('powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath '%V_FILE%').Hash.ToLower()"') do set "V_ACTUAL=%%H"
if /i "!V_ACTUAL!"=="!V_EXPECTED!" (
    echo [verified] sha256 ok ^(!V_FILE!^)
    exit /b 0
) else (
    echo [error] checksum mismatch for !V_FILE! 1>&2
    echo         expected: !V_EXPECTED! 1>&2
    echo         actual:   !V_ACTUAL! 1>&2
    exit /b 1
)
