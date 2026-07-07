@echo off
REM ============================================================================
REM  GaussianSplat_videoTools green-workstation bootstrap (batch fallback)
REM ============================================================================
REM
REM  Companion to bootstrap.ps1. Use this only if PowerShell script execution
REM  is blocked by policy on the target machine. The PowerShell version is
REM  preferred - it has verification, idempotency, and better error handling.
REM
REM  This batch file does the minimum: installs Git and uv via winget.
REM  All heavy lifting still requires PowerShell after this completes.
REM
REM  Requires: Windows 10 1809+ or Windows 11, winget available.
REM  Pure ASCII source. No special characters anywhere.
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo GaussianSplat_videoTools green-workstation bootstrap (batch fallback)
echo Installs Git and uv, the two prerequisites needed before setup.ps1.
echo.

REM ----------------------------------------------------------------------------
REM  Step 1: Verify winget is available
REM ----------------------------------------------------------------------------
echo ==^> Checking for winget...
where winget >nul 2>&1
if errorlevel 1 (
    echo     [FAIL] winget not found.
    echo.
    echo     winget is preinstalled on Windows 11 and modern Windows 10.
    echo     If missing, install App Installer from the Microsoft Store:
    echo         https://apps.microsoft.com/store/detail/9NBLGGH4NNS1
    echo.
    exit /b 1
)
echo     [OK] winget available.
echo.

REM ----------------------------------------------------------------------------
REM  Step 2: Install Git
REM ----------------------------------------------------------------------------
echo ==^> Installing Git for Windows...
where git >nul 2>&1
if errorlevel 1 (
    echo     Installing via winget ^(this takes a couple of minutes^)...
    winget install --id Git.Git --exact --silent --accept-source-agreements --accept-package-agreements
    echo     Note: PATH changes require closing and reopening this shell.
) else (
    echo     [OK] Git already installed.
    git --version
)
echo.

REM ----------------------------------------------------------------------------
REM  Step 3: Install uv via winget
REM ----------------------------------------------------------------------------
echo ==^> Installing uv...
where uv >nul 2>&1
if errorlevel 1 (
    echo     Installing via winget...
    winget install --id astral-sh.uv --exact --silent --accept-source-agreements --accept-package-agreements
    echo     Note: PATH changes require closing and reopening this shell.
) else (
    echo     [OK] uv already installed.
    uv --version
)
echo.

REM ----------------------------------------------------------------------------
REM  Summary
REM ----------------------------------------------------------------------------
echo ==^> Bootstrap complete.
echo.
echo   Next steps ^(in a NEW PowerShell or CMD window^):
echo     1. git clone https://github.com/YOUR_USERNAME/GaussianSplat_videoTools.git
echo     2. cd GaussianSplat_videoTools
echo     3. .\setup.ps1
echo.
echo   If setup.ps1 execution is blocked by policy, launch PowerShell with:
echo     powershell -ExecutionPolicy Bypass -File .\setup.ps1
echo.

endlocal
