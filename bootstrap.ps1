<#
.SYNOPSIS
    Green-workstation bootstrap for GaussianSplat_videoTools.

.DESCRIPTION
    Installs the minimum prerequisites needed to clone and run the pipeline
    on a fresh Windows machine. Specifically:

      - Git for Windows (needed to clone the repo)
      - uv (needed by setup.ps1 to install Python and package the venv)

    That's all. This script does NOT install the pipeline itself; after
    this completes, you clone the repo and run setup.ps1 inside it.

    Design principles:
      - Idempotent. Re-running is safe; already-installed tools are detected
        and skipped with a notice.
      - Every step announces before running. No silent long operations.
      - Verifies each install actually worked before continuing.
      - Uses winget where available (Windows 10 1809+ has winget preinstalled
        or trivially available). Falls back to manual instructions if not.
      - Pure ASCII. No em-dashes, en-dashes, or curly quotes anywhere.
        PowerShell on Windows Server sometimes reads scripts as Windows-1252;
        multi-byte UTF-8 causes hard-to-diagnose parse errors.

    After this completes, the workflow is:
        git clone https://github.com/YOUR_USERNAME/GaussianSplat_videoTools.git
        cd GaussianSplat_videoTools
        .\setup.ps1

.PARAMETER SkipGit
    Skip installing Git. Use if Git is provided by another tool
    (GitHub Desktop, Visual Studio) and you know it's on PATH.

.PARAMETER SkipUv
    Skip installing uv. Use if uv is already present at a specific version.

.EXAMPLE
    .\bootstrap.ps1
    Full green-machine bootstrap.

.EXAMPLE
    .\bootstrap.ps1 -SkipGit
    You already have Git from another install; just add uv.

.NOTES
    Author: MAKE pipeline
    Requires: Windows 10 1809+ or Windows 11, PowerShell 5.1+.
    Admin rights: not required (winget and uv install to per-user locations).
#>

[CmdletBinding()]
param(
    [switch]$SkipGit,
    [switch]$SkipUv
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest


# =============================================================================
# HELPERS
# =============================================================================

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Info {
    param([string]$Message)
    Write-Host "    $Message" -ForegroundColor Gray
}

function Write-OK {
    param([string]$Message)
    Write-Host "    [OK] $Message" -ForegroundColor Green
}

function Write-WarnMsg {
    param([string]$Message)
    Write-Host "    [WARN] $Message" -ForegroundColor Yellow
}

function Write-FailMsg {
    param([string]$Message)
    Write-Host "    [FAIL] $Message" -ForegroundColor Red
}

function Test-Command {
    <#
        Returns true if a command is available on PATH in the CURRENT shell.
        After installing something new, PATH may not have refreshed yet;
        we handle that with Update-PathFromRegistry below.
    #>
    param([string]$Name)
    $null = Get-Command $Name -ErrorAction SilentlyContinue
    return $?
}

function Update-PathFromRegistry {
    <#
        winget and uv both modify the User PATH environment variable during
        install, but the current PowerShell session's $env:PATH doesn't
        auto-refresh. This function reads both User and Machine PATH from
        the registry and updates the current session so freshly-installed
        tools are immediately available without restarting the shell.
    #>
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath    = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"
}


# =============================================================================
# PRE-FLIGHT
# =============================================================================

Write-Host ""
Write-Host "GaussianSplat_videoTools green-workstation bootstrap" -ForegroundColor Cyan
Write-Host "This installs Git and uv, the two tools needed before setup.ps1."
Write-Host ""

Write-Step "Pre-flight checks"

# PowerShell version. 5.1 is the minimum modern baseline; earlier versions
# lack some cmdlets we use. Not a hard block but worth flagging.
$psVersion = $PSVersionTable.PSVersion
Write-Info "PowerShell version: $psVersion"
if ($psVersion.Major -lt 5) {
    Write-FailMsg "PowerShell 5.1 or newer is required. This machine has $psVersion."
    Write-Host ""
    Write-Host "Install PowerShell 7 from: https://aka.ms/PSWindows" -ForegroundColor Yellow
    exit 1
}

# Windows version. Anything from Windows 10 1809 onward has winget-friendly
# provisioning. Older versions can still work but need manual installs.
$os = Get-CimInstance -ClassName Win32_OperatingSystem
Write-Info "OS: $($os.Caption) (build $($os.BuildNumber))"


# =============================================================================
# STEP 1: WINGET (needed to install Git)
# =============================================================================

if (-not $SkipGit) {
    Write-Step "Checking for winget (Windows package manager)"
    if (Test-Command "winget") {
        $wingetVer = (& winget --version).Trim()
        Write-OK "winget available: $wingetVer"
    } else {
        Write-FailMsg "winget not found."
        Write-Host ""
        Write-Host "  winget is preinstalled on Windows 11 and modern Windows 10." -ForegroundColor Yellow
        Write-Host "  If missing, install App Installer from the Microsoft Store:" -ForegroundColor Yellow
        Write-Host "    ms-windows-store://pdp?ProductId=9NBLGGH4NNS1" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Alternatively, install Git manually from:" -ForegroundColor Yellow
        Write-Host "    https://git-scm.com/download/win" -ForegroundColor Yellow
        Write-Host "  then re-run bootstrap.ps1 with -SkipGit." -ForegroundColor Yellow
        exit 1
    }
}


# =============================================================================
# STEP 2: GIT
# =============================================================================

if ($SkipGit) {
    Write-Step "Skipping Git install (per -SkipGit)"
    if (Test-Command "git") {
        Write-OK "Git is available on PATH."
    } else {
        Write-WarnMsg "Git is NOT on PATH. You will need it to clone the repo."
    }
} else {
    Write-Step "Installing Git for Windows"

    if (Test-Command "git") {
        $gitVer = (& git --version).Trim()
        Write-OK "Git already installed: $gitVer"
    } else {
        Write-Info "Installing via winget (this takes a couple of minutes)..."
        # --silent avoids the installer GUI; --accept flags skip prompts.
        # exit code from winget can be non-zero even on success if the package
        # is already installed a different way; we verify via Get-Command below.
        & winget install --id Git.Git --exact --silent --accept-source-agreements --accept-package-agreements

        # Refresh PATH so newly installed git is visible in this session.
        Update-PathFromRegistry

        if (Test-Command "git") {
            $gitVer = (& git --version).Trim()
            Write-OK "Git installed: $gitVer"
        } else {
            Write-FailMsg "Git install completed but 'git' is not on PATH."
            Write-Host "  Close and reopen PowerShell, then verify with 'git --version'." -ForegroundColor Yellow
            Write-Host "  If that still fails, install manually from https://git-scm.com/download/win"
            exit 1
        }
    }
}


# =============================================================================
# STEP 3: UV
# =============================================================================

if ($SkipUv) {
    Write-Step "Skipping uv install (per -SkipUv)"
    if (Test-Command "uv") {
        Write-OK "uv is available on PATH."
    } else {
        Write-WarnMsg "uv is NOT on PATH. setup.ps1 will fail without it."
    }
} else {
    Write-Step "Installing uv (Python package manager)"

    if (Test-Command "uv") {
        $uvVer = (& uv --version).Trim()
        Write-OK "uv already installed: $uvVer"
    } else {
        Write-Info "Installing via Astral's official installer..."
        Write-Info "Source: https://astral.sh/uv/install.ps1"

        # Astral's install script. This is Python's most-recommended install
        # method and pulls the latest stable uv into %LOCALAPPDATA%\Programs.
        # The -ExecutionPolicy ByPass ensures it runs even in restricted policy
        # environments (this scope applies only to the child process).
        try {
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        } catch {
            Write-FailMsg "uv install failed: $($_.Exception.Message)"
            Write-Host ""
            Write-Host "  If the machine is offline, download the release manually from:" -ForegroundColor Yellow
            Write-Host "    https://github.com/astral-sh/uv/releases" -ForegroundColor Yellow
            Write-Host "  and place uv.exe on your PATH." -ForegroundColor Yellow
            exit 1
        }

        # uv installer modifies User PATH. Refresh so it's visible in this session.
        Update-PathFromRegistry

        if (Test-Command "uv") {
            $uvVer = (& uv --version).Trim()
            Write-OK "uv installed: $uvVer"
        } else {
            Write-FailMsg "uv install completed but 'uv' is not on PATH."
            Write-Host "  Close and reopen PowerShell, then verify with 'uv --version'." -ForegroundColor Yellow
            exit 1
        }
    }
}


# =============================================================================
# SUMMARY AND NEXT STEPS
# =============================================================================

Write-Step "Bootstrap complete"
Write-Host ""
Write-Host "  Tools installed:" -ForegroundColor Cyan
if (Test-Command "git") { Write-Host "    git:  $((& git --version).Trim())" }
if (Test-Command "uv")  { Write-Host "    uv:   $((& uv --version).Trim())" }
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Cyan
Write-Host "    1. Clone the pipeline repo:"
Write-Host "         git clone https://github.com/YOUR_USERNAME/GaussianSplat_videoTools.git"
Write-Host "    2. Enter the repo:"
Write-Host "         cd GaussianSplat_videoTools"
Write-Host "    3. Run the pipeline installer:"
Write-Host "         .\setup.ps1"
Write-Host ""
Write-Host "  If PATH changes are not visible in your current shell, close and reopen"
Write-Host "  PowerShell before running the next commands."
Write-Host ""
