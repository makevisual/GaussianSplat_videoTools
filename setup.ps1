<#
.SYNOPSIS
    GaussianSplat_videoTools installer (Windows / PowerShell).

.DESCRIPTION
    One-shot install for the splat pipeline on a fresh machine, verified end-to-end.

    Design principles:
      - Every step announces itself BEFORE it starts, not after. If the script
        appears to hang, the last printed line tells you where.
      - Every step verifies its own outcome. A step that "completes" without
        actually working (empty download, CPU-only torch, missing binary)
        stops the script with a specific error instead of cascading forward.
      - Idempotent: re-running is always safe. Already-installed pieces are
        detected and skipped with a clear notice.
      - No hidden config parsing. All settings live in $Config at the top
        of this script, in plain PowerShell. No TOML/JSON reading required.
      - Pass-through output from git, uv, and downloads. Silence during long
        operations is a code smell.
      - Pure ASCII. No em-dashes, en-dashes, or curly quotes anywhere.
        PowerShell on Windows Server sometimes reads scripts as Windows-1252;
        multi-byte UTF-8 causes parse errors that are hard to diagnose.

.PARAMETER Force
    Reinstall components even if they appear already present. Useful for
    recovering from a broken install.

.PARAMETER SkipCheckpoint
    Skip the SHARP checkpoint download. Use if the checkpoint is already
    verified in place and you're just re-verifying other pieces.

.PARAMETER CheckOnly
    Run only the verification checks; skip installs entirely. Fast way to
    audit a completed install without touching anything.

.EXAMPLE
    .\setup.ps1
    Full install, skipping steps whose outcome is already present.

.EXAMPLE
    .\setup.ps1 -Force
    Full reinstall, replacing all detected components.

.EXAMPLE
    .\setup.ps1 -CheckOnly
    Verify the install without changing anything.

.NOTES
    Author: MAKE pipeline
    Tested: Windows 10/11, PowerShell 5.1+, RTX A6000 with driver 566.07
#>

[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$SkipCheckpoint,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Set-Location -Path $PSScriptRoot


# =============================================================================
# CONFIG
# =============================================================================
# All settings in one place, plain PowerShell hashtable. Edit here to update
# URLs, versions, or paths. This replaces the earlier config/pipeline.toml
# read from the script (that file is retained for reference but not parsed).

$Config = @{
    # --- Apple ML-SHARP source ---------------------------------------------
    # Pin a specific commit hash after your first known-good install so
    # future installs are reproducible. Until then, we clone the tip of main.
    SharpRepoUrl         = "https://github.com/apple/ml-sharp.git"
    SharpPinnedCommit    = "1eaa046834b81852261262b41b0919f5c1efdd2e"    # empty = clone tip; fill in after v0.1.0

    # --- SHARP checkpoint --------------------------------------------------
    CheckpointFilename   = "sharp_2572gikvuh.pt"
    CheckpointMinBytes   = 1500000000   # 1.5GB minimum; real file is ~2GB
    # Try TrueNAS mirror first (if reachable), fall back to Apple's CDN.
    # UNC paths are tried as file copies; https URLs are downloaded.
    CheckpointSources    = @(
        "\\TRUENAS\path\to\splat_vendor\sharp_2572gikvuh.pt",
        "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    )
    # SHA256 hash. Fill in after first successful download, then future
    # installs verify integrity. Set to "" to skip verification.
    CheckpointSha256     = "94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24F073d"

    # --- PyTorch (CUDA build) ---------------------------------------------
    # CUDA 12.4 wheels are compatible with any NVIDIA driver supporting
    # CUDA 12.4 or newer. Update this if Apple changes SHARP's torch
    # requirement, or if you want to move to a newer CUDA family.
    TorchIndexUrl        = "https://download.pytorch.org/whl/cu124"
    # No version pin; pip picks the latest compatible cu124 wheel. Pin
    # specific versions here after first known-good install for reproducibility.
    TorchPackages        = @("torch", "torchvision")

    # --- Local paths -------------------------------------------------------
    SharpInstallDir      = "external\ml-sharp"
    CheckpointDir        = "checkpoints"
}


# =============================================================================
# HELPERS
# =============================================================================

# Elapsed-time tracking helps identify slow phases. StartTime captures when
# the current step began; PhaseTimings accumulates per-step durations for
# the summary at the end.
$Script:PhaseStart = $null
$Script:PhaseTimings = [ordered]@{}

function Write-Step {
    <#
        Announces a phase before it runs. Prints a blank line above for
        readability, cyan header for scannability. Records the start time
        so we can report elapsed at end-of-phase.
    #>
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
    $Script:PhaseStart = Get-Date
    $Script:CurrentPhase = $Message
}

function Complete-Step {
    <#
        Marks end of the current phase, records duration, prints result.
    #>
    param([string]$Result = "done")
    if ($null -ne $Script:PhaseStart) {
        $elapsed = (Get-Date) - $Script:PhaseStart
        $Script:PhaseTimings[$Script:CurrentPhase] = $elapsed
        $secs = [math]::Round($elapsed.TotalSeconds, 1)
        Write-Host "    [OK] $Result ($secs sec)" -ForegroundColor Green
    } else {
        Write-Host "    [OK] $Result" -ForegroundColor Green
    }
    $Script:PhaseStart = $null
}

function Write-Info {
    param([string]$Message)
    Write-Host "    $Message" -ForegroundColor Gray
}

function Write-WarnMsg {
    param([string]$Message)
    Write-Host "    [WARN] $Message" -ForegroundColor Yellow
}

function Write-FailMsg {
    <#
        Prints a failure message. Does not exit; caller decides whether to
        continue or abort. Called before `throw` in most cases.
    #>
    param([string]$Message)
    Write-Host "    [FAIL] $Message" -ForegroundColor Red
}

function Test-Command {
    param([string]$Name)
    $null = Get-Command $Name -ErrorAction SilentlyContinue
    return $?
}

function Get-FileSha256Lower {
    param([string]$Path)
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLower()
}


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================
# Each verification returns $true if the check passes, $false otherwise.
# Called both after install steps AND from -CheckOnly mode.

function Test-UvAvailable {
    if (-not (Test-Command "uv")) { return $false }
    return $true
}

function Test-PythonVersionOk {
    <#
        Uses uv's own reporting rather than shelling out to python directly,
        because "python" on PATH may be a different install than the one
        uv provisioned for this project.
    #>
    try {
        $out = & uv run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
        return ($out -eq "3.13")
    } catch {
        return $false
    }
}

function Test-TorchCudaAvailable {
    <#
        Runs a small Python probe inside the venv to check torch's CUDA
        availability. This is the critical check that catches "install
        succeeded but got the CPU wheel" - the exact failure mode we
        hit last time.
    #>
    try {
        $out = & uv run python -c "import torch; print(torch.cuda.is_available())" 2>&1
        return ($out.Trim() -eq "True")
    } catch {
        return $false
    }
}

function Get-TorchVersion {
    try {
        return (& uv run python -c "import torch; print(torch.__version__)" 2>&1).Trim()
    } catch {
        return "unknown"
    }
}

function Test-SharpCliAvailable {
    try {
        $result = & uv run sharp --help 2>&1
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Test-CheckpointValid {
    <#
        Checks that the checkpoint file exists AND is at least the minimum
        expected size. A zero-byte or truncated file is worse than a missing
        file because it silently corrupts downstream inference.
    #>
    param([string]$Path, [long]$MinBytes)
    if (-not (Test-Path $Path)) { return $false }
    $size = (Get-Item $Path).Length
    return ($size -ge $MinBytes)
}


# =============================================================================
# INSTALL STEPS
# =============================================================================

function Install-UvIfMissing {
    Write-Step "Checking for uv"
    if (Test-UvAvailable) {
        $version = (& uv --version) -join " "
        Write-Info "Found: $version"
        Complete-Step "uv is installed"
        return
    }

    Write-FailMsg "uv not installed. Cannot continue."
    Write-Host ""
    Write-Host "  Install uv with this one-liner (then re-run setup.ps1):" -ForegroundColor Yellow
    Write-Host '    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"' -ForegroundColor Yellow
    Write-Host ""
    throw "uv missing"
}

function Install-PythonRuntime {
    Write-Step "Ensuring Python 3.13 is available via uv"
    # `uv python install` is idempotent; it noops if 3.13 is already there.
    # Passing through output so we see progress if it does install.
    & uv python install 3.13
    if ($LASTEXITCODE -ne 0) {
        throw "uv python install failed with exit code $LASTEXITCODE"
    }
    Complete-Step "Python 3.13 available"
}

function Install-ProjectVenv {
    <#
        `uv sync` reads pyproject.toml (and uv.lock if present) to create a
        venv at .venv that exactly matches the declared dependency tree.
        First run generates uv.lock; subsequent runs replay it deterministically.
    #>
    Write-Step "Syncing venv from pyproject.toml"
    Write-Info "Running: uv sync"
    & uv sync
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE"
    }
    Complete-Step ".venv is in sync with pyproject.toml"
}

function Install-TorchCuda {
    <#
        This is the critical step that most commonly fails silently. Order
        of operations:
          1. Uninstall any torch/torchvision that pyproject.toml may have
             pulled from PyPI as a CPU-only wheel. Idempotent - safe if
             they're not installed.
          2. Install torch/torchvision from PyTorch's CUDA wheel index.
          3. Verify that torch.cuda.is_available() returns True. If not,
             throw - we do NOT want to complete this step silently with
             a CPU torch.
    #>
    Write-Step "Installing PyTorch with CUDA support"
    Write-Info "Index URL: $($Config.TorchIndexUrl)"

    # Check current state.
    $currentVersion = Get-TorchVersion
    $cudaOk = Test-TorchCudaAvailable

    if ($cudaOk -and -not $Force) {
        Write-Info "torch already installed with CUDA: $currentVersion"
        Complete-Step "PyTorch CUDA already present"
        return
    }

    if ($currentVersion -ne "unknown") {
        Write-Info "Current torch: $currentVersion (will replace)"
        Write-Info "Uninstalling CPU torch..."
        & uv pip uninstall torch torchvision
        # Uninstall exit code can be non-zero if the packages weren't there;
        # that's fine for our purposes.
    }

    Write-Info "Installing CUDA torch (this takes several minutes; ~2.5GB download)..."
    & uv pip install --index-url $Config.TorchIndexUrl @($Config.TorchPackages)
    if ($LASTEXITCODE -ne 0) {
        throw "torch install failed with exit code $LASTEXITCODE"
    }

    # Verify - this is the critical check.
    $newVersion = Get-TorchVersion
    Write-Info "Installed torch: $newVersion"
    if (-not (Test-TorchCudaAvailable)) {
        Write-FailMsg "torch installed but CUDA still not available. Investigate manually."
        throw "torch CUDA verification failed"
    }
    Complete-Step "PyTorch CUDA verified working"
}

function Install-SharpRepo {
    <#
        Clone Apple ML-SHARP to external/ml-sharp. If already cloned, either
        pull the latest (if no pinned commit) or verify we're at the pinned
        commit. With -Force, delete and re-clone.
    #>
    Write-Step "Cloning Apple ML-SHARP"
    $sharpDir = Join-Path $PSScriptRoot $Config.SharpInstallDir
    $sharpGit = Join-Path $sharpDir ".git"
    Write-Info "Target: $sharpDir"

    if ((Test-Path $sharpGit) -and -not $Force) {
        Write-Info "SHARP already cloned. Fetching latest..."
        Push-Location $sharpDir
        try {
            & git fetch
            if ($Config.SharpPinnedCommit -ne "") {
                & git checkout $Config.SharpPinnedCommit
                Write-Info "Checked out pinned commit: $($Config.SharpPinnedCommit)"
            } else {
                Write-Info "No pinned commit; leaving current branch as-is."
            }
        } finally {
            Pop-Location
        }
    } else {
        if (Test-Path $sharpDir) {
            Write-Info "Removing existing directory (-Force specified)"
            Remove-Item -Recurse -Force $sharpDir
        }
        Write-Info "Cloning from $($Config.SharpRepoUrl)..."
        & git clone $Config.SharpRepoUrl $sharpDir
        if ($LASTEXITCODE -ne 0) {
            throw "git clone failed with exit code $LASTEXITCODE"
        }
        if ($Config.SharpPinnedCommit -ne "") {
            Push-Location $sharpDir
            try {
                & git checkout $Config.SharpPinnedCommit
                Write-Info "Checked out pinned commit: $($Config.SharpPinnedCommit)"
            } finally {
                Pop-Location
            }
        }
    }
    Complete-Step "SHARP repo ready"
}

function Install-SharpPackage {
    <#
        Install SHARP into the venv as an editable install. Editable so
        that `git pull` in external/ml-sharp propagates to the venv
        without a reinstall.
    #>
    Write-Step "Installing SHARP into venv"
    $sharpDir = Join-Path $PSScriptRoot $Config.SharpInstallDir

    # Check if already installed cleanly - the sharp CLI as our proxy.
    if ((Test-SharpCliAvailable) -and -not $Force) {
        Write-Info "sharp CLI already working, skipping install."
        Complete-Step "SHARP already installed"
        return
    }

    Write-Info "Running: uv pip install -e $sharpDir"
    & uv pip install -e $sharpDir
    if ($LASTEXITCODE -ne 0) {
        throw "SHARP pip install failed with exit code $LASTEXITCODE"
    }

    if (-not (Test-SharpCliAvailable)) {
        Write-FailMsg "SHARP installed but 'sharp --help' fails."
        throw "SHARP CLI verification failed"
    }
    Complete-Step "SHARP installed and CLI verified"
}

function Get-CheckpointWithProgress {
    <#
        Download the checkpoint from a URL, showing progress. Uses .NET
        HttpClient for control over the transfer. Progress bar so a 2GB
        download doesn't look like a hang.

        Returns $true on success, $false on any failure.
    #>
    param(
        [string]$Url,
        [string]$Dest
    )

    try {
        Add-Type -AssemblyName System.Net.Http

        # Ensure TLS 1.2 for older PowerShell.
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

        $client = New-Object System.Net.Http.HttpClient
        $client.Timeout = [TimeSpan]::FromMinutes(30)

        $response = $client.GetAsync($Url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        if (-not $response.IsSuccessStatusCode) {
            Write-Info "  HTTP $([int]$response.StatusCode) from $Url"
            return $false
        }

        $totalBytes = $response.Content.Headers.ContentLength
        if ($null -eq $totalBytes) { $totalBytes = 0 }
        $totalMb = [math]::Round($totalBytes / 1MB, 1)
        Write-Info "  Downloading $totalMb MB..."

        $stream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $fileStream = [System.IO.File]::Create($Dest)

        $buffer = New-Object byte[] 1048576   # 1MB read buffer
        $totalRead = 0L
        $lastPrint = Get-Date
        $printIntervalSec = 2

        try {
            while ($true) {
                $bytesRead = $stream.Read($buffer, 0, $buffer.Length)
                if ($bytesRead -le 0) { break }
                $fileStream.Write($buffer, 0, $bytesRead)
                $totalRead += $bytesRead

                # Rate-limited progress print. 2 sec cadence balances
                # useful feedback against console spam.
                if (((Get-Date) - $lastPrint).TotalSeconds -ge $printIntervalSec) {
                    if ($totalBytes -gt 0) {
                        $pct = [math]::Round(100.0 * $totalRead / $totalBytes, 1)
                        $mb  = [math]::Round($totalRead / 1MB, 1)
                        Write-Info "    $mb MB / $totalMb MB  ($pct%)"
                    } else {
                        $mb  = [math]::Round($totalRead / 1MB, 1)
                        Write-Info "    $mb MB downloaded"
                    }
                    $lastPrint = Get-Date
                }
            }
        } finally {
            $fileStream.Close()
            $stream.Close()
            $client.Dispose()
        }
        return $true
    } catch {
        Write-Info "  Download error: $($_.Exception.Message)"
        # Delete partial file so we don't leave a truncated blob behind.
        if (Test-Path $Dest) { Remove-Item -Force $Dest }
        return $false
    }
}

function Get-CheckpointFromUnc {
    param([string]$Path, [string]$Dest)
    Write-Info "  Trying UNC path: $Path"
    if (-not (Test-Path $Path)) {
        Write-Info "  UNC path not reachable."
        return $false
    }
    try {
        Copy-Item -Path $Path -Destination $Dest -Force
        return $true
    } catch {
        Write-Info "  UNC copy failed: $($_.Exception.Message)"
        return $false
    }
}

function Install-Checkpoint {
    Write-Step "Fetching SHARP checkpoint"
    $ckptDir  = Join-Path $PSScriptRoot $Config.CheckpointDir
    $ckptPath = Join-Path $ckptDir $Config.CheckpointFilename

    if (-not (Test-Path $ckptDir)) {
        New-Item -ItemType Directory -Path $ckptDir -Force | Out-Null
    }

    if ((Test-CheckpointValid -Path $ckptPath -MinBytes $Config.CheckpointMinBytes) -and -not $Force) {
        $sizeMb = [math]::Round((Get-Item $ckptPath).Length / 1MB, 0)
        Write-Info "Checkpoint already present ($sizeMb MB). Use -Force to re-download."
        Complete-Step "Checkpoint verified"
        return
    }

    # If a partial/broken file exists, remove it before trying sources.
    if (Test-Path $ckptPath) {
        $currentSize = (Get-Item $ckptPath).Length
        Write-Info "Existing file at $ckptPath is $currentSize bytes; removing before re-download."
        Remove-Item -Force $ckptPath
    }

    $success = $false
    foreach ($src in $Config.CheckpointSources) {
        Write-Info "Trying source: $src"
        if ($src.StartsWith("\\")) {
            if (Get-CheckpointFromUnc -Path $src -Dest $ckptPath) {
                $success = $true
                Write-Info "  Copied from UNC path successfully."
                break
            }
        } elseif ($src.StartsWith("http")) {
            if (Get-CheckpointWithProgress -Url $src -Dest $ckptPath) {
                $success = $true
                Write-Info "  Downloaded successfully."
                break
            }
        } else {
            Write-WarnMsg "Unknown source scheme: $src (skipping)"
        }
    }

    if (-not $success) {
        throw "All checkpoint sources failed."
    }

    # Verify size AND (if configured) SHA256.
    if (-not (Test-CheckpointValid -Path $ckptPath -MinBytes $Config.CheckpointMinBytes)) {
        $size = (Get-Item $ckptPath).Length
        Write-FailMsg "Checkpoint is $size bytes; expected at least $($Config.CheckpointMinBytes)."
        throw "Checkpoint size check failed"
    }

    if ($Config.CheckpointSha256 -ne "") {
        Write-Info "Computing SHA256..."
        $actual = Get-FileSha256Lower -Path $ckptPath
        if ($actual -ne $Config.CheckpointSha256) {
            Write-FailMsg "SHA256 mismatch. Expected $($Config.CheckpointSha256), got $actual"
            throw "Checkpoint SHA256 verification failed"
        }
        Write-Info "SHA256 matches."
    } else {
        Write-Info "SHA256 not configured; skipping hash verification."
        Write-Info "Compute with: Get-FileHash '$ckptPath' -Algorithm SHA256"
        Write-Info "and paste into setup.ps1 `$Config.CheckpointSha256 for future integrity checks."
    }

    Complete-Step "Checkpoint downloaded and verified"
}


# =============================================================================
# VERIFICATION SUMMARY
# =============================================================================

function Invoke-VersionCheck {
    Write-Step "Running full diagnostic (gs-version-check)"
    & uv run gs-version-check
    $exit = $LASTEXITCODE
    if ($exit -eq 0) {
        Complete-Step "All diagnostic checks passed"
    } elseif ($exit -eq 1) {
        Write-WarnMsg "Diagnostic passed with warnings (install usable)."
    } else {
        Write-FailMsg "Diagnostic reports install is broken."
        throw "gs-version-check reported errors"
    }
}

function Show-Summary {
    Write-Step "Install summary"
    Write-Host ""
    Write-Host "  Repo root:     $PSScriptRoot"
    Write-Host "  venv:          $PSScriptRoot\.venv"
    Write-Host "  SHARP repo:    $PSScriptRoot\$($Config.SharpInstallDir)"
    Write-Host "  Checkpoint:    $PSScriptRoot\$($Config.CheckpointDir)\$($Config.CheckpointFilename)"
    Write-Host ""
    Write-Host "  Phase timings:" -ForegroundColor Cyan
    foreach ($k in $Script:PhaseTimings.Keys) {
        $secs = [math]::Round($Script:PhaseTimings[$k].TotalSeconds, 1)
        Write-Host "    $secs sec  $k"
    }
    Write-Host ""
    Write-Host "  Next steps:" -ForegroundColor Cyan
    Write-Host "    uv run gs-version-check       # re-verify anytime"
    Write-Host "    uv run gs-footage-to-splats --help"
    Write-Host "    uv run gs-filter --help"
    Write-Host ""
    Write-Host "  Or activate the venv manually:"
    Write-Host "    .\.venv\Scripts\Activate.ps1"
    Write-Host ""
}


# =============================================================================
# MAIN
# =============================================================================

try {
    Write-Host ""
    Write-Host "GaussianSplat_videoTools setup" -ForegroundColor Cyan
    Write-Host "Repo root: $PSScriptRoot"
    if ($Force)          { Write-Host "Mode:      Force reinstall" -ForegroundColor Yellow }
    if ($CheckOnly)      { Write-Host "Mode:      CheckOnly (no installs)" -ForegroundColor Yellow }
    if ($SkipCheckpoint) { Write-Host "Mode:      SkipCheckpoint" -ForegroundColor Yellow }
    Write-Host ""

    if ($CheckOnly) {
        # Report-only mode: skip installs, run diagnostic, exit.
        Invoke-VersionCheck
        exit $LASTEXITCODE
    }

    Install-UvIfMissing
    Install-PythonRuntime
    Install-ProjectVenv
    Install-TorchCuda
    Install-SharpRepo
    Install-SharpPackage

    if ($SkipCheckpoint) {
        Write-Step "Skipping checkpoint (per -SkipCheckpoint)"
        Complete-Step "Skipped"
    } else {
        Install-Checkpoint
    }

    Invoke-VersionCheck
    Show-Summary

    Write-Host "Install complete." -ForegroundColor Green
    exit 0

} catch {
    Write-Host ""
    Write-Host "SETUP FAILED: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Phase that failed: $Script:CurrentPhase" -ForegroundColor Red
    Write-Host ""
    Write-Host "Rerun with -Force to reinstall from scratch, or investigate the specific step above."
    exit 1
}