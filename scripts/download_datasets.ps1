# Tentellect dataset download (Windows / PowerShell).
# Parity with scripts/download_datasets.sh — run from repo root:
#   powershell -ExecutionPolicy Bypass -File scripts/download_datasets.ps1
#
# Optional env vars (same semantics as the bash script):
#   COCO_TRAIN_SHA256, COCO_VAL_SHA256, COCO_ANN_SHA256
#   ISAFETYBENCH_ARCHIVE_URL, ISAFETYBENCH_SHA256
#   ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION
#   SH17_EXPECTED_COMMIT

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DataDir = Join-Path $RootDir "data"
$DownloadDir = Join-Path $DataDir ".downloads"

function Write-Log([string]$Message) {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

function Test-Sha256([string]$FilePath, [string]$Expected) {
    if ([string]::IsNullOrWhiteSpace($Expected)) {
        Write-Warning "No expected checksum for $FilePath; skipping verification."
        return
    }
    $hash = (Get-FileHash -Path $FilePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $Expected.ToLowerInvariant()) {
        throw "Checksum mismatch for $FilePath. expected=$Expected actual=$hash"
    }
    Write-Log "Checksum verified for $FilePath"
}

function Get-FileCount([string]$Dir) {
    if (-not (Test-Path $Dir)) { return 0 }
    return (Get-ChildItem -Path $Dir -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
}

New-Item -ItemType Directory -Force -Path $DownloadDir, (Join-Path $DataDir "coco"), (Join-Path $DataDir "isafety\clips"), (Join-Path $DataDir "roboflow") | Out-Null

# --- COCO: annotations (required for keypoint JSON); images are large — use bash script or download zips separately ---
$annZip = Join-Path $DownloadDir "annotations_trainval2017.zip"
if (-not (Test-Path $annZip)) {
    Write-Log "Downloading COCO annotations_trainval2017.zip"
    Invoke-WebRequest -Uri "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -OutFile $annZip -UseBasicParsing
}
if ($env:COCO_ANN_SHA256) { Test-Sha256 $annZip $env:COCO_ANN_SHA256 }
$CocoRoot = Join-Path $DataDir "coco"
Write-Log "Extracting COCO annotations to $CocoRoot"
Expand-Archive -Path $annZip -DestinationPath $CocoRoot -Force

# Optional: train2017 / val2017 image zips (set $env:DOWNLOAD_COCO_IMAGES=1 to fetch ~18GB)
if ($env:DOWNLOAD_COCO_IMAGES -eq "1") {
    $trainZip = Join-Path $DownloadDir "train2017.zip"
    $valZip = Join-Path $DownloadDir "val2017.zip"
    if (-not (Test-Path $trainZip)) {
        Write-Log "Downloading COCO train2017.zip (large)"
        Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/train2017.zip" -OutFile $trainZip -UseBasicParsing
    }
    if ($env:COCO_TRAIN_SHA256) { Test-Sha256 $trainZip $env:COCO_TRAIN_SHA256 }
    if (-not (Test-Path $valZip)) {
        Write-Log "Downloading COCO val2017.zip"
        Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/val2017.zip" -OutFile $valZip -UseBasicParsing
    }
    if ($env:COCO_VAL_SHA256) { Test-Sha256 $valZip $env:COCO_VAL_SHA256 }
    $imgRoot = Join-Path $CocoRoot "images"
    New-Item -ItemType Directory -Force -Path $imgRoot | Out-Null
    Expand-Archive -Path $trainZip -DestinationPath $imgRoot -Force
    Expand-Archive -Path $valZip -DestinationPath $imgRoot -Force
}

# --- SH17: upstream repo (YOLO images/labels: download from Kaggle — see data/sh17/README.md) ---
$Sh17Dir = Join-Path $DataDir "sh17"
if (Test-Path (Join-Path $Sh17Dir ".git")) {
    Write-Log "SH17 repository already cloned."
} else {
    if (Test-Path $Sh17Dir) { Remove-Item -Recurse -Force $Sh17Dir }
    Write-Log "Cloning SH17 dataset repo"
    git clone --depth 1 "https://github.com/ahmadmughees/SH17dataset.git" $Sh17Dir
}
if ($env:SH17_EXPECTED_COMMIT) {
    Push-Location $Sh17Dir
    try {
        $head = (git rev-parse HEAD).Trim()
        if ($head -ne $env:SH17_EXPECTED_COMMIT) {
            throw "SH17 commit mismatch. expected=$($env:SH17_EXPECTED_COMMIT) actual=$head"
        }
        Write-Log "SH17 commit verified: $head"
    } finally { Pop-Location }
}
New-Item -ItemType Directory -Force -Path (Join-Path $Sh17Dir "images"), (Join-Path $Sh17Dir "labels") | Out-Null

# --- iSafetyBench (optional URL) ---
if ($env:ISAFETYBENCH_ARCHIVE_URL) {
    $name = Split-Path $env:ISAFETYBENCH_ARCHIVE_URL -Leaf
    $arch = Join-Path $DownloadDir $name
    Write-Log "Downloading iSafetyBench archive"
    Invoke-WebRequest -Uri $env:ISAFETYBENCH_ARCHIVE_URL -OutFile $arch -UseBasicParsing
    if ($env:ISAFETYBENCH_SHA256) { Test-Sha256 $arch $env:ISAFETYBENCH_SHA256 }
    $isafety = Join-Path $DataDir "isafety"
    if ($arch -match '\.zip$') {
        Expand-Archive -Path $arch -DestinationPath $isafety -Force
    } else {
        tar -xzf $arch -C $isafety
    }
}

# --- Roboflow (optional API key) ---
if ($env:ROBOFLOW_API_KEY) {
    $out = Join-Path $DataDir "roboflow"
    $env:ROBOFLOW_DOWNLOAD_DIR = $out
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) { $py = Get-Command py }
    & $py.Source -c @'
from pathlib import Path
import os
from roboflow import Roboflow
api_key = os.environ["ROBOFLOW_API_KEY"]
workspace = os.environ.get("ROBOFLOW_WORKSPACE", "roboflow-universe-projects")
project = os.environ.get("ROBOFLOW_PROJECT", "worker-safety")
version = int(os.environ.get("ROBOFLOW_VERSION", "1"))
output_dir = Path(os.environ["ROBOFLOW_DOWNLOAD_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)
rf = Roboflow(api_key=api_key)
dataset = rf.workspace(workspace).project(project).version(version).download("coco", location=str(output_dir))
print(dataset.location)
'@
}

Write-Log "Download summary"
Write-Log ("COCO files: {0}" -f (Get-FileCount (Join-Path $DataDir "coco")))
Write-Log ("SH17 files: {0}" -f (Get-FileCount $Sh17Dir))
Write-Log ("iSafety files: {0}" -f (Get-FileCount (Join-Path $DataDir "isafety")))
Write-Log ("Roboflow files: {0}" -f (Get-FileCount (Join-Path $DataDir "roboflow")))
Write-Log "Done. Run: python scripts/validate_datasets.py --data-root data"
