#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
DOWNLOAD_DIR="${DATA_DIR}/.downloads"

mkdir -p "${DOWNLOAD_DIR}" "${DATA_DIR}/coco" "${DATA_DIR}/sh17" "${DATA_DIR}/isafety" "${DATA_DIR}/roboflow"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf '[%s] WARNING: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

fail() {
  printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Required command not found: $1"
  fi
}

detect_hash_command() {
  if command -v sha256sum >/dev/null 2>&1; then
    echo "sha256sum"
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    echo "shasum -a 256"
    return
  fi
  fail "Neither sha256sum nor shasum is available for checksum verification."
}

HASH_CMD="$(detect_hash_command)"

hash_file() {
  eval "${HASH_CMD} \"$1\"" | awk '{print $1}'
}

download_file() {
  local url="$1"
  local output_path="$2"

  if [[ -f "${output_path}" ]]; then
    log "Using cached file: ${output_path}"
    return
  fi

  log "Downloading ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --retry 3 --fail --output "${output_path}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget --output-document="${output_path}" "${url}"
  else
    fail "Either curl or wget is required to download datasets."
  fi
}

verify_checksum() {
  local file_path="$1"
  local expected_checksum="${2:-}"

  if [[ -z "${expected_checksum}" ]]; then
    warn "No expected checksum provided for ${file_path}; strict checksum comparison skipped."
    return
  fi

  local actual_checksum
  actual_checksum="$(hash_file "${file_path}")"

  if [[ "${actual_checksum}" != "${expected_checksum}" ]]; then
    fail "Checksum mismatch for ${file_path}. expected=${expected_checksum} actual=${actual_checksum}"
  fi

  log "Checksum verified for ${file_path}"
}

extract_archive() {
  local archive_path="$1"
  local destination="$2"

  mkdir -p "${destination}"

  if [[ "${archive_path}" == *.zip ]]; then
    require_command unzip
    unzip -oq "${archive_path}" -d "${destination}"
    return
  fi

  if [[ "${archive_path}" == *.tar.gz || "${archive_path}" == *.tgz ]]; then
    tar -xzf "${archive_path}" -C "${destination}"
    return
  fi

  fail "Unsupported archive format: ${archive_path}"
}

count_files() {
  local target_dir="$1"
  if [[ ! -d "${target_dir}" ]]; then
    echo "0"
    return
  fi
  find "${target_dir}" -type f | wc -l | tr -d ' '
}

download_coco() {
  local coco_dir="${DATA_DIR}/coco"
  local train_zip="${DOWNLOAD_DIR}/train2017.zip"
  local val_zip="${DOWNLOAD_DIR}/val2017.zip"
  local ann_zip="${DOWNLOAD_DIR}/annotations_trainval2017.zip"

  download_file "http://images.cocodataset.org/zips/train2017.zip" "${train_zip}"
  verify_checksum "${train_zip}" "${COCO_TRAIN_SHA256:-}"

  download_file "http://images.cocodataset.org/zips/val2017.zip" "${val_zip}"
  verify_checksum "${val_zip}" "${COCO_VAL_SHA256:-}"

  download_file "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "${ann_zip}"
  verify_checksum "${ann_zip}" "${COCO_ANN_SHA256:-}"

  mkdir -p "${coco_dir}/images" "${coco_dir}/annotations"
  extract_archive "${train_zip}" "${coco_dir}/images"
  extract_archive "${val_zip}" "${coco_dir}/images"
  extract_archive "${ann_zip}" "${coco_dir}"

  if [[ ! -f "${coco_dir}/annotations/person_keypoints_train2017.json" ]]; then
    fail "Missing COCO file: person_keypoints_train2017.json"
  fi
  if [[ ! -f "${coco_dir}/annotations/person_keypoints_val2017.json" ]]; then
    fail "Missing COCO file: person_keypoints_val2017.json"
  fi

  log "COCO ready: train=$(count_files "${coco_dir}/images/train2017") val=$(count_files "${coco_dir}/images/val2017")"
}

clone_sh17() {
  local sh17_dir="${DATA_DIR}/sh17"

  if [[ -d "${sh17_dir}/.git" ]]; then
    log "SH17 repository already cloned."
  else
    rm -rf "${sh17_dir}"
    git clone --depth 1 https://github.com/ahmadmughees/SH17dataset.git "${sh17_dir}"
  fi

  if [[ -n "${SH17_EXPECTED_COMMIT:-}" ]]; then
    local commit_hash
    commit_hash="$(git -C "${sh17_dir}" rev-parse HEAD)"
    if [[ "${commit_hash}" != "${SH17_EXPECTED_COMMIT}" ]]; then
      fail "SH17 commit mismatch. expected=${SH17_EXPECTED_COMMIT} actual=${commit_hash}"
    fi
    log "SH17 commit verified: ${commit_hash}"
  else
    warn "SH17_EXPECTED_COMMIT not provided; commit hash verification skipped."
  fi

  log "SH17 ready: files=$(count_files "${sh17_dir}")"
}

download_isafety() {
  local archive_url="${ISAFETYBENCH_ARCHIVE_URL:-}"
  local isafety_dir="${DATA_DIR}/isafety"

  if [[ -z "${archive_url}" ]]; then
    warn "ISAFETYBENCH_ARCHIVE_URL is not set. Skipping iSafetyBench download."
    return
  fi

  local filename
  filename="$(basename "${archive_url}")"
  local archive_path="${DOWNLOAD_DIR}/${filename}"

  download_file "${archive_url}" "${archive_path}"
  verify_checksum "${archive_path}" "${ISAFETYBENCH_SHA256:-}"
  extract_archive "${archive_path}" "${isafety_dir}"

  log "iSafetyBench ready: clips=$(count_files "${isafety_dir}/clips")"
}

download_roboflow_worker_safety() {
  if [[ -z "${ROBOFLOW_API_KEY:-}" ]]; then
    warn "ROBOFLOW_API_KEY is not set. Skipping Roboflow Worker Safety download."
    return
  fi

  require_command python

  local output_dir="${DATA_DIR}/roboflow"
  export ROBOFLOW_DOWNLOAD_DIR="${output_dir}"

  python - <<'PY'
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
project_obj = rf.workspace(workspace).project(project)
dataset = project_obj.version(version).download("coco", location=str(output_dir))

print(dataset.location)
PY

  log "Roboflow Worker Safety ready: files=$(count_files "${output_dir}")"
}

print_summary() {
  log "Download summary"
  log "COCO files: $(count_files "${DATA_DIR}/coco")"
  log "SH17 files: $(count_files "${DATA_DIR}/sh17")"
  log "iSafety files: $(count_files "${DATA_DIR}/isafety")"
  log "Roboflow files: $(count_files "${DATA_DIR}/roboflow")"
}

main() {
  require_command git
  require_command find
  require_command wc

  download_coco
  clone_sh17
  download_isafety
  download_roboflow_worker_safety
  print_summary
}

main "$@"
