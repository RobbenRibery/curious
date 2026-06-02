#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MODAL_BIN:-}" ]]; then
  if [[ -x "${HOME}/.local/bin/modal" ]]; then
    MODAL_BIN="${HOME}/.local/bin/modal"
  else
    MODAL_BIN="modal"
  fi
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  MODAL_TARGET="$(readlink "${MODAL_BIN}" || printf '%s' "${MODAL_BIN}")"
  MODAL_BIN_DIR="$(cd "$(dirname "${MODAL_TARGET}")" && pwd)"
  MODAL_PYTHON="${MODAL_BIN_DIR}/python"
  if [[ -x "${MODAL_PYTHON}" ]]; then
    exec "${MODAL_PYTHON}" -m curious.modal_train --help
  fi
fi

exec "${MODAL_BIN}" run -m curious.modal_train -- "$@"
