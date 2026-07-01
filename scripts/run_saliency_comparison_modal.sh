#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MODAL_BIN:-}" ]]; then
  if [[ -x "${HOME}/.local/bin/modal" ]]; then
    MODAL_BIN="${HOME}/.local/bin/modal"
  else
    MODAL_BIN="modal"
  fi
fi

exec "${MODAL_BIN}" run -m curious.modal_saliency_compare -- "$@"
