#!/usr/bin/env bash
set -e
echo "[1/3] Creating venv..."
python3 -m venv .venv
source .venv/bin/activate
echo "[2/3] Installing requirements..."
pip install -r requirements.txt
echo "[3/3] Running simulation..."
python atheer_sim.py
echo "Done. Check outputs/ and paper_artifacts/"
