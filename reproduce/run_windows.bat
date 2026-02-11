\
@echo off
setlocal
echo [1/3] Creating venv...
python -m venv .venv
call .venv\Scripts\activate
echo [2/3] Installing requirements...
pip install -r requirements.txt
echo [3/3] Running simulation...
python atheer_sim.py
echo Done. Check outputs/ and paper_artifacts/
endlocal
