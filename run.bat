@echo off
REM ===============================================================
REM  Script chạy nhanh giao diện Streamlit CSTT - Swarm Intelligence
REM ===============================================================
cd /d "%~dp0"
echo [CSTT] Cai dat dependencies...
pip install -r requirements.txt
echo.
echo [CSTT] Khoi dong Streamlit app tren http://localhost:8501
streamlit run app.py
pause
