$ErrorActionPreference = "Stop"

$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:STREAMLIT_SERVER_ADDRESS = "127.0.0.1"

python -m streamlit run frontend/app.py

