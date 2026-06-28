# Lance l'interface web subgen. Usage : .\web.ps1   puis ouvre http://localhost:7860
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Start-Process "http://localhost:7860"
& $py -m subgen.webapp
