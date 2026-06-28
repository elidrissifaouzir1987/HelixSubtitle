# Lance subgen avec le Python du venv. Usage : .\run.ps1 "ma_video.mp4" -t fr
param([Parameter(ValueFromRemainingArguments = $true)] $Rest)
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m subgen @Rest
exit $LASTEXITCODE
