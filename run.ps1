param([Parameter(ValueFromRemainingArguments=$true)][string[]]$AppArguments)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath($PSScriptRoot)
$launcher=Join-Path $projectRoot 'scripts\launch_kd.py'

function Try-Python([string]$executable,[string[]]$prefixArguments) {
    $command=Get-Command $executable -ErrorAction SilentlyContinue
    if ($null -eq $command -or $command.Source -like '*\WindowsApps\*') { return }
    & $executable @prefixArguments -I -X utf8 -c 'import sys; sys.exit(sys.version_info[:2] != (3,12) or sys.maxsize <= 2**32)' *> $null
    if ($LASTEXITCODE -eq 0) {
        & $executable @prefixArguments -I -X utf8 $launcher @AppArguments
        exit $LASTEXITCODE
    }
}

try {
    if ($env:KD_PYTHON) {
        Try-Python $env:KD_PYTHON @()
        throw 'KD_PYTHON must point to a working Python 3.12 executable.'
    }
    if ($env:KD_USE_MANAGED_PYTHON -ne '1') {
        Try-Python (Join-Path $projectRoot '.venv\Scripts\python.exe') @()
        Try-Python 'py' @('-3.12')
        Try-Python 'python' @()
        Try-Python 'python3' @()
    }
    & (Join-Path $projectRoot 'scripts\bootstrap_windows.ps1') @AppArguments
    exit $LASTEXITCODE
} catch {
    Write-Host ('Knowledge distillation: '+$_.Exception.Message)
    exit 1
}
