param(
    [Parameter(Mandatory=$true)][ValidateSet('p600','b580')][string]$Profile,
    [switch]$Verify,
    [string]$Source,
    [string]$Target,
    [string]$OutputDir = 'outputs/local',
    [switch]$Enhance,
    [switch]$RestoreFaces
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location (Split-Path $PSScriptRoot -Parent)
$pythonPath = Join-Path (Get-Location) ".venv-$Profile/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Run setup-hardware.ps1 -Profile $Profile first." }
$cliArgs = @('scripts/faceoff_local.py', '--profile', $Profile, '--output-dir', $OutputDir)
if ($Verify) { $cliArgs += '--verify' }
if ($Source) { $cliArgs += @('--source', $Source) }
if ($Target) { $cliArgs += @('--target', $Target) }
if ($Enhance) { $cliArgs += '--enhance' }
if ($RestoreFaces) { $cliArgs += '--restore-faces' }
& $pythonPath @cliArgs
if ($LASTEXITCODE -ne 0) { throw 'FaceOff failed. Review the local report/log.' }
