param([Parameter(Mandatory=$true)][ValidateSet('p600','b580')][string]$Profile)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$venvPath = Join-Path (Get-Location) ".venv-$Profile"
if (-not (Test-Path -LiteralPath "$venvPath/Scripts/python.exe")) {
    & py -3.12 -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
}
$pythonPath = Join-Path $venvPath 'Scripts/python.exe'
function Run-Python {
    & $pythonPath @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
Run-Python -m pip install 'pip==25.1.1' 'setuptools==78.1.0' 'wheel==0.45.1'
$torchIndex = if ($Profile -eq 'p600') { 'https://download.pytorch.org/whl/cu126' } else { 'https://download.pytorch.org/whl/cpu' }
Run-Python -m pip install 'torch==2.6.0' 'torchvision==0.21.0' --index-url $torchIndex
Run-Python -m pip install 'numpy==1.26.4' 'Cython==3.0.12'
# BasicSR's build isolation otherwise downloads another, unpinned Torch stack.
Run-Python -m pip install --no-build-isolation -r "requirements/$Profile.txt"
# These distributions all install the same onnxruntime module. Never leave two
# of them sharing module files. Keep pip's missing CPU dependency warning visible.
& $pythonPath -m pip show onnxruntime *> $null
if ($LASTEXITCODE -eq 0) {
    Run-Python -m pip uninstall -y onnxruntime
    $ortPackage = if ($Profile -eq 'p600') { 'onnxruntime-gpu==1.20.2' } else { 'onnxruntime-openvino==1.23.0' }
    Run-Python -m pip install --force-reinstall --no-deps $ortPackage
}
Write-Host "Installed $Profile. Run: .\scripts\run-hardware.ps1 -Profile $Profile -Verify"
