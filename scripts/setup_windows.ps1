[CmdletBinding()]
param(
    [switch]$SkipPythonPackages
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$repoRoot = Split-Path -Parent $PSScriptRoot
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'

if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Không tìm thấy Visual Studio Installer (vswhere.exe). Hãy cài Visual Studio với workload Desktop development with C++.'
}

$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    throw 'Không tìm thấy MSVC x64. Hãy thêm workload Desktop development with C++ trong Visual Studio Installer.'
}

$devShell = Join-Path $vsPath 'Common7\Tools\Launch-VsDevShell.ps1'
& $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

foreach ($command in @('cl', 'nvcc')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Thiếu công cụ bắt buộc: $command"
    }
}

$cmake = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ninja = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
if (-not (Test-Path -LiteralPath $cmake) -or -not (Test-Path -LiteralPath $ninja)) {
    throw 'Thiếu CMake/Ninja đi kèm Visual Studio. Hãy thêm component C++ CMake tools for Windows.'
}

if (-not $SkipPythonPackages) {
    $venv = Join-Path $repoRoot '.venv'
    if (-not (Test-Path -LiteralPath (Join-Path $venv 'Scripts\python.exe'))) {
        python -m venv $venv
    }
    $venvPython = Join-Path $venv 'Scripts\python.exe'
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r (Join-Path $repoRoot 'requirements.txt')
}

Write-Host 'Setup Windows hoàn tất.'
Write-Host "Visual Studio: $vsPath"
Write-Host "CMake: $cmake"
Write-Host "Ninja: $ninja"
