[CmdletBinding()]
param(
    [switch]$BuildFirst
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$repoRoot = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $repoRoot 'build'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'Không tìm thấy Visual Studio C++ toolchain.' }

if ($BuildFirst) {
    & (Join-Path $PSScriptRoot 'build_windows.ps1') -Configuration Release
}
if (-not (Test-Path -LiteralPath $buildDir)) {
    throw 'Chưa có thư mục build. Chạy scripts/build_windows.ps1 trước hoặc dùng -BuildFirst.'
}

$ctest = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'
if (-not (Test-Path -LiteralPath $ctest)) { throw 'Không tìm thấy CTest đi kèm Visual Studio.' }
& $ctest --test-dir $buildDir --output-on-failure -C Release
