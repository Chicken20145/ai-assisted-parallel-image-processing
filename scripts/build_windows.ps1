[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$repoRoot = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $repoRoot 'build'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'Không tìm thấy Visual Studio C++ toolchain.' }

& (Join-Path $vsPath 'Common7\Tools\Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$cmake = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ninja = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'

if ($Clean -and (Test-Path -LiteralPath $buildDir)) {
    $resolvedBuild = (Resolve-Path -LiteralPath $buildDir).Path
    if (-not $resolvedBuild.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Từ chối xóa thư mục ngoài repository: $resolvedBuild"
    }
    Remove-Item -LiteralPath $resolvedBuild -Recurse -Force
}

& $cmake -S $repoRoot -B $buildDir -G Ninja `
    "-DCMAKE_BUILD_TYPE=$Configuration" `
    "-DCMAKE_MAKE_PROGRAM=$ninja" `
    '-DCMAKE_CUDA_ARCHITECTURES=native'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $cmake --build $buildDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $buildDir 'parallel_image_processing.exe')
