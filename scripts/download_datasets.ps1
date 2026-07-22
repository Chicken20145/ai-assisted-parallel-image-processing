[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$repoRoot = Split-Path -Parent $PSScriptRoot
$downloadDir = Join-Path $repoRoot 'data\downloads'
$externalDir = Join-Path $repoRoot 'data\external'
$archive = Join-Path $downloadDir 'BSDS300-images.tgz'
$expectedHash = 'A5F7D0E49FE135C75518A3543CED24470156FD69305AE77845DFF2A5138652B4'
$url = 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz'

New-Item -ItemType Directory -Force -Path $downloadDir, $externalDir | Out-Null
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -Uri $url -OutFile $archive
}

$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash
if ($actualHash -ne $expectedHash) {
    throw "Checksum BSDS300 không hợp lệ. Mong đợi $expectedHash, nhận $actualHash"
}

$imageRoot = Join-Path $externalDir 'BSDS300\images'
if (-not (Test-Path -LiteralPath $imageRoot)) {
    tar -xzf $archive -C $externalDir
}

$count = (Get-ChildItem -LiteralPath $imageRoot -Recurse -File -Filter '*.jpg').Count
if ($count -ne 300) { throw "BSDS300 phải có 300 ảnh, hiện có $count ảnh." }
Write-Host "BSDS300 sẵn sàng: $count ảnh tại $imageRoot"
