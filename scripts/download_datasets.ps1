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
$maximumAttempts = 3

New-Item -ItemType Directory -Force -Path $downloadDir, $externalDir | Out-Null

if (Test-Path -LiteralPath $archive) {
    $cachedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash
    if ($cachedHash -ne $expectedHash) {
        Write-Warning "Xóa archive BSDS300 cache không hợp lệ ($cachedHash)."
        Remove-Item -LiteralPath $archive -Force
    }
}

if (-not (Test-Path -LiteralPath $archive)) {
    $temporary = "$archive.part"
    $lastError = $null
    for ($attempt = 1; $attempt -le $maximumAttempts; $attempt++) {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
        try {
            Write-Host "Tải BSDS300 (lần $attempt/$maximumAttempts)..."
            Invoke-WebRequest -Uri $url -OutFile $temporary -Headers @{ 'User-Agent' = 'parallel-image-processing/1.0' }
            $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $temporary).Hash
            if ($actualHash -ne $expectedHash) {
                throw "Checksum không khớp: mong đợi $expectedHash, nhận $actualHash"
            }
            Move-Item -LiteralPath $temporary -Destination $archive -Force
            $lastError = $null
            break
        } catch {
            $lastError = $_
            Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
            if ($attempt -lt $maximumAttempts) {
                Start-Sleep -Seconds ([math]::Pow(2, $attempt - 1))
            }
        }
    }
    if ($null -ne $lastError) {
        throw "Không tải được archive BSDS300 hợp lệ sau $maximumAttempts lần. $lastError"
    }
}

$datasetRoot = Join-Path $externalDir 'BSDS300'
$imageRoot = Join-Path $externalDir 'BSDS300\images'
$count = if (Test-Path -LiteralPath $imageRoot) {
    (Get-ChildItem -LiteralPath $imageRoot -Recurse -File -Filter '*.jpg').Count
} else { 0 }

if ($count -ne 300) {
    $extractionRoot = Join-Path $externalDir '.BSDS300.extracting'
    Remove-Item -LiteralPath $extractionRoot -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $extractionRoot | Out-Null
    try {
        tar -xzf $archive -C $extractionRoot
        $candidate = Join-Path $extractionRoot 'BSDS300'
        $candidateImageRoot = Join-Path $candidate 'images'
        $candidateCount = if (Test-Path -LiteralPath $candidateImageRoot) {
            (Get-ChildItem -LiteralPath $candidateImageRoot -Recurse -File -Filter '*.jpg').Count
        } else { 0 }
        if ($candidateCount -ne 300) {
            throw "BSDS300 phải có 300 ảnh, archive hiện có $candidateCount ảnh."
        }

        Remove-Item -LiteralPath $datasetRoot -Recurse -Force -ErrorAction SilentlyContinue
        Move-Item -LiteralPath $candidate -Destination $datasetRoot
        $count = $candidateCount
    } finally {
        Remove-Item -LiteralPath $extractionRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "BSDS300 sẵn sàng: $count ảnh tại $imageRoot"
