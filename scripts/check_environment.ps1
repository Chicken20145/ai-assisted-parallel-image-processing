$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$repoRoot = Split-Path -Parent $PSScriptRoot
$output = Join-Path $repoRoot 'data\external\environment.txt'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $output) | Out-Null

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("timestamp=$(Get-Date -Format o)")
$lines.Add("os=$((Get-CimInstance Win32_OperatingSystem).Caption)")
$cpu = Get-CimInstance Win32_Processor
$lines.Add("cpu=$($cpu.Name.Trim())")
$lines.Add("cpu_cores=$($cpu.NumberOfCores)")
$lines.Add("cpu_logical_processors=$($cpu.NumberOfLogicalProcessors)")
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
$lines.Add("ram_gb=$ram")
$lines.Add("python=$(python --version 2>&1)")
$lines.Add("nvcc=$((nvcc --version | Select-Object -Last 1).Trim())")
$lines.Add('nvidia_smi_begin')
$lines.AddRange([string[]](nvidia-smi))
$lines.Add('nvidia_smi_end')
$lines | Set-Content -LiteralPath $output -Encoding utf8
Write-Host "Đã ghi cấu hình môi trường vào $output"
