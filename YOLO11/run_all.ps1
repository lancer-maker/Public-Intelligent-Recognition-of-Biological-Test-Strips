param(
    [string]$PythonExe = "D:\Anaconda_3\envs\TestStrips_py3.9_cu128\python.exe",
    [string]$Model = "yolo11n-cls.pt",
    [int]$Imgsz = 224,
    [int]$Epochs = 100,
    [int]$Batch = 16,
    [int[]]$Seeds = @(42, 831, 2026, 7, 2691),
    [string]$Device = "0",
    [switch]$SkipPrepare,
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [switch]$NoFigures
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path $PythonExe)) {
    Write-Error "找不到 Python 解释器: $PythonExe (可用 -PythonExe 指定)"
}

Write-Host "项目根目录 : $ProjectRoot"
Write-Host "解释器     : $PythonExe"
Write-Host "模型/尺寸  : $Model @ $Imgsz"
Write-Host "随机种子   : $($Seeds -join ', ')"
Write-Host "输出目录   : outputs\YOLO11"

$cmd = @(
    (Join-Path $PSScriptRoot "run_all.py"),
    "--model", $Model,
    "--imgsz", "$Imgsz",
    "--epochs", "$Epochs",
    "--batch", "$Batch",
    "--device", $Device,
    "--seeds"
) + ($Seeds | ForEach-Object { "$_" })

if ($SkipPrepare) { $cmd += "--skip-prepare" }
if ($SkipTrain)   { $cmd += "--skip-train" }
if ($SkipEval)    { $cmd += "--skip-eval" }
if ($NoFigures)   { $cmd += "--no-figures" }

& $PythonExe @cmd
exit $LASTEXITCODE
