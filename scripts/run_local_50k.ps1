param(
    [string]$RunId = "en_de_50k",
    [string]$ProcessedDir = "data\processed\en_de_50k",
    [string]$ArtifactsDir = "artifacts\en_de_50k",
    [string]$NormalizedSource = "data\processed\en_de\all_trainable.jsonl",
    [int]$HeadTrainJobs = 8
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing virtual environment Python at $Python"
}

$ArtifactsPath = Join-Path $RepoRoot $ArtifactsDir
$LogsPath = Join-Path $ArtifactsPath "logs"
New-Item -ItemType Directory -Force -Path $LogsPath | Out-Null

$Stdout = Join-Path $LogsPath "launcher.stdout.log"
$Stderr = Join-Path $LogsPath "launcher.stderr.log"
$PidPath = Join-Path $LogsPath "launcher.pid"
$LockPath = Join-Path $LogsPath "run.lock"

if (Test-Path $LockPath) {
    $Lock = Get-Content $LockPath -Raw | ConvertFrom-Json
    $Existing = Get-Process -Id ([int]$Lock.pid) -ErrorAction SilentlyContinue
    if ($Existing) {
        "Run $RunId is already active with PID $($Existing.Id)."
        "status: .venv\Scripts\python.exe -m lucid_decoders.tools.local_status --run-id $RunId --processed-dir $ProcessedDir --artifacts-dir $ArtifactsDir"
        exit 0
    }
}

$Args = @(
    "-m", "lucid_decoders.tools.local_run",
    "--run-id", $RunId,
    "--processed-dir", $ProcessedDir,
    "--artifacts-dir", $ArtifactsDir,
    "--normalized-source", $NormalizedSource,
    "--device", "cuda",
    "--chunk-size", "250",
    "--seed", "13",
    "--train-per-label", "24037",
    "--validation-per-label", "758",
    "--test-per-label", "205",
    "--head-train-jobs", "$HeadTrainJobs",
    "--persist-head-models", "best"
)

$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $Args `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -PassThru `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr

$Process.Id | Set-Content -Path $PidPath -Encoding UTF8
"Started $RunId with PID $($Process.Id)"
"stdout: $Stdout"
"stderr: $Stderr"
"status: .venv\Scripts\python.exe -m lucid_decoders.tools.local_status --run-id $RunId --processed-dir $ProcessedDir --artifacts-dir $ArtifactsDir"
