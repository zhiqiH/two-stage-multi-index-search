param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("dense", "file_fts", "graph_path")]
    [string]$Method,

    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Config = "config.yaml",
    [string]$Split = "final",
    [string]$Ks = "1,3,5",
    [switch]$NoEvaluate,
    [switch]$NoSaveIndexes,
    [switch]$RebuildDense
)

$ErrorActionPreference = "Stop"

function Assert-ProjectRoot {
    if (-not (Test-Path -LiteralPath $Config)) {
        throw "Run this script from the project root. Missing $Config."
    }
    if (-not (Test-Path -LiteralPath "experiments\run_experiment_box.py")) {
        throw "Run this script from the project root. Missing experiments\run_experiment_box.py."
    }
}

function Assert-FileExists {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required file is missing: $Path"
    }
}

function Invoke-BoxMethod {
    param([string]$BoxMethod)

    $args = @(
        "experiments\run_experiment_box.py",
        "--config", $Config,
        "--method", $BoxMethod,
        "--split", $Split,
        "--ks", $Ks
    )
    if ($NoEvaluate) {
        $args += "--no-evaluate"
    }
    if (-not $NoSaveIndexes) {
        $args += "--save-indexes"
    }
    if ($RebuildDense) {
        $args += "--rebuild-dense"
    }
    & $Python @args
}

function Update-Summary {
    if (-not $NoEvaluate) {
        & $Python "experiments\summarize_results.py" `
            "--config" $Config `
            "--results-dir" "results\final" `
            "--out" "results\final\main_results.csv"
    }
}

Assert-ProjectRoot

switch ($Method) {
    "file_fts" {
        Invoke-BoxMethod "file_fts"
        Assert-FileExists "results\final\file_fts_final_predictions.jsonl"
        & $Python "experiments\check_no_gold_leakage.py" "results\final\file_fts_final_predictions.jsonl"
        Update-Summary
    }
    "graph_path" {
        Invoke-BoxMethod "graph_path"
        Assert-FileExists "results\final\graph_path_final_predictions.jsonl"
        & $Python "experiments\check_no_gold_leakage.py" "results\final\graph_path_final_predictions.jsonl"
        Update-Summary
    }
    "dense" {
        if (-not $env:DEEPINFRA_TOKEN) {
            throw "DEEPINFRA_TOKEN is not set. Set it in the environment before running Dense."
        }
        Invoke-BoxMethod "dense"
        Assert-FileExists "results\final\dense_final_predictions.jsonl"
        & $Python "experiments\check_no_gold_leakage.py" "results\final\dense_final_predictions.jsonl"
        Update-Summary
    }
}
