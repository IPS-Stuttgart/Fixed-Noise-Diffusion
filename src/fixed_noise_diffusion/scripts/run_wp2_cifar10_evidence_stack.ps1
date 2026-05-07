[CmdletBinding()]
param(
    [string]$Python = "py",
    [string[]]$PythonArgs = @("-3.12"),
    [string]$OutputRoot = "runs/wp2_cifar10_pool_scaling_100ep",
    [string]$DataDir = "data",
    [switch]$DownloadData,
    [switch]$SaveCheckpoints,
    [switch]$NoPlot
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = if ($env:PYTHONPATH) { $env:PYTHONPATH } else { "src" }

$PoolSizes = @(250, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
$Seeds = @(0, 1, 2)

function Format-Bool([bool]$Value) {
    if ($Value) { return "true" }
    return "false"
}

function Format-PoolLabel([int]$PoolSize) {
    if (($PoolSize -ge 1000) -and (($PoolSize % 1000) -eq 0)) {
        return "$(($PoolSize / 1000))k"
    }
    return "$PoolSize"
}

function Invoke-PaperCifarTrain {
    param(
        [Parameter(Mandatory=$true)][string]$RunName,
        [Parameter(Mandatory=$true)][int]$Seed,
        [Parameter(Mandatory=$true)][string]$NoiseMode,
        [Nullable[int]]$PoolSize = $null,
        [bool]$Whiten = $false
    )

    $trainArgs = @(
        "-m", "fixed_noise_diffusion.train",
        "--config", "cifar10_base.yaml",
        "--set", "run_name=$RunName",
        "--set", "output_dir=$OutputRoot",
        "--set", "seed=$Seed",
        "--set", "data.data_dir=$DataDir",
        "--set", "data.download=$(Format-Bool $DownloadData.IsPresent)",
        "--set", "diffusion.beta_schedule=cosine",
        "--set", "training.epochs=100",
        "--set", "training.checkpoint_epochs=[1,5,10,25,50,100]",
        "--set", "training.save_checkpoint=$(Format-Bool $SaveCheckpoints.IsPresent)",
        "--set", "evaluation.enable_metrics=false",
        "--set", "noise.mode=$NoiseMode",
        "--set", "noise.whiten=$(Format-Bool $Whiten)"
    )

    if ($PoolSize.HasValue) {
        $trainArgs += @("--set", "noise.pool_size=$($PoolSize.Value)")
    } else {
        $trainArgs += @("--set", "noise.pool_size=null")
    }

    Write-Host "==> $Python $($PythonArgs -join ' ') $($trainArgs -join ' ')"
    & $Python @PythonArgs @trainArgs
}

foreach ($Seed in $Seeds) {
    Invoke-PaperCifarTrain `
        -RunName "wp2_100ep_cifar10_gaussian_seed$Seed" `
        -Seed $Seed `
        -NoiseMode "gaussian"
}

foreach ($Seed in $Seeds) {
    foreach ($PoolSize in $PoolSizes) {
        $PoolLabel = Format-PoolLabel $PoolSize
        Invoke-PaperCifarTrain `
            -RunName "wp2_100ep_cifar10_fixed_pool_${PoolLabel}_seed$Seed" `
            -Seed $Seed `
            -NoiseMode "fixed_pool" `
            -PoolSize $PoolSize
    }
}

foreach ($Seed in $Seeds) {
    foreach ($PoolSize in $PoolSizes) {
        $PoolLabel = Format-PoolLabel $PoolSize
        Invoke-PaperCifarTrain `
            -RunName "wp2_100ep_cifar10_fixed_pool_whitened_${PoolLabel}_seed$Seed" `
            -Seed $Seed `
            -NoiseMode "fixed_pool_whitened" `
            -PoolSize $PoolSize `
            -Whiten $true
    }
}

if (-not $NoPlot.IsPresent) {
    $RunDirs = Get-ChildItem -Path $OutputRoot -Directory | Sort-Object FullName | ForEach-Object { $_.FullName }
    if ($RunDirs.Count -gt 0) {
        & $Python @PythonArgs -m fixed_noise_diffusion.plot_results --runs @RunDirs --output "$OutputRoot/wp2_cifar10_pool_scaling_100ep.png"
    }
}
