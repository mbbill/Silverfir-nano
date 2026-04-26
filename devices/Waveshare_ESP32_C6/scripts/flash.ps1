param(
    [string] $Port = "COM4",
    [switch] $NoMonitor,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $CargoArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."

$ArgsList = @("espflash", "flash", "--chip", "esp32c6", "--port", $Port, "--flash-size", "8mb")
if (-not $NoMonitor) {
    $ArgsList += "--monitor"
}
$ArgsList += $CargoArgs

Push-Location $ProjectRoot
try {
    cargo @ArgsList
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
