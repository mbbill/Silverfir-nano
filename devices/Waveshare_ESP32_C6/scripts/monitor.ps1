param(
    [string] $Port = "COM4",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $MonitorArgs
)

$ErrorActionPreference = "Stop"

$ArgsList = @("monitor", "--port", $Port)
$ArgsList += $MonitorArgs

espflash @ArgsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
