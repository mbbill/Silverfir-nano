param(
    [string] $Port = "COM4",
    [int] $Baud = 115200,
    [int] $Seconds = 0,
    [switch] $NoReset,
    [switch] $FpsOnly
)

$ErrorActionPreference = "Stop"

$Serial = [System.IO.Ports.SerialPort]::new($Port, $Baud, [System.IO.Ports.Parity]::None, 8, [System.IO.Ports.StopBits]::One)
$Serial.ReadTimeout = 200
$Serial.DtrEnable = $false
$Serial.RtsEnable = -not $NoReset

function Write-MonitorText {
    param([string] $Text)

    if (-not $FpsOnly) {
        [Console]::Write($Text)
        return
    }

    $script:Pending += $Text
    while ($script:Pending -match "^(.*?)(\r?\n)(.*)$") {
        $Line = $Matches[1]
        $script:Pending = $Matches[3]
        if ($Line -match "(^fps mode=|\sfps\s)") {
            Write-Output $Line
        }
    }
}

$script:Pending = ""
$Deadline = $null
if ($Seconds -gt 0) {
    $Deadline = (Get-Date).AddSeconds($Seconds)
}

try {
    $Serial.Open()
    while ($true) {
        if ($Deadline -ne $null -and (Get-Date) -ge $Deadline) {
            break
        }

        $Text = $Serial.ReadExisting()
        if ($Text.Length -gt 0) {
            Write-MonitorText $Text
        }
        Start-Sleep -Milliseconds 50
    }
} finally {
    if ($Serial.IsOpen) {
        $Serial.Close()
    }
}
