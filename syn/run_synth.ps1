# Run Yosys Synthesis for VoltTensor
$ErrorActionPreference = "Stop"

# Load OSS-CAD-Suite environment if yosys is not in global PATH
if (-not (Get-Command yosys -ErrorAction SilentlyContinue)) {
    $envPath = "C:\Users\surya\.oss-cad-suite\oss-cad-suite\environment.ps1"
    if (Test-Path $envPath) {
        . $envPath
    }
}

Write-Host "Running Yosys Logic Synthesis for volttensor_top..."
yosys -s syn/synth.ys | Tee-Object -FilePath syn/synthesis_report.txt
Write-Host "Synthesis complete. Report written to syn/synthesis_report.txt."
