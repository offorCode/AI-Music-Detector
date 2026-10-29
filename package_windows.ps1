$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
$dist = Join-Path $project 'dist'
$stage = Join-Path $dist '_package-stage'
$archive = Join-Path $dist 'AI-Music-Detector-source.zip'

New-Item -ItemType Directory -Force -Path $stage | Out-Null
$files = @(
    'earprint_desktop.py',
    'README.md',
    'THIRD_PARTY_NOTICES.md',
    'requirements.txt',
    'setup_windows.bat',
    'run_detector.bat',
    'OnnTech.png',
    'OnnTech.ico',
    'OnnTech.mp3'
)
foreach ($file in $files) {
    Copy-Item -LiteralPath (Join-Path $project $file) -Destination $stage -Force
}
$vendorSource = Join-Path $project 'vendor'
$vendorPackageSource = Join-Path $vendorSource 'aigm_detector'
$vendorStage = Join-Path $stage 'vendor'
New-Item -ItemType Directory -Force -Path $vendorStage | Out-Null
Copy-Item -LiteralPath (Join-Path $vendorSource 'AIGM-LICENSE') -Destination $vendorStage -Force
Get-ChildItem -LiteralPath $vendorPackageSource -File -Recurse |
    Where-Object { $_.Extension -ne '.pyc' } |
    ForEach-Object {
        $relative = $_.FullName.Substring($vendorPackageSource.Length).TrimStart([char[]]@('\', '/'))
        $target = Join-Path (Join-Path $vendorStage 'aigm_detector') $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $target -Force
    }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $archive -Force
Write-Output "Created $archive"
