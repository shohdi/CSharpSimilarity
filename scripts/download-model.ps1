[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# Windows PowerShell 5.1 may otherwise negotiate an older TLS version.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$modelDirectory = Join-Path $projectDirectory 'Models'
[IO.Directory]::CreateDirectory($modelDirectory) | Out-Null

$xenovaRevision = 'cad454171d918d9873a2701ba245054b6c1760dd'
$upstreamRevision = 'bfe45d0732ca50787611c0fe107ba278c7f3f889'
$assets = @(
    @{
        Name = 'model_quantized.onnx'
        Url = "https://huggingface.co/Xenova/distiluse-base-multilingual-cased-v2/resolve/$xenovaRevision/onnx/model_quantized.onnx?download=true"
        Sha256 = '1724c10ba3b33b58afb6ce73bbf3e0528921bd4d4e1bf5a569b6d73f234e72bf'
    },
    @{
        Name = 'tokenizer.json'
        Url = "https://huggingface.co/Xenova/distiluse-base-multilingual-cased-v2/resolve/$xenovaRevision/tokenizer.json?download=true"
        Sha256 = 'bf1b59b7b11c95f194f51708d918eea378e09d05f84c0e1656dc5180e8117088'
    },
    @{
        Name = 'dense.safetensors'
        Url = "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2/resolve/$upstreamRevision/2_Dense/model.safetensors?download=true"
        Sha256 = '0a21b1ce908e772ebf09f93c20ca09524c32706e9918d9c0169a3f0663b191ed'
    }
)

foreach ($asset in $assets) {
    $destination = Join-Path $modelDirectory $asset.Name
    if ((Test-Path -LiteralPath $destination -PathType Leaf) -and
        ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -eq $asset.Sha256)) {
        Write-Host "Verified $($asset.Name); download skipped."
        continue
    }

    # Verify the temporary file before atomically replacing the destination.
    $temporary = Join-Path $modelDirectory ('.' + $asset.Name + '.' + [Guid]::NewGuid().ToString('N') + '.tmp')
    try {
        Write-Host "Downloading $($asset.Name)..."
        Invoke-WebRequest -Uri $asset.Url -OutFile $temporary -UseBasicParsing -TimeoutSec 600
        $actualHash = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash
        if ($actualHash -ne $asset.Sha256) {
            throw "SHA256 verification failed for $($asset.Name). Expected $($asset.Sha256), received $actualHash."
        }
        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            [IO.File]::Replace($temporary, $destination, $null)
        }
        else {
            [IO.File]::Move($temporary, $destination)
        }
        Write-Host "Verified and installed $($asset.Name)."
    }
    finally {
        if (Test-Path -LiteralPath $temporary -PathType Leaf) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

Write-Host 'All model assets are ready. Runtime inference does not require internet access.'
