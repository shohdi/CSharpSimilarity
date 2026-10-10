[CmdletBinding()]
param([switch]$IncludeTestAssets)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# Windows PowerShell 5.1 may otherwise negotiate an older TLS version.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$modelDirectory = Join-Path $projectDirectory 'Models'
[IO.Directory]::CreateDirectory($modelDirectory) | Out-Null

$xenovaRevision = '147002d501fc76ecd313c631999ea796520fea74'
$upstreamRevision = '836121a0533e5664b21c7aacc5d22951f2b8b25b'
$assets = @(
    @{
        Name = 'LaBSE/model_quantized.onnx'
        Url = "https://huggingface.co/Xenova/LaBSE/resolve/$xenovaRevision/onnx/model_quantized.onnx?download=true"
        Sha256 = '2d209704455c8e5718dbef9c46ca22c823eb35256b74b4122d90845310c7ef82'
    },
    @{
        Name = 'LaBSE/tokenizer.json'
        Url = "https://huggingface.co/Xenova/LaBSE/resolve/$xenovaRevision/tokenizer.json?download=true"
        Sha256 = '09216b42d2697b7b4a26ac05ff09ba8bf52dc19b896c5ceee8bbff9f39055322'
    },
    @{
        Name = 'LaBSE/dense.safetensors'
        Url = "https://huggingface.co/sentence-transformers/LaBSE/resolve/$upstreamRevision/2_Dense/model.safetensors?download=true"
        Sha256 = 'f866c945fa2147ce4940c8ead0b4b20af4217e0f4664bd35f41c2a0967c4fe77'
    }
)

# Historical independent fixtures still exercise the shared tokenizer and
# generic Dense reader. These small reference assets are never deployed.
if ($IncludeTestAssets) {
    $assets += @(
        @{
            Name = 'tokenizer.json'
            Url = 'https://huggingface.co/Xenova/distiluse-base-multilingual-cased-v2/resolve/cad454171d918d9873a2701ba245054b6c1760dd/tokenizer.json?download=true'
            Sha256 = 'bf1b59b7b11c95f194f51708d918eea378e09d05f84c0e1656dc5180e8117088'
        },
        @{
            Name = 'dense.safetensors'
            Url = 'https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2/resolve/bfe45d0732ca50787611c0fe107ba278c7f3f889/2_Dense/model.safetensors?download=true'
            Sha256 = '0a21b1ce908e772ebf09f93c20ca09524c32706e9918d9c0169a3f0663b191ed'
        }
    )
}

foreach ($asset in $assets) {
    $destination = Join-Path $modelDirectory $asset.Name
    $assetDirectory = [IO.Path]::GetDirectoryName($destination)
    [IO.Directory]::CreateDirectory($assetDirectory) | Out-Null
    if ((Test-Path -LiteralPath $destination -PathType Leaf) -and
        ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -eq $asset.Sha256)) {
        Write-Host "Verified $($asset.Name); download skipped."
        continue
    }

    # Verify the temporary file before atomically replacing the destination.
    $temporary = Join-Path $assetDirectory ('.' + [IO.Path]::GetFileName($asset.Name) + '.' + [Guid]::NewGuid().ToString('N') + '.tmp')
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
