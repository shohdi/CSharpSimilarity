[CmdletBinding()]
param(
    [string]$BaseUrl = 'http://localhost:5000',
    [switch]$StartServer,
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Debug'
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Net.Http
$BaseUrl = $BaseUrl.TrimEnd('/')
$server = $null
$serverOutput = $null
$serverError = $null
$client = [Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds(120)

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Get-ComparisonUrl([string]$First, [string]$Second) {
    return $BaseUrl + '/IsSame?first_name=' + [Uri]::EscapeDataString($First) +
        '&second_name=' + [Uri]::EscapeDataString($Second)
}

function Read-Comparison($Response, [string]$First, [string]$Second) {
    try {
        Assert-True ($Response.StatusCode -eq [Net.HttpStatusCode]::OK) 'Comparison request did not return HTTP 200.'
        $body = $Response.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
        $fields = @($body.PSObject.Properties.Name)
        Assert-True ($fields.Count -eq 3 -and $fields -ccontains 'Percentage' -and
            $fields -ccontains 'first_name' -and $fields -ccontains 'second_name') 'Response fields do not match the API contract.'
        Assert-True ($body.first_name -ceq $First) 'The first name was not echoed exactly.'
        Assert-True ($body.second_name -ceq $Second) 'The second name was not echoed exactly.'
        Assert-True ($body.Percentage -is [string]) 'Percentage must be a JSON string.'
        Assert-True ($body.Percentage -match '^([0-9]|[1-9][0-9]|100)$') 'Percentage is not an integer string between 0 and 100.'
        return [int]$body.Percentage
    }
    finally { $Response.Dispose() }
}

function Compare-Names([string]$First, [string]$Second) {
    $response = $client.GetAsync((Get-ComparisonUrl $First $Second)).GetAwaiter().GetResult()
    return Read-Comparison $response $First $Second
}

function Assert-BadRequest([string]$Path) {
    $response = $client.GetAsync($BaseUrl + $Path).GetAwaiter().GetResult()
    try {
        Assert-True ($response.StatusCode -eq [Net.HttpStatusCode]::BadRequest) "Expected HTTP 400 for invalid input: $Path"
    }
    finally { $response.Dispose() }
}

try {
    if ($StartServer) {
        $projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
        $dll = Join-Path $projectDirectory "bin/$Configuration/net8.0/CSharpSimilarity.dll"
        Assert-True (Test-Path -LiteralPath $dll -PathType Leaf) "Build first with: dotnet build --configuration $Configuration"
        $serverOutput = Join-Path ([IO.Path]::GetTempPath()) ('CSharpSimilarity-' + [Guid]::NewGuid().ToString('N') + '.stdout.log')
        $serverError = Join-Path ([IO.Path]::GetTempPath()) ('CSharpSimilarity-' + [Guid]::NewGuid().ToString('N') + '.stderr.log')
        $startOptions = @{
            FilePath = 'dotnet'
            ArgumentList = @(('"' + $dll + '"'), '--urls', ('"' + $BaseUrl + '"'))
            WorkingDirectory = $projectDirectory
            RedirectStandardOutput = $serverOutput
            RedirectStandardError = $serverError
            PassThru = $true
        }
        if ($env:OS -eq 'Windows_NT') { $startOptions.WindowStyle = 'Hidden' }
        $server = Start-Process @startOptions
    }

    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            $response = $client.GetAsync($BaseUrl + '/health').GetAwaiter().GetResult()
            try { $ready = $response.IsSuccessStatusCode }
            finally { $response.Dispose() }
        }
        catch { $ready = $false }
        if ($ready) { break }
        if ($null -ne $server -and $server.HasExited) { throw 'API process exited before becoming healthy.' }
        Start-Sleep -Seconds 1
    }
    Assert-True $ready "API was not healthy at $BaseUrl."

    $example = Compare-Names 'shohdi' 'شهدي'
    Assert-True ($example -eq 99) 'The Arabic/English example should score 99.'
    Assert-True ((Compare-Names 'شهدي' 'shohdi') -eq $example) 'The score should be symmetric.'
    Assert-True ((Compare-Names 'محمد علي' 'محمد علي') -eq 100) 'Identical names should score 100.'
    Assert-True ((Compare-Names 'mohamed ali' 'محمد علي') -ge 85) 'A common transliteration should be a strong match.'
    Assert-True ((Compare-Names 'Ahmed Hassan' 'Victoria Brown') -lt 50) 'Clearly different names should have a low score.'

    Assert-BadRequest '/IsSame'
    Assert-BadRequest '/IsSame?first_name=Ahmed'
    Assert-BadRequest '/IsSame?first_name=&second_name=Ahmed'
    Assert-BadRequest '/IsSame?first_name=Ahmed&first_name=Ali&second_name=Ahmed'
    Assert-BadRequest '/IsSame?first_name=%21%21%21&second_name=Ahmed'
    Assert-BadRequest ('/IsSame?first_name=' + ('a' * 257) + '&second_name=Ahmed')
    # 127 one-token words fit the character limit but exceed CLS/SEP + 126 name tokens.
    Assert-BadRequest ('/IsSame?first_name=' + [Uri]::EscapeDataString(('a ' * 126) + 'a') + '&second_name=Ahmed')

    # All requests are started before awaiting results to exercise concurrent inference.
    $pairs = @(
        @('shohdi', 'شهدي'),
        @('Ahmed Hassan', 'Victoria Brown'),
        @('محمد علي', 'محمد علي'),
        @('شهدي', 'shohdi')
    )
    $pending = @()
    foreach ($pair in $pairs) { $pending += $client.GetAsync((Get-ComparisonUrl $pair[0] $pair[1])) }
    for ($index = 0; $index -lt $pending.Count; $index++) {
        $score = Read-Comparison ($pending[$index].GetAwaiter().GetResult()) $pairs[$index][0] $pairs[$index][1]
        if ($index -eq 0 -or $index -eq 3) { Assert-True ($score -eq $example) 'Concurrent responses were inconsistent.' }
    }
    Write-Host "Smoke tests passed. shohdi / Arabic equivalent: $example%."
}
catch {
    if ($null -ne $server) {
        foreach ($log in @($serverOutput, $serverError)) {
            if (Test-Path -LiteralPath $log -PathType Leaf) { Get-Content -LiteralPath $log | Write-Host }
        }
    }
    throw
}
finally {
    $client.Dispose()
    if ($null -ne $server) {
        if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
        $server.WaitForExit()
        $server.Dispose()
    }
    foreach ($log in @($serverOutput, $serverError)) {
        if ($null -ne $log -and (Test-Path -LiteralPath $log -PathType Leaf)) {
            Remove-Item -LiteralPath $log -Force -ErrorAction SilentlyContinue
        }
    }
}
