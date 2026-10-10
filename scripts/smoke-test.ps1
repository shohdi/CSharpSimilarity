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

    $userEnglish = 'Shohdy Mohamed Hussien ElSheemy'
    $userArabic = 'شهدي محمد حسين الشيمي'
    $bilingualName = $userEnglish + ' ' + $userArabic
    $fullNameScore = Compare-Names $userEnglish $userArabic
    Assert-True ($fullNameScore -ge 90) 'The complete English/Arabic name should remain a strong match.'
    Assert-True ((Compare-Names $userArabic $userEnglish) -eq $fullNameScore) 'The complete English/Arabic score should be symmetric.'

    $bilingualArabicScore = Compare-Names $bilingualName $userArabic
    Assert-True ($bilingualArabicScore -ge 98) 'A consistent bilingual full name should match its Arabic alias.'
    Assert-True ((Compare-Names $userArabic $bilingualName) -eq $bilingualArabicScore) 'The bilingual/Arabic score should be symmetric.'
    $bilingualEnglishScore = Compare-Names $bilingualName $userEnglish
    Assert-True ($bilingualEnglishScore -ge 95) 'A consistent bilingual full name should match its English alias.'
    Assert-True ((Compare-Names $userEnglish $bilingualName) -eq $bilingualEnglishScore) 'The bilingual/English score should be symmetric.'
    Assert-True ((Compare-Names ($userArabic + ' ' + $userEnglish) $userArabic) -ge 98) 'The bilingual alias order should not reduce the match.'
    Assert-True ((Compare-Names 'Ahmed Hassan أحمد حسن' 'أحمد حسن') -ge 98) 'Bilingual alias recognition should generalize to another full name.'
    Assert-True ((Compare-Names 'Ahmed Hassan 12 أحمد حسن ١٢' 'أحمد حسن 12') -ge 98) 'Consistent numeric identifiers should survive bilingual alias selection.'

    foreach ($variant in @('Shohdy Mohamed Hussein ElSheemy', 'Shohdy Mohamed Hussien El Sheemy', 'Shohdy Mohamed Hussein Al Shimi')) {
        Assert-True ((Compare-Names $userEnglish $variant) -ge 90) 'Common English spelling/article alternatives should remain strong matches.'
    }
    Assert-True ((Compare-Names $userEnglish 'شهدي محمد حسين الحسيني') -lt 95) 'A different surname must not become a near-certain match.'
    Assert-True ((Compare-Names $userEnglish 'Ahmed Mohamed Hussien ElSheemy') -lt 95) 'A shared surname must not erase a different first name.'
    Assert-True ((Compare-Names 'Ahmed Hassan محمد علي' 'محمد علي') -lt 95) 'Contradictory concatenated aliases must not silently match only one half.'
    Assert-True ((Compare-Names 'Ahmed Hassan محمد علي' 'Ahmed Hassan') -lt 95) 'Contradictory aliases must not silently match only the English half.'
    Assert-True ((Compare-Names ($userEnglish + ' شهدي محمد حسين الحسيني') $userArabic) -lt 95) 'A conflicting surname in one bilingual half must retain its penalty.'
    Assert-True ((Compare-Names ($bilingualName + ' 12') $userArabic) -lt 95) 'An identifier in only one bilingual half must not disappear.'
    Assert-True ((Compare-Names 'Ahmed Hassan 12 أحمد حسن 13' 'أحمد حسن 13') -lt 95) 'Conflicting identifiers inside bilingual alias halves must prevent a strong match.'
    Assert-True ((Compare-Names ($userEnglish + ' شهدي محمد حسين') $userEnglish) -lt 95) 'An incomplete Arabic half must not be discarded as a complete alias.'
    Assert-True ((Compare-Names ($userEnglish + ' شهدي محمد حسين') $userArabic) -lt 95) 'An incomplete bilingual field must retain the missing-surname penalty.'
    Assert-True ((Compare-Names ($bilingualName + ' 12') ($userArabic + ' 13')) -lt 95) 'Bilingual alias handling must retain different numeric identifiers.'
    $partialNameScore = Compare-Names 'Shohdy Mohamed Hussien' $userArabic
    Assert-True ($partialNameScore -lt $fullNameScore) 'A missing surname should score below the full-name match.'

    $organizationShort = 'Cavalier Dynamics Technologies'
    $organizationEnglish = 'Cavalier Dynamics for Technologies Company'
    $organizationArabic = 'شركة كافاليير دايناميكس للتقنيات'
    $organizationRecord = $organizationEnglish + ' ' + $organizationArabic
    $organizationRecordScore = Compare-Names $organizationShort $organizationRecord
    Assert-True ($organizationRecordScore -ge 85) 'The short organization name should strongly match its complete bilingual source record.'
    Assert-True ((Compare-Names $organizationRecord $organizationShort) -eq $organizationRecordScore) 'Organization matching should be symmetric.'
    Assert-True ((Compare-Names $organizationShort $organizationEnglish) -gt 70) 'English connectors and a legal suffix should not prevent the organization match.'
    $organizationArabicScore = Compare-Names $organizationShort $organizationArabic
    Assert-True ($organizationArabicScore -gt 70) 'The Arabic-only organization name should match its English transliteration and sector.'
    Assert-True ((Compare-Names $organizationShort 'Frontier Dynamics Technologies') -lt 70) 'Shared organization terminology should not hide a different core brand.'
    Assert-True ((Compare-Names $organizationShort 'Atlantic Mercury Technologies') -lt 70) 'A shared technology sector should not match unrelated brands.'
    Assert-True ((Compare-Names $organizationShort 'Cavalier Dynamics Construction Company') -lt 90) 'A different explicit sector should retain a penalty for the same brand.'
    Assert-True ((Compare-Names ($organizationShort + ' 12') ($organizationEnglish + ' 13')) -lt 95) 'Different organization identifiers must not become a near-certain match.'
    Assert-True ((Compare-Names $organizationShort ($organizationEnglish + ' شركة فرونتير دايناميكس للتقنيات')) -lt 70) 'A contradictory Arabic brand must not disappear from the source record.'
    Assert-True ((Compare-Names 'Cavalier Dynamics Bank' ($organizationEnglish + ' شركة كافاليير دايناميكس')) -lt 90) 'An alias without a sector must retain the sector stated elsewhere in the source record.'
    Assert-True ((Compare-Names 'Cavalier for Dynamics كافاليير دايناميكس' $organizationShort) -gt 70) 'A bilingual customer name without legal or sector wording must be selectable in organization context.'

    Assert-BadRequest '/IsSame'
    Assert-BadRequest '/IsSame?first_name=Ahmed'
    Assert-BadRequest '/IsSame?first_name=&second_name=Ahmed'
    Assert-BadRequest '/IsSame?first_name=Ahmed&first_name=Ali&second_name=Ahmed'
    Assert-BadRequest '/IsSame?first_name=%21%21%21&second_name=Ahmed'
    Assert-BadRequest ('/IsSame?first_name=' + ('a' * 257) + '&second_name=Ahmed')
    # 127 one-token words fit the character limit but exceed CLS/SEP + 126 name tokens.
    Assert-BadRequest ('/IsSame?first_name=' + [Uri]::EscapeDataString(('a ' * 126) + 'a') + '&second_name=Ahmed')
    # LaBSE: each mixed-case English word has six pieces; each Arabic word one.
    # Both aliases fit separately, but the full 246-character field is 135 tokens.
    $casedBilingualLimit = (('mOhAmEd ' * 18) + 'mOhAmEd') + ' ' + (('محمد ' * 18) + 'محمد')
    Assert-BadRequest ('/IsSame?first_name=' + [Uri]::EscapeDataString($casedBilingualLimit) + '&second_name=Ahmed')

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
    Write-Host "Smoke tests passed. shohdi: $example%; full English/Arabic: $fullNameScore%; bilingual/Arabic: $bilingualArabicScore%; bilingual/English: $bilingualEnglishScore%; organization source record: $organizationRecordScore%; organization Arabic: $organizationArabicScore%."
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
