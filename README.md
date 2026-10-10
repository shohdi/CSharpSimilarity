# Arabic and English name similarity API

A .NET 8 C# Web API that compares person or entity names in Arabic and English. Inference runs on the local CPU with ONNX Runtime. It needs no Python, GPU, cloud service, Hugging Face account, or network connection at runtime.

The API combines the multilingual Hugging Face [LaBSE model](https://huggingface.co/sentence-transformers/LaBSE) with Arabic normalization, transliteration-aware person matching, and organization matching that gives distinctive brand words more weight than generic company terms. `Percentage` is a **name similarity score**, not a calibrated probability that two records identify the same person or organization. Identical names can belong to different people; reliable identity resolution needs additional attributes and validation against your own labeled data.

## Run locally

Install the [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0), then run these commands from this folder in Windows PowerShell or PowerShell 7:

```powershell
.\scripts\download-model.ps1
dotnet run --no-launch-profile --urls http://localhost:5000
```

If Windows blocks local scripts, run `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download-model.ps1`. This changes the policy only for that process; the same prefix works for the smoke test below.

The first setup downloads about 488 MB of pinned LaBSE model files into `Models/LaBSE` and verifies their SHA256 checksums. NuGet restore also needs internet access initially. After setup, the application uses only local files. Startup fails with an actionable error if a required model asset is missing or invalid.

Call the endpoint with URL-encoded names:

```powershell
$first = [Uri]::EscapeDataString('shohdi')
$second = [Uri]::EscapeDataString('شهدي')
Invoke-RestMethod "http://localhost:5000/IsSame?first_name=$first&second_name=$second"
```

```json
{
  "first_name": "shohdi",
  "second_name": "شهدي",
  "Percentage": "99"
}
```

`Percentage` is returned as a string containing an integer from `0` through `100`. The original query values are echoed exactly. Scores are symmetric when the names are swapped.

For person comparisons, normalized identical names score `100`; strong Arabic/English phonetic equivalents and complete matches against verified bilingual aliases score `99`. Other person pairs use `85%` spelling/transliteration similarity plus `15%` positive embedding cosine similarity squared, rounded and capped at `98`. These weights are heuristics for short names and can be tuned after evaluation on your data. The Hugging Face model runs for every valid pair, using original casing for the selected complete names.

Matching tolerates reordered words, common transliteration variants, and detached `Al`/`El` articles. It penalizes missing words and different numeric identifiers. A finite dictionary aligns organization types and sector terms across Arabic and English. The model supplies additional semantic evidence for translated organization names.

## Screening organization records

Supply the name being screened as `first_name` and the name from the Dow Jones record as `second_name`. The API compares the supplied text locally; your application provides the record value. For example:

```powershell
$screenedName = [Uri]::EscapeDataString('Cavalier Dynamics Technologies')
$recordName = [Uri]::EscapeDataString('Cavalier Dynamics for Technologies Company شركة كافاليير دايناميكس للتقنيات')
Invoke-RestMethod "http://localhost:5000/IsSame?first_name=$screenedName&second_name=$recordName"
```

Organization matching separates distinctive brand words from connectors, legal suffixes, and sector terms. Complete English/Arabic forms are treated as organization aliases after checking that their brands, sectors, and identifiers agree. Brand similarity and LaBSE semantic evidence contribute to the comparison. Shared words such as `Company` or `Technologies` do not make different brands a strong match. Different explicit sectors or numeric identifiers retain a mismatch penalty.

The current regression examples return `98` for the complete bilingual record above and `88` when its Arabic form alone is compared with `Cavalier Dynamics Technologies`. These are observed example scores, not accuracy measurements for a screening dataset.

## Bilingual record names

A record may contain the complete English and Arabic forms of the same name in one field. The API compares these as equivalent alternatives after checking that the two complete forms agree, rather than penalizing the record for repeating its name in another script. For example:

```text
first_name=Shohdy Mohamed Hussien ElSheemy شهدي محمد حسين الشيمي
second_name=شهدي محمد حسين الشيمي
Percentage=99
```

The reverse comparison and matching that bilingual field against its English form also score `99`. Encode each entire query value with `Uri.EscapeDataString` as shown above.

Person alias recognition requires two complete Arabic/Latin script runs, equal name-component counts after article handling, matching numeric identifiers, and a strong match for every assigned component. Partial translations, conflicting names, or unexplained extra components retain the full-field mismatch penalty. Adding a different name beside a matching name does not automatically produce `99`. Unrelated aliases of the same entity cannot be inferred from unstructured text; keep distinct source aliases as separate comparisons when they do not represent equivalent full names.

Model choice and score thresholds should be evaluated against labeled matching and nonmatching records from your own data. LaBSE is the default multilingual model for this implementation; the examples and regression tests do not establish an AML screening error rate or prove that one model is best for every dataset.

## Endpoints and validation

| Request | Behavior |
| --- | --- |
| `GET /IsSame?first_name=...&second_name=...` | Compare two names and return the JSON contract above. |
| `GET /health` | Report whether the local inference service is ready. |
| `GET /` | Show service and model information. |

Each name must contain a letter or digit, be at most 256 characters, and fit within the model's 128-token input limit. Missing, blank, punctuation-only, duplicate parameters, or over-limit values return HTTP `400`. The application does not log submitted names or query strings. Because the requested endpoint uses GET, configure any external proxy or access logger to avoid storing query strings too.

## Verify and publish

```powershell
.\scripts\download-model.ps1 -IncludeTestAssets
dotnet build
.\scripts\smoke-test.ps1 -StartServer
dotnet run --project tests/CSharpSimilarity.Checks.csproj --configuration Release
```

To test an already running API, use `.\scripts\smoke-test.ps1 -BaseUrl http://localhost:5000`. The smoke checks cover person and organization matching, conflicting aliases and identifiers, response formatting, symmetry, invalid input, and concurrent requests. `-IncludeTestAssets` also downloads the small legacy tokenizer and dense-projection reference assets used by the independent C# fixtures; these assets are separate from the deployed LaBSE model.

```powershell
dotnet publish --configuration Release --output publish
dotnet .\publish\CSharpSimilarity.dll --urls http://localhost:5000
```

Model files are copied into the publish output. Deploy that complete output to run offline. The included GitHub Actions workflow downloads the pinned assets, builds on Linux, and runs the API smoke tests.

Settings and model paths are resolved beside the executable, so the published API also works when launched from another working directory. Set `Similarity__ModelDirectory` to an absolute path if you keep the model assets elsewhere.

## Model provenance

The default model is [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE). Its model card describes support for 109 languages, including Arabic and English. The quantized ONNX encoder and tokenizer come from [Xenova's ONNX conversion](https://huggingface.co/Xenova/LaBSE). C# performs WordPiece tokenization, selects the CLS representation, applies the model's trained `768 → 768` dense layer with tanh, normalizes the embedding to unit length, and computes cosine similarity locally.

Pinned revisions, download links, file sizes, and SHA256 checksums are documented in [Models/README.md](Models/README.md). The application code is licensed under [MIT](LICENSE). Model assets retain their upstream [Apache 2.0 license](https://huggingface.co/sentence-transformers/LaBSE). Large model files are excluded from Git; the download script makes setup reproducible.
