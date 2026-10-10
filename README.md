# Arabic and English name similarity API

A .NET 8 C# Web API that compares person or entity names in Arabic and English. Inference runs on the local CPU with ONNX Runtime. It needs no Python, GPU, cloud service, Hugging Face account, or network connection at runtime.

The API combines a multilingual Hugging Face embedding model with Arabic normalization and transliteration-aware name matching. `Percentage` is a **name similarity score**, not a calibrated probability that two records identify the same person or organization. Identical names can belong to different people; reliable identity resolution needs additional attributes and validation against your own labeled data.

## Run locally

Install the [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0), then run these commands from this folder in Windows PowerShell or PowerShell 7:

```powershell
.\scripts\download-model.ps1
dotnet run --no-launch-profile --urls http://localhost:5000
```

If Windows blocks local scripts, run `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download-model.ps1`. This changes the policy only for that process; the same prefix works for the smoke test below.

The first setup downloads about 140 MB of pinned model files and verifies their SHA256 checksums. NuGet restore also needs internet access initially. After setup, the application uses only local files. Startup fails with an actionable error if a required model asset is missing or invalid.

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

Normalized identical names score `100`; strong Arabic/English phonetic equivalents and complete matches against verified bilingual aliases score `99`. Other pairs use `85%` spelling/transliteration similarity plus `15%` positive embedding cosine similarity squared, rounded and capped at `98`. These weights are heuristics for short names and can be tuned after evaluation on your data. The Hugging Face model runs for every valid pair, using original casing for the selected complete names.

Matching tolerates reordered words, common transliteration variants, and detached `Al`/`El` articles. It penalizes missing words and different numeric identifiers. A finite dictionary aligns organization types such as `Company` / `شركة`, `Bank` / `بنك`, and `University` / `جامعة`; it does not provide arbitrary translation or verified organization aliases.

## Bilingual record names

A record may contain the complete English and Arabic forms of the same name in one field. The API compares these as equivalent alternatives after checking that the two complete forms agree, rather than penalizing the record for repeating its name in another script. For example:

```text
first_name=Shohdy Mohamed Hussien ElSheemy شهدي محمد حسين الشيمي
second_name=شهدي محمد حسين الشيمي
Percentage=99
```

The reverse comparison and matching that bilingual field against its English form also score `99`. Encode each entire query value with `Uri.EscapeDataString` as shown above.

Alias recognition requires two complete Arabic/Latin script runs, equal name-component counts after article handling, matching numeric identifiers, and a strong match for every assigned component. Partial translations, conflicting names, or unexplained extra components retain the full-field mismatch penalty. This is not an unrestricted substring match: adding a different name beside a matching name does not automatically produce `99`. Unrelated aliases of the same entity cannot be inferred from unstructured text; keep distinct source aliases as separate comparisons when they do not represent equivalent full names.

The existing sentence model remains a secondary signal. Model choice and score thresholds should be evaluated against labeled matching and nonmatching records from your own data; the examples and regression tests do not establish an AML screening error rate.

## Endpoints and validation

| Request | Behavior |
| --- | --- |
| `GET /IsSame?first_name=...&second_name=...` | Compare two names and return the JSON contract above. |
| `GET /health` | Report whether the local inference service is ready. |
| `GET /` | Show service and model information. |

Each name must contain a letter or digit, be at most 256 characters, and fit within the model's 128-token input limit. Missing, blank, punctuation-only, duplicate parameters, or over-limit values return HTTP `400`. The application does not log submitted names or query strings. Because the requested endpoint uses GET, configure any external proxy or access logger to avoid storing query strings too.

## Verify and publish

```powershell
dotnet build
.\scripts\smoke-test.ps1 -StartServer
dotnet run --project tests/CSharpSimilarity.Checks.csproj --configuration Release
```

To test an already running API, use `.\scripts\smoke-test.ps1 -BaseUrl http://localhost:5000`. The smoke checks cover Arabic/English matching, identical and clearly different names, response formatting, symmetry, invalid input, and concurrent requests. The C# checks verify tokenization against independent Hugging Face tokenizer fixtures and the model's dense projection.

```powershell
dotnet publish --configuration Release --output publish
dotnet .\publish\CSharpSimilarity.dll --urls http://localhost:5000
```

Model files are copied into the publish output. Deploy that complete output to run offline. The included GitHub Actions workflow downloads the pinned assets, builds on Linux, and runs the API smoke tests.

Settings and model paths are resolved beside the executable, so the published API also works when launched from another working directory. Set `Similarity__ModelDirectory` to an absolute path if you keep the model assets elsewhere.

## Model provenance

The model is [sentence-transformers/distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2), which supports Arabic and English among 50 languages. The quantized ONNX encoder and tokenizer come from [Xenova's ONNX conversion](https://huggingface.co/Xenova/distiluse-base-multilingual-cased-v2). C# performs tokenization, masked mean pooling, the original dense projection, and cosine similarity locally.

| Local asset | Source revision | SHA256 |
| --- | --- | --- |
| `Models/model_quantized.onnx` | Xenova `cad454171d918d9873a2701ba245054b6c1760dd` | `1724c10ba3b33b58afb6ce73bbf3e0528921bd4d4e1bf5a569b6d73f234e72bf` |
| `Models/tokenizer.json` | Xenova `cad454171d918d9873a2701ba245054b6c1760dd` | `bf1b59b7b11c95f194f51708d918eea378e09d05f84c0e1656dc5180e8117088` |
| `Models/dense.safetensors` | Sentence Transformers `bfe45d0732ca50787611c0fe107ba278c7f3f889` | `0a21b1ce908e772ebf09f93c20ca09524c32706e9918d9c0169a3f0663b191ed` |

The application code is licensed under [MIT](LICENSE). Model assets retain their upstream [Apache 2.0 license](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2). Large model files are excluded from Git; the download script makes setup reproducible.
