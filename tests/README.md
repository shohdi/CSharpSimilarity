Run the permanent checks from the project folder:

```powershell
dotnet run --project tests/CSharpSimilarity.Checks.csproj
```

The runner uses only .NET and local files, with no Python, Node, HTTP calls, or
additional NuGet packages. It checks independent tokenizer fixtures, token limits,
normalization, positive and negative transliteration examples, symmetry, unmatched
token coverage, organization names, and both trained Dense projections. The
DistilUSE assets must already exist in `Models/`, and LaBSE assets in `Models/LaBSE/`.

See `tokenizer-fixtures.md` for tokenizer fixture provenance.
`labse-tokenizer-fixtures.json` was captured independently using Hugging Face
Transformers.js 3.8.1's `BertTokenizer` against the pinned local LaBSE tokenizer,
with special tokens enabled and padding and truncation disabled. Its source is
[Xenova/LaBSE revision 147002d501fc76ecd313c631999ea796520fea74](https://huggingface.co/Xenova/LaBSE/tree/147002d501fc76ecd313c631999ea796520fea74).
The corpus includes company and person names, casing, Arabic/English aliases, and
a 246-character bilingual field with 135 tokens. The latter has independently
valid aliases of 116 and 21 tokens; it verifies that alias selection cannot bypass
the service's 128-token limit for the complete original field.

`dense-fixture.json` was captured using an independent JavaScript implementation
that reads the pinned original `2_Dense/model.safetensors` with Node's little-endian
`Buffer` methods, applies the 512 by 768 weight matrix and bias, then tanh and L2
normalization. Its fixed input is `((i * 37) % 101 - 50) / 64`, for `i = 0..767`.
The comparison allows `1e-6` absolute error for float accumulation differences.

`labse-dense-fixture.json` provides an independent reference for LaBSE's trained
768 by 768 Dense matrix, bias, tanh, and L2 normalization. Its synthetic CLS input
uses the same fixed formula; it is not a recorded transformer output. The fixture
was generated using JavaScript's little-endian `Buffer` reads and double-precision
arithmetic, independently of the C# projection implementation. Generation verified
the SHA256 of the original
[`2_Dense/model.safetensors`](https://huggingface.co/sentence-transformers/LaBSE/resolve/836121a0533e5664b21c7aacc5d22951f2b8b25b/2_Dense/model.safetensors):
`f866c945fa2147ce4940c8ead0b4b20af4217e0f4664bd35f41c2a0967c4fe77`.
The runner checks all 768 output coordinates with `1e-6` absolute tolerance, output
length, unit norm, and rejection of an invalid input shape. It requires no
JavaScript or Python runtime to execute these checks.

The examples check selected regressions. They are not a calibrated evaluation
dataset and do not establish a probability that two names identify one person.
