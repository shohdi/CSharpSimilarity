Run the permanent checks from the project folder:

```powershell
dotnet run --project tests/CSharpSimilarity.Checks.csproj
```

The runner uses only .NET and local files, with no Python, Node, HTTP calls, or
additional NuGet packages. It checks independent tokenizer fixtures, token limits,
normalization, positive and negative transliteration examples, symmetry, unmatched
token coverage, and the trained Dense projection. Model files must already exist
in `Models/`.

See `tokenizer-fixtures.md` for tokenizer fixture provenance.
`dense-fixture.json` was captured using an independent JavaScript implementation
that reads the pinned original `2_Dense/model.safetensors` with Node's little-endian
`Buffer` methods, applies the 512 by 768 weight matrix and bias, then tanh and L2
normalization. Its fixed input is `((i * 37) % 101 - 50) / 64`, for `i = 0..767`.
The comparison allows `1e-6` absolute error for float accumulation differences.

The examples check selected regressions. They are not a calibrated evaluation
dataset and do not establish a probability that two names identify one person.
