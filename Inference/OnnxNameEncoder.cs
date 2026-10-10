using System.Security.Cryptography;
using Microsoft.ML.OnnxRuntime;

namespace CSharpSimilarity.Inference;

public sealed class OnnxNameEncoder : IDisposable
{
    public const string ModelName = "sentence-transformers/distiluse-base-multilingual-cased-v2";
    private readonly InferenceSession _session;
    private readonly WordPieceTokenizer _tokenizer;
    private readonly DenseProjection _projection;
    private readonly SemaphoreSlim _inferenceGate = new(1, 1);

    public OnnxNameEncoder(IHostEnvironment environment, IConfiguration configuration, ILogger<OnnxNameEncoder> logger)
    {
        var configured = configuration["Similarity:ModelDirectory"] ?? "Models";
        var directory = Path.GetFullPath(configured, environment.ContentRootPath);
        var model = VerifiedFile("model_quantized.onnx", "1724c10ba3b33b58afb6ce73bbf3e0528921bd4d4e1bf5a569b6d73f234e72bf");
        var tokenizer = VerifiedFile("tokenizer.json", "bf1b59b7b11c95f194f51708d918eea378e09d05f84c0e1656dc5180e8117088");
        var dense = VerifiedFile("dense.safetensors", "0a21b1ce908e772ebf09f93c20ca09524c32706e9918d9c0169a3f0663b191ed");
        _tokenizer = new WordPieceTokenizer(tokenizer);
        _projection = new DenseProjection(dense);
        using var options = new Microsoft.ML.OnnxRuntime.SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = Math.Clamp(configuration.GetValue("Similarity:IntraOpThreads", 2), 1, Environment.ProcessorCount),
            InterOpNumThreads = 1
        };
        _session = new InferenceSession(model, options);
        if (!_session.InputMetadata.Keys.Order().SequenceEqual(new[] { "attention_mask", "input_ids" }) ||
            !_session.OutputMetadata.ContainsKey("last_hidden_state"))
            throw new InvalidDataException("The local ONNX model has unexpected inputs/outputs.");
        logger.LogInformation("Loaded multilingual name model for local CPU inference ({Dimensions} dimensions).", DenseProjection.OutputDimensions);

        string VerifiedFile(string filename, string expectedHash)
        {
            var path = Path.Combine(directory, filename);
            if (!File.Exists(path))
                throw new FileNotFoundException($"Missing local model asset '{filename}'. Run scripts/download-model.ps1 once before starting the API.", path);
            using var stream = File.OpenRead(path);
            if (!Convert.ToHexString(SHA256.HashData(stream)).Equals(expectedHash, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"Checksum mismatch for '{filename}'. Run scripts/download-model.ps1 to repair the asset.");
            return path;
        }
    }

    public void ValidateName(string name) => _ = _tokenizer.Encode(name);

    public async Task<double> CosineSimilarityAsync(string first, string second, CancellationToken cancellationToken)
    {
        var firstIds = _tokenizer.Encode(first);
        var secondIds = _tokenizer.Encode(second);
        // Bound CPU/memory use. Do not store names or embeddings between requests.
        await _inferenceGate.WaitAsync(cancellationToken);
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            return RunPair(firstIds, secondIds);
        }
        finally
        {
            _inferenceGate.Release();
        }
    }

    private double RunPair(long[] first, long[] second)
    {
        var sequenceLength = Math.Max(first.Length, second.Length);
        var ids = new long[2 * sequenceLength];
        var mask = new long[ids.Length];
        first.CopyTo(ids, 0);
        second.CopyTo(ids, sequenceLength);
        Array.Fill(mask, 1L, 0, first.Length);
        Array.Fill(mask, 1L, sequenceLength, second.Length);
        using var inputIds = OrtValue.CreateTensorValueFromMemory(ids, [2, sequenceLength]);
        using var attentionMask = OrtValue.CreateTensorValueFromMemory(mask, [2, sequenceLength]);
        var inputs = new Dictionary<string, OrtValue> { ["input_ids"] = inputIds, ["attention_mask"] = attentionMask };
        using var runOptions = new RunOptions();
        using var output = _session.Run(runOptions, inputs, ["last_hidden_state"]);
        var shape = output[0].GetTensorTypeAndShape();
        if (!shape.Shape.SequenceEqual(new long[] { 2, sequenceLength, DenseProjection.InputDimensions }))
            throw new InvalidDataException("Unexpected ONNX output shape.");
        var hidden = output[0].GetTensorDataAsSpan<float>();
        var firstPooled = new float[DenseProjection.InputDimensions];
        var secondPooled = new float[DenseProjection.InputDimensions];
        for (var row = 0; row < 2; row++)
        {
            var tokenCount = row == 0 ? first.Length : second.Length;
            var pooled = row == 0 ? firstPooled : secondPooled;
            for (var token = 0; token < tokenCount; token++)
            {
                var offset = (row * sequenceLength + token) * DenseProjection.InputDimensions;
                for (var dimension = 0; dimension < pooled.Length; dimension++)
                    pooled[dimension] += hidden[offset + dimension] / tokenCount;
            }
        }
        var firstEmbedding = _projection.Project(firstPooled);
        var secondEmbedding = _projection.Project(secondPooled);
        double cosine = 0;
        for (var i = 0; i < firstEmbedding.Length; i++)
            cosine += firstEmbedding[i] * secondEmbedding[i];
        return Math.Clamp(cosine, -1, 1);
    }

    public void Dispose()
    {
        _session.Dispose();
        _inferenceGate.Dispose();
    }
}
