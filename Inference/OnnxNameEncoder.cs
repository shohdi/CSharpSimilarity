using System.Security.Cryptography;
using Microsoft.ML.OnnxRuntime;

namespace CSharpSimilarity.Inference;

public sealed class OnnxNameEncoder : IDisposable
{
    public const string ModelName = "sentence-transformers/LaBSE";
    public const int EmbeddingDimensions = 768;
    private readonly InferenceSession _session;
    private readonly WordPieceTokenizer _tokenizer;
    private readonly DenseProjection _projection;
    private readonly SemaphoreSlim _inferenceGate = new(1, 1);

    public OnnxNameEncoder(IHostEnvironment environment, IConfiguration configuration, ILogger<OnnxNameEncoder> logger)
    {
        var configured = configuration["Similarity:ModelDirectory"] ?? "Models/LaBSE";
        var directory = Path.GetFullPath(configured, environment.ContentRootPath);
        var model = VerifiedFile("model_quantized.onnx", "2d209704455c8e5718dbef9c46ca22c823eb35256b74b4122d90845310c7ef82");
        var tokenizer = VerifiedFile("tokenizer.json", "09216b42d2697b7b4a26ac05ff09ba8bf52dc19b896c5ceee8bbff9f39055322");
        var dense = VerifiedFile("dense.safetensors", "f866c945fa2147ce4940c8ead0b4b20af4217e0f4664bd35f41c2a0967c4fe77");
        _tokenizer = new WordPieceTokenizer(tokenizer);
        _projection = new DenseProjection(dense, EmbeddingDimensions, EmbeddingDimensions);
        using var options = new Microsoft.ML.OnnxRuntime.SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = Math.Clamp(configuration.GetValue("Similarity:IntraOpThreads", 2), 1, Environment.ProcessorCount),
            InterOpNumThreads = 1
        };
        _session = new InferenceSession(model, options);
        if (!_session.InputMetadata.Keys.Order().SequenceEqual(new[] { "attention_mask", "input_ids", "token_type_ids" }) ||
            !_session.OutputMetadata.ContainsKey("last_hidden_state"))
            throw new InvalidDataException("The local ONNX model has unexpected inputs/outputs.");
        logger.LogInformation("Loaded {Model} for local CPU inference ({Dimensions} dimensions).", ModelName, EmbeddingDimensions);

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
        using var tokenTypes = OrtValue.CreateTensorValueFromMemory(new long[ids.Length], [2, sequenceLength]);
        var inputs = new Dictionary<string, OrtValue> { ["input_ids"] = inputIds, ["attention_mask"] = attentionMask, ["token_type_ids"] = tokenTypes };
        using var runOptions = new RunOptions();
        using var output = _session.Run(runOptions, inputs, ["last_hidden_state"]);
        var shape = output[0].GetTensorTypeAndShape();
        if (!shape.Shape.SequenceEqual(new long[] { 2, sequenceLength, EmbeddingDimensions }))
            throw new InvalidDataException("Unexpected ONNX output shape.");
        var hidden = output[0].GetTensorDataAsSpan<float>();
        // LaBSE uses CLS pooling, followed by its trained Dense layer.
        // DistilUSE's mean-pooling rule is not valid for this model.
        var firstPooled = hidden[..EmbeddingDimensions].ToArray();
        var secondPooled = hidden.Slice(sequenceLength * EmbeddingDimensions, EmbeddingDimensions).ToArray();
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
