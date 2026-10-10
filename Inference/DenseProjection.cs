using System.Buffers.Binary;
using System.Text.Json;

namespace CSharpSimilarity.Inference;

// Reads the original SentenceTransformer Dense module without Python/PyTorch.
internal sealed class DenseProjection
{
    public const int InputDimensions = 768;
    public const int OutputDimensions = 512;
    private readonly float[] _weights;
    private readonly float[] _bias;

    public DenseProjection(string path)
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length < 8)
            throw new InvalidDataException("Invalid safetensors file.");
        var headerLength = checked((int)BinaryPrimitives.ReadUInt64LittleEndian(bytes));
        if (headerLength <= 0 || headerLength > bytes.Length - 8)
            throw new InvalidDataException("Invalid safetensors header.");
        using var header = JsonDocument.Parse(bytes.AsMemory(8, headerLength));
        var dataStart = 8 + headerLength;
        _bias = ReadTensor(header.RootElement.GetProperty("linear.bias"), [OutputDimensions]);
        _weights = ReadTensor(header.RootElement.GetProperty("linear.weight"), [OutputDimensions, InputDimensions]);

        float[] ReadTensor(JsonElement tensor, int[] shape)
        {
            if (tensor.GetProperty("dtype").GetString() != "F32" ||
                !tensor.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).SequenceEqual(shape))
                throw new InvalidDataException("Unexpected Dense tensor type or dimensions.");
            var count = shape.Aggregate(1, (product, size) => checked(product * size));
            var offsets = tensor.GetProperty("data_offsets");
            var start = offsets[0].GetInt32();
            var end = offsets[1].GetInt32();
            if (start < 0 || end - start != count * sizeof(float) || end > bytes.Length - dataStart)
                throw new InvalidDataException("Invalid Dense tensor offsets.");
            var values = new float[count];
            for (var i = 0; i < count; i++)
                values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(dataStart + start + i * sizeof(float), sizeof(float)));
            return values;
        }
    }

    public float[] Project(ReadOnlySpan<float> pooled)
    {
        if (pooled.Length != InputDimensions)
            throw new InvalidDataException("Unexpected embedding dimension.");
        var result = new float[OutputDimensions];
        double squaredNorm = 0;
        for (var row = 0; row < OutputDimensions; row++)
        {
            double value = _bias[row];
            var offset = row * InputDimensions;
            for (var column = 0; column < InputDimensions; column++)
                value += _weights[offset + column] * pooled[column];
            result[row] = (float)Math.Tanh(value);
            squaredNorm += result[row] * result[row];
        }
        var norm = Math.Sqrt(squaredNorm);
        if (!double.IsFinite(norm) || norm <= 0)
            throw new InvalidDataException("Model produced an invalid embedding.");
        for (var i = 0; i < result.Length; i++)
            result[i] = (float)(result[i] / norm);
        return result;
    }
}
