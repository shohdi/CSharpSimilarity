using CSharpSimilarity.Inference;

namespace CSharpSimilarity.Matching;

public sealed class NameSimilarityService(OnnxNameEncoder encoder, NameLexicalMatcher matcher)
{
    public async Task<int> CompareAsync(string first, string second, CancellationToken cancellationToken)
    {
        var normalizedFirst = NameNormalizer.Normalize(first);
        var normalizedSecond = NameNormalizer.Normalize(second);
        var lexical = matcher.Compare(first, second);
        // Run the real model for every valid pair, including transliteration matches.
        var cosine = await encoder.CosineSimilarityAsync(normalizedFirst, normalizedSecond, cancellationToken);
        if (lexical.ExactNormalizedMatch)
            return 100;
        if (lexical.CrossScriptPhoneticMatch)
            return 99;

        // A general sentence model can make unrelated personal names look similar.
        // Give spelling/transliteration the main weight and use embeddings as evidence.
        var semantic = Math.Pow(Math.Max(0, cosine), 2);
        var combined = 0.85 * lexical.Similarity + 0.15 * semantic;
        return Math.Clamp((int)Math.Round(combined * 100, MidpointRounding.AwayFromZero), 0, 98);
    }
}
