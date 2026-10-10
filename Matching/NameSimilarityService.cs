using CSharpSimilarity.Inference;

namespace CSharpSimilarity.Matching;

public sealed class NameSimilarityService(OnnxNameEncoder encoder, NameLexicalMatcher matcher)
{
    public async Task<int> CompareAsync(string first, string second, CancellationToken cancellationToken)
    {
        // Validate the complete submitted fields before selecting equivalent
        // aliases, so splitting does not bypass the original input limits.
        encoder.ValidateName(first);
        encoder.ValidateName(second);
        var lexical = matcher.Compare(first, second);
        // Embed the matched complete variants, not both translations joined
        // together. Run the real model for every valid pair.
        var cosine = await encoder.CosineSimilarityAsync(
            lexical.MatchedFirstName ?? first,
            lexical.MatchedSecondName ?? second,
            cancellationToken);
        if (lexical.ExactNormalizedMatch)
            return 100;
        if (lexical.CrossScriptPhoneticMatch || lexical.EquivalentAliasMatch)
            return 99;

        // A general sentence model can make unrelated personal names look similar.
        // Give spelling/transliteration the main weight and use embeddings as evidence.
        var semantic = Math.Pow(Math.Max(0, cosine), 2);
        var combined = 0.85 * lexical.Similarity + 0.15 * semantic;
        return Math.Clamp((int)Math.Round(combined * 100, MidpointRounding.AwayFromZero), 0, 98);
    }
}
