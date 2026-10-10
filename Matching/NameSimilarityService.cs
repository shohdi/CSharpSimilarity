using CSharpSimilarity.Inference;

namespace CSharpSimilarity.Matching;

public sealed class NameSimilarityService(
    OnnxNameEncoder encoder,
    NameLexicalMatcher matcher,
    OrganizationNameMatcher organizationMatcher)
{
    public async Task<int> CompareAsync(string first, string second, CancellationToken cancellationToken)
    {
        // Validate the complete submitted fields before selecting equivalent
        // aliases, so splitting does not bypass the original input limits.
        encoder.ValidateName(first);
        encoder.ValidateName(second);
        var lexical = matcher.Compare(first, second);
        var organization = organizationMatcher.Compare(first, second);
        if (organization.Applicable)
        {
            // Only verified complete aliases can be selected from a bilingual
            // record. Embed significant brand and sector words without wrappers.
            var firstCandidate = organizationMatcher.GetCandidateNames(first).First(candidate =>
                candidate.Name == organization.MatchedFirstName);
            var secondCandidate = organizationMatcher.GetCandidateNames(second).First(candidate =>
                candidate.Name == organization.MatchedSecondName);
            var organizationCosine = await encoder.CosineSimilarityAsync(
                firstCandidate.SemanticName.Length > 0 ? firstCandidate.SemanticName : firstCandidate.Name,
                secondCandidate.SemanticName.Length > 0 ? secondCandidate.SemanticName : secondCandidate.Name,
                cancellationToken);
            if (lexical.ExactNormalizedMatch)
                return 100;

            // LaBSE supplies multilingual evidence; distinctive spelling remains
            // the larger component because shared sector words are not identity.
            var organizationScore = 0.55 * organization.Similarity + 0.45 * Math.Max(0, organizationCosine);
            if (organization.ConflictingIdentifiers)
                organizationScore = Math.Min(organizationScore, 0.45);
            if (organization.ConflictingDistinctiveNames)
                organizationScore = Math.Min(organizationScore, 0.69);
            if (organization.ConflictingSectors)
                organizationScore = Math.Min(organizationScore, 0.79);
            return ToPercentage(organizationScore);
        }
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
        return ToPercentage(combined);
    }

    private static int ToPercentage(double score) =>
        Math.Clamp((int)Math.Round(score * 100, MidpointRounding.AwayFromZero), 0, 98);
}
