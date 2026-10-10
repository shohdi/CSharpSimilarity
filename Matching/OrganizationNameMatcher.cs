using System.Text.RegularExpressions;

namespace CSharpSimilarity.Matching;

/// <summary>Organization spelling evidence; this is not an identity probability.</summary>
public sealed record OrganizationMatch(
    bool Applicable,
    double Similarity,
    bool DistinctiveNamesAgree,
    bool ConflictingIdentifiers,
    bool ConflictingDistinctiveNames,
    string? MatchedFirstName,
    string? MatchedSecondName)
{
    public bool ConflictingSectors { get; init; }
    /// <summary>Whole pure-script brands need multilingual model evidence.</summary>
    public bool CrossScriptBrandsRequireModel { get; init; }
}

/// <summary>A complete original-case name and its significant brand words.</summary>
public sealed record OrganizationNameCandidate(
    string Name,
    string DistinctiveName,
    bool IsVerifiedAlias)
{
    /// <summary>Brand and explicit sector words, without legal/function wrappers.</summary>
    public string SemanticName { get; init; } = DistinctiveName;
}

/// <summary>
/// Separates finite, known organizational descriptors from significant unknown
/// name words. Only fully verified bilingual duplicates are treated as aliases.
/// All unknown words and numeric identifiers remain significant.
/// </summary>
public sealed class OrganizationNameMatcher(NameLexicalMatcher? lexicalMatcher = null)
{
    private readonly NameLexicalMatcher _lexical = lexicalMatcher ?? new NameLexicalMatcher();
    private static readonly Regex NameWords = new(@"[\p{L}\p{N}\p{M}\u0640]+", RegexOptions.CultureInvariant);
    private static readonly Regex CamelBoundary = new(@"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", RegexOptions.CultureInvariant);
    private static readonly Regex Digits = new(@"\p{Nd}+", RegexOptions.CultureInvariant);
    private static readonly IReadOnlyDictionary<string, Term> KnownTerms = BuildTerms();
    private static readonly HashSet<string> FunctionWords = new(StringComparer.Ordinal)
    {
        "for", "of", "the", "and", "a", "an", "in", "at", "to", "by",
        "و", "ل", "لل", "ال", "في", "من", "عن", "الى", "الي", "مع", "ذات"
    };
    private static readonly string[] ArabicDescriptorPrefixes = ["وال", "لل", "بال", "فال", "كال", "ال", "و", "ل", "ب", "ف", "ك"];

    public OrganizationMatch Compare(string? first, string? second)
    {
        var firstText = first?.Trim() ?? "";
        var secondText = second?.Trim() ?? "";
        var firstOriginal = Parse(firstText, organizationContext: false);
        var secondOriginal = Parse(secondText, organizationContext: false);
        if (!firstOriginal.HasOrganizationTerms && !secondOriginal.HasOrganizationTerms)
            return new OrganizationMatch(false, 0, false, false, false, firstText, secondText);

        var firstBundle = Variants(firstText, organizationContext: true);
        var secondBundle = Variants(secondText, organizationContext: true);
        OrganizationMatch? best = null;
        foreach (var firstVariant in firstBundle.Names)
        {
            foreach (var secondVariant in secondBundle.Names)
            {
                var result = CompareParsed(firstVariant, secondVariant);
                var conflictingIds = result.ConflictingIdentifiers || firstBundle.ConflictingIdentifiers || secondBundle.ConflictingIdentifiers;
                var conflictingBrands = result.ConflictingDistinctiveNames || firstBundle.ConflictingBrands || secondBundle.ConflictingBrands;
                var conflictingSectors = result.ConflictingSectors || firstBundle.ConflictingSectors || secondBundle.ConflictingSectors;
                var score = result.Similarity;
                if (conflictingIds)
                    score = Math.Min(score, 0.45);
                if (conflictingBrands)
                    score = Math.Min(score, 0.69);
                if (conflictingSectors)
                    score = Math.Min(score, 0.79);
                var candidate = result with
                {
                    Similarity = score,
                    DistinctiveNamesAgree = result.DistinctiveNamesAgree && !firstBundle.ConflictingBrands && !secondBundle.ConflictingBrands,
                    ConflictingIdentifiers = conflictingIds,
                    ConflictingDistinctiveNames = conflictingBrands,
                    ConflictingSectors = conflictingSectors,
                    CrossScriptBrandsRequireModel = result.CrossScriptBrandsRequireModel &&
                        !firstBundle.ConflictingBrands && !secondBundle.ConflictingBrands
                };
                if (best is null || candidate.Similarity > best.Similarity + 1e-12 ||
                    (Math.Abs(candidate.Similarity - best.Similarity) <= 1e-12 && string.CompareOrdinal(PairKey(candidate), PairKey(best)) < 0))
                    best = candidate;
            }
        }
        return best!;
    }

    /// <summary>
    /// Complete verified alternatives for embedding. Unverified mixed records
    /// remain a whole record, so a contradictory half cannot silently disappear.
    /// </summary>
    public IReadOnlyList<OrganizationNameCandidate> GetCandidateNames(string? name)
    {
        var text = name?.Trim() ?? "";
        // This API is used after Compare establishes organization context.
        // Apply the same context to a marker-free customer-side brand so the
        // selected model candidates cannot diverge from Compare's variants.
        var bundle = Variants(text, organizationContext: true);
        return bundle.Names.Select(candidate => new OrganizationNameCandidate(
            candidate.Name, string.Join(' ', candidate.Brands.Select(word => word.Original)), bundle.VerifiedAliases)
        {
            SemanticName = string.Join(' ', candidate.Brands.Select(word => word.Original).Concat(candidate.SemanticDescriptors))
        }).ToArray();
    }

    private OrganizationMatch CompareParsed(ParsedName first, ParsedName second)
    {
        var brand = CompareBrands(first.Brands, second.Brands);
        var agree = first.Brands.Length > 0 && first.Brands.Length == second.Brands.Length && brand.Minimum >= 0.90;
        // Unexplained pure Arabic/English spelling differences need model
        // evidence; they are not themselves proof of different brand names.
        // Missing brand words and contradictory mixed records stay significant.
        var needsModel = !agree && first.Brands.Length > 0 && first.Brands.Length == second.Brands.Length &&
            OppositeBrandScripts(first.Brands, second.Brands);
        var conflictingBrands = first.Brands.Length > 0 && second.Brands.Length > 0 && !agree && !needsModel;
        var conflictingIds = !IdentifiersEqual(first.Identifiers, second.Identifiers);
        var conflictingSectors = first.Sectors.Count > 0 && second.Sectors.Count > 0 && !first.Sectors.Overlaps(second.Sectors);
        var sector = DescriptorSimilarity(first.Sectors, second.Sectors);
        var types = DescriptorSimilarity(first.Types, second.Types);
        // Unknown brand words dominate. Generic words cannot make two unrelated
        // brands a strong match, even if every sector/type word is identical.
        var score = 0.84 * brand.Similarity + 0.12 * sector + 0.04 * types;
        if (first.Brands.Length == 0 || second.Brands.Length == 0)
            score = Math.Min(score, 0.30);
        if (first.Sectors.Count > 0 && second.Sectors.Count > 0 && !first.Sectors.SetEquals(second.Sectors))
            score *= 0.80 + 0.10 * sector;
        if (conflictingIds)
            score = Math.Min(score, 0.45);
        if (conflictingBrands)
            score = Math.Min(score, 0.69);
        return new OrganizationMatch(true, Math.Clamp(score, 0, 1), agree,
            conflictingIds, conflictingBrands, first.Name, second.Name)
        {
            ConflictingSectors = conflictingSectors,
            CrossScriptBrandsRequireModel = needsModel
        };
    }

    private VariantBundle Variants(string original, bool organizationContext)
    {
        var whole = Parse(original, organizationContext);
        var runs = new List<List<string>>();
        var current = new List<string>();
        var script = 0;
        foreach (var word in Words(original))
        {
            var normalized = NameNormalizer.Normalize(word);
            if (normalized.Length == 0)
                continue;
            var nextScript = normalized.All(char.IsDigit) ? 0 : IsLatin(normalized) ? 1 :
                IsArabic(normalized) ? 2 : -1;
            if (nextScript < 0)
                return new VariantBundle([whole], false, false, false, false);
            if (nextScript != 0 && nextScript != script)
            {
                if (script != 0)
                {
                    runs.Add(current);
                    current = [];
                }
                script = nextScript;
            }
            current.Add(word);
        }
        if (current.Count > 0)
            runs.Add(current);
        if (runs.Count != 2)
            return new VariantBundle([whole], false, false, false, false);

        var first = Parse(string.Join(' ', runs[0]), organizationContext);
        var second = Parse(string.Join(' ', runs[1]), organizationContext);
        var brand = CompareBrands(first.Brands, second.Brands);
        var brandsAgree = first.Brands.Length > 0 && first.Brands.Length == second.Brands.Length && brand.Minimum >= 0.90;
        var identifiersAgree = IdentifiersEqual(first.Identifiers, second.Identifiers);
        // Missing legal suffixes are normal; a sector explicitly present in
        // each alias must be fully explained by the same known descriptors.
        var sectorsAgree = first.Sectors.Count == 0 || second.Sectors.Count == 0 || first.Sectors.SetEquals(second.Sectors);
        if (brandsAgree && identifiersAgree && sectorsAgree)
        {
            // A sector omitted from one translation still belongs to the same
            // verified source record. Carry that metadata into every alias so
            // selecting the shorter translation cannot conceal a sector conflict.
            var sourceSectors = new HashSet<string>(first.Sectors, StringComparer.Ordinal);
            sourceSectors.UnionWith(second.Sectors);
            first = first with { Sectors = new HashSet<string>(sourceSectors, StringComparer.Ordinal) };
            second = second with { Sectors = new HashSet<string>(sourceSectors, StringComparer.Ordinal) };
            return new VariantBundle([first, second], true, false, false, false);
        }
        return new VariantBundle([whole], false, !identifiersAgree,
            !brandsAgree && first.Brands.Length > 0 && second.Brands.Length > 0, !sectorsAgree);
    }

    private ParsedName Parse(string original, bool organizationContext)
    {
        var words = Words(original).Select(word => new Word(word, NameNormalizer.Normalize(word))).Where(word => word.Normalized.Length > 0).ToArray();
        var terms = words.Select(word => FindTerm(word.Normalized)).ToArray();
        var hasOrganizationTerms = terms.Any(term => term is not null);
        var stripFunctions = organizationContext || hasOrganizationTerms;
        var brands = new List<Word>();
        var sectors = new HashSet<string>(StringComparer.Ordinal);
        var types = new HashSet<string>(StringComparer.Ordinal);
        var semanticDescriptors = new List<string>();
        for (var index = 0; index < words.Length; index++)
        {
            var word = words[index];
            var term = terms[index];
            if (term is not null)
            {
                (term.Sector ? sectors : types).Add(term.Canonical);
                if (term.Sector)
                    semanticDescriptors.Add(SemanticSectorWord(word));
                continue;
            }
            if (word.Normalized.All(char.IsDigit))
                continue;
            if (stripFunctions && word.Normalized is "al" or "el" or "ال" && index + 1 < words.Length &&
                terms[index + 1] is null && !words[index + 1].Normalized.All(char.IsDigit))
            {
                var article = word.Normalized == "el" ? "al" : word.Normalized;
                brands.Add(new Word(word.Original + words[index + 1].Original, article + words[++index].Normalized));
                continue;
            }
            if (stripFunctions && FunctionWords.Contains(word.Normalized))
                continue;
            brands.Add(word);
        }
        var identifiers = Digits.Matches(NameNormalizer.Normalize(original)).Select(match => match.Value).Order(StringComparer.Ordinal).ToArray();
        return new ParsedName(original, brands.ToArray(), sectors, types, identifiers, hasOrganizationTerms,
            semanticDescriptors.Distinct(StringComparer.Ordinal).ToArray());
    }

    private (double Similarity, double Minimum) CompareBrands(Word[] first, Word[] second)
    {
        var size = Math.Max(first.Length, second.Length);
        if (size == 0)
            return (0, 0);
        var scores = new double[size, size];
        for (var row = 0; row < first.Length; row++)
            for (var column = 0; column < second.Length; column++)
                scores[row, column] = BrandTokenScore(first[row].Normalized, second[column].Normalized);
        var assignment = MaximumAssignment(scores);
        var total = 0d;
        var minimum = 1d;
        for (var row = 0; row < size; row++)
        {
            var score = scores[row, assignment[row]];
            total += score;
            minimum = Math.Min(minimum, score);
        }
        return (total / size, minimum);
    }

    private double BrandTokenScore(string first, string second)
    {
        var score = _lexical.Compare(first, second).Similarity;
        var firstArabic = NameNormalizer.ContainsArabic(first);
        var secondArabic = NameNormalizer.ContainsArabic(second);
        if (firstArabic == secondArabic)
            return score;
        var arabic = firstArabic ? first : second;
        var latin = firstArabic ? second : first;
        if (!IsLatin(latin) || !IsArabic(arabic))
            return score;
        foreach (var variant in LatinBrandVariants(latin))
            score = Math.Max(score, Math.Min(0.98, _lexical.Compare(variant, arabic).Similarity));
        // The Arabic definite article can be absent in an English brand name.
        // Its remaining name still has to match completely, not as a substring.
        if (arabic.StartsWith("ال", StringComparison.Ordinal) && arabic.Length >= 5)
        {
            var withoutArticle = arabic[2..];
            foreach (var variant in LatinBrandVariants(latin))
                score = Math.Max(score, Math.Min(0.96, _lexical.Compare(variant, withoutArticle).Similarity));
        }
        return score;
    }

    private static IEnumerable<string> LatinBrandVariants(string latin)
    {
        yield return latin;
        // Foreign v is routinely written ف in Arabic transliterations.
        if (latin.Contains('v'))
            yield return latin.Replace('v', 'f');
        // English consonant-adjacent y can denote /ai/, represented by اي in
        // Arabic (Dynamics, Sky). Preserve its position and all other letters.
        if (latin.Contains('y'))
        {
            var withAi = latin.Replace("y", "ai", StringComparison.Ordinal);
            yield return withAi;
            if (withAi.Contains('v'))
                yield return withAi.Replace('v', 'f');
        }
    }

    private static IEnumerable<string> Words(string original)
    {
        foreach (Match match in NameWords.Matches(original))
            foreach (var word in CamelBoundary.Split(match.Value))
                yield return word;
    }

    private static Term? FindTerm(string word)
    {
        if (KnownTerms.TryGetValue(word, out var term))
            return term;
        if (!NameNormalizer.ContainsArabic(word))
            return null;
        foreach (var prefix in ArabicDescriptorPrefixes)
            if (word.StartsWith(prefix, StringComparison.Ordinal) && KnownTerms.TryGetValue(word[prefix.Length..], out term))
                return term;
        return null;
    }

    private static string SemanticSectorWord(Word word)
    {
        if (!NameNormalizer.ContainsArabic(word.Normalized))
            return word.Original;
        if (KnownTerms.ContainsKey(word.Normalized))
            return word.Normalized;
        foreach (var prefix in ArabicDescriptorPrefixes)
            if (word.Normalized.StartsWith(prefix, StringComparison.Ordinal) && KnownTerms.ContainsKey(word.Normalized[prefix.Length..]))
                return word.Normalized[prefix.Length..];
        return word.Normalized;
    }

    private static bool IdentifiersEqual(string[] first, string[] second) => first.SequenceEqual(second, StringComparer.Ordinal);

    private static double DescriptorSimilarity(HashSet<string> first, HashSet<string> second)
    {
        if (first.Count == 0 && second.Count == 0)
            return 1;
        if (first.Count == 0 || second.Count == 0)
            return 0.85;
        return (double)first.Intersect(second).Count() / Math.Max(first.Count, second.Count);
    }

    private static bool IsLatin(string value) => value.All(character => character is >= 'a' and <= 'z' or >= '0' and <= '9');
    private static bool IsArabic(string value) => NameNormalizer.ContainsArabic(value) && value.All(character => character is >= '\u0600' and <= '\u06FF' or >= '0' and <= '9');

    private static bool OppositeBrandScripts(Word[] first, Word[] second) =>
        first.All(word => IsLatin(word.Normalized)) && second.All(word => IsArabic(word.Normalized)) ||
        first.All(word => IsArabic(word.Normalized)) && second.All(word => IsLatin(word.Normalized));

    private static string PairKey(OrganizationMatch match)
    {
        var first = match.MatchedFirstName ?? "";
        var second = match.MatchedSecondName ?? "";
        return string.CompareOrdinal(first, second) <= 0 ? first + "\0" + second : second + "\0" + first;
    }

    private static IReadOnlyDictionary<string, Term> BuildTerms()
    {
        var terms = new Dictionary<string, Term>(StringComparer.Ordinal);
        Add(false, "company", "company companies co corporation corp corporate incorporated inc enterprise enterprises firm firms شركة شركات شركه");
        Add(false, "limited", "limited ltd llc plc محدود محدودة محدوده مسؤولية مسئولية");
        Add(false, "foundation", "foundation establishment institution مؤسسة مؤسسه منشأة منشئه");
        Add(false, "group", "group مجموعة مجموعه");
        Add(false, "holding", "holding holdings قابضة قابضه");
        Add(false, "association", "association society جمعية جمعيه");
        Add(true, "technology", "technology technologies technological technical tech تقنية تقنيه تقنيات تكنولوجيا تكنلوجيا تكنولوجية تكنولوجيه تقني تقنيه");
        Add(true, "banking", "bank banks banking بنك بنوك مصرف مصارف مصرفية مصرفيه");
        Add(true, "trading", "trade trading commercial commerce تجارة تجاره تجارية تجاريه تجاري");
        Add(true, "construction", "construction constructions contracting contractor contractors building buildings مقاولات مقاول إنشاء انشاء إنشاءات انشاءات تشييد بناء");
        Add(true, "engineering", "engineering engineer engineers هندسة هندسه هندسية هندسيه هندسي");
        Add(true, "finance", "finance financial financing finance تمويل مالية ماليه مالي");
        Add(true, "investment", "investment investments investing استثمار استثمارات استثماري استثمارية استثماريه");
        Add(true, "insurance", "insurance insurer insurers تأمين تامين تأمينات تامينات");
        Add(true, "telecommunications", "telecommunications telecom communication communications اتصالات اتصال");
        Add(true, "energy", "energy power طاقة طاقه كهرباء كهربائية كهربايية");
        Add(true, "petroleum", "petroleum oil gas نفط نفطية نفطيه بترول غاز");
        Add(true, "pharmaceutical", "pharmaceutical pharmaceuticals pharma دواء أدوية ادويه ادوية صيدلة صيدله");
        Add(true, "medical", "medical medicine healthcare hospital hospitals مستشفى مستشفي مستشفيات طب طبي طبيه طبية صحة صحه");
        Add(true, "education", "education educational school schools university universities academy أكاديمية اكاديمية اكاديميه أكاديميه تعليم تعليمي تعليمية تعليميه مدرسة مدرسه جامعة جامعه");
        Add(true, "transport", "transport transportation logistics logistic shipping freight نقل شحن لوجستية لوجستيه لوجستيات");
        Add(true, "agriculture", "agriculture agricultural farming زراعة زراعه زراعي زراعية زراعيه");
        Add(true, "food", "food foods beverage beverages أغذية اغذية أغذيه اغذيه غذاء مشروبات");
        Add(true, "manufacturing", "manufacturing manufacturer manufacturers industry industries industrial صناعة صناعه صناعات صناعية صناعيه صناعي");
        Add(false, "services", "service services خدمات خدمة خدمه");
        Add(false, "solutions", "solution solutions حلول");
        Add(false, "center", "center centre centers centres مركز مراكز");
        return terms;

        void Add(bool sector, string canonical, string aliases)
        {
            foreach (var alias in aliases.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                var normalized = NameNormalizer.Normalize(alias);
                terms[normalized] = new Term(canonical, sector);
                if (normalized.EndsWith('ة'))
                    terms[normalized[..^1] + "ه"] = new Term(canonical, sector);
            }
        }
    }

    private static int[] MaximumAssignment(double[,] scores)
    {
        var size = scores.GetLength(0);
        var rowPotential = new double[size + 1];
        var columnPotential = new double[size + 1];
        var matching = new int[size + 1];
        var route = new int[size + 1];
        for (var row = 1; row <= size; row++)
        {
            matching[0] = row;
            var column = 0;
            var minimum = Enumerable.Repeat(double.PositiveInfinity, size + 1).ToArray();
            var used = new bool[size + 1];
            do
            {
                used[column] = true;
                var currentRow = matching[column];
                var delta = double.PositiveInfinity;
                var nextColumn = 0;
                for (var candidate = 1; candidate <= size; candidate++)
                {
                    if (used[candidate])
                        continue;
                    var cost = 1 - scores[currentRow - 1, candidate - 1] - rowPotential[currentRow] - columnPotential[candidate];
                    if (cost < minimum[candidate])
                    {
                        minimum[candidate] = cost;
                        route[candidate] = column;
                    }
                    if (minimum[candidate] < delta)
                    {
                        delta = minimum[candidate];
                        nextColumn = candidate;
                    }
                }
                for (var candidate = 0; candidate <= size; candidate++)
                {
                    if (used[candidate])
                    {
                        rowPotential[matching[candidate]] += delta;
                        columnPotential[candidate] -= delta;
                    }
                    else
                        minimum[candidate] -= delta;
                }
                column = nextColumn;
            } while (matching[column] != 0);
            do
            {
                var previous = route[column];
                matching[column] = matching[previous];
                column = previous;
            } while (column != 0);
        }
        var result = new int[size];
        for (var column = 1; column <= size; column++)
            result[matching[column] - 1] = column - 1;
        return result;
    }

    private sealed record Term(string Canonical, bool Sector);
    private sealed record Word(string Original, string Normalized);
    private sealed record ParsedName(string Name, Word[] Brands, HashSet<string> Sectors, HashSet<string> Types,
        string[] Identifiers, bool HasOrganizationTerms, string[] SemanticDescriptors);
    private sealed record VariantBundle(ParsedName[] Names, bool VerifiedAliases, bool ConflictingIdentifiers, bool ConflictingBrands, bool ConflictingSectors);
}
