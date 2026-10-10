using System.Text.Json;
using CSharpSimilarity.Inference;
using CSharpSimilarity.Matching;

var projectRoot = args.Length == 1 ? Path.GetFullPath(args[0]) : FindProjectRoot();
var failures = new List<string>();
var checks = 0;
void Check(bool condition, string description)
{
    checks++;
    if (!condition) failures.Add(description);
}
void Reject<T>(Action action, string description) where T : Exception
{
    var rejected = false;
    try { action(); }
    catch (T) { rejected = true; }
    Check(rejected, description);
}

var tokenizer = new WordPieceTokenizer(Path.Combine(projectRoot, "Models", "tokenizer.json"));
using (var fixtures = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "tokenizer-fixtures.json"))))
{
    foreach (var fixture in fixtures.RootElement.EnumerateArray())
    {
        var text = fixture.GetProperty("text").GetString()!;
        var expected = fixture.GetProperty("input_ids").EnumerateArray().Select(x => x.GetInt64());
        Check(expected.SequenceEqual(tokenizer.Encode(text)), $"Independent tokenizer fixture: {JsonSerializer.Serialize(text)}");
    }
}

Check(tokenizer.Encode("a𠮷b").SequenceEqual(new long[] { 101, 169, 100, 170, 102 }),
    "Supplementary CJK characters must be separated without splitting their surrogate pair.");
Check(tokenizer.Encode(string.Join(' ', Enumerable.Repeat("Ali", 126))).Length == 128,
    "Exactly 126 name tokens plus CLS/SEP must fit.");
Reject<ArgumentException>(() => tokenizer.Encode(string.Join(' ', Enumerable.Repeat("Ali", 127))),
    "A name longer than the model limit must reject instead of silently truncating.");
Reject<ArgumentOutOfRangeException>(() => tokenizer.Encode("Ali", 1), "A token limit without room for CLS/SEP must reject.");
var casedBilingualLimit = string.Join(' ', Enumerable.Repeat("aHmEd", 23)) + " " +
    string.Join(' ', Enumerable.Repeat("احمد", 23));
Check(casedBilingualLimit.Length <= 256, "The cased bilingual token-limit fixture must fit the API character limit.");
Reject<ArgumentException>(() => tokenizer.Encode(casedBilingualLimit),
    "The complete cased bilingual field must exceed the token limit, even when either alias fits separately.");

var labseTokenizer = new WordPieceTokenizer(Path.Combine(projectRoot, "Models", "LaBSE", "tokenizer.json"));
using (var fixtures = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "labse-tokenizer-fixtures.json"))))
{
    foreach (var fixture in fixtures.RootElement.EnumerateArray())
    {
        var text = fixture.GetProperty("text").GetString()!;
        var expected = fixture.GetProperty("input_ids").EnumerateArray().Select(x => x.GetInt64());
        // Capture the complete tokenization first; the endpoint's stricter 128-token
        // limit is checked separately, including for a verified bilingual alias.
        Check(expected.SequenceEqual(labseTokenizer.Encode(text, 256)),
            $"Independent LaBSE tokenizer fixture: {JsonSerializer.Serialize(text)}");
    }
}
var labseEnglishLimit = string.Join(' ', Enumerable.Repeat("mOhAmEd", 19));
var labseArabicLimit = string.Join(' ', Enumerable.Repeat("محمد", 19));
var labseBilingualLimit = labseEnglishLimit + " " + labseArabicLimit;
Check(labseBilingualLimit.Length <= 256, "The LaBSE bilingual limit fixture must fit the API character limit.");
Check(labseTokenizer.Encode(labseEnglishLimit).Length == 116 && labseTokenizer.Encode(labseArabicLimit).Length == 21,
    "Each alias individually must fit the model's 128-token service limit.");
Check(labseTokenizer.Encode(labseBilingualLimit, 256).Length == 135,
    "The complete original LaBSE bilingual field must have 135 tokens including CLS/SEP.");
Reject<ArgumentException>(() => labseTokenizer.Encode(labseBilingualLimit),
    "The complete LaBSE bilingual field must reject rather than bypass its token limit through alias selection.");

Check(NameNormalizer.Normalize(" شَـهْدِي ") == "شهدي", "Arabic diacritics and tatweel are normalized.");
Check(NameNormalizer.Normalize("أحمد إِبراهيم") == "احمد ابراهيم", "Arabic alef variants normalize consistently.");
Check(NameNormalizer.Normalize("François  O’Connor") == "francois o connor", "Latin accents, case and punctuation normalize.");
Check(NameNormalizer.Normalize("شركة ١٢۳") == "شركة 123", "Arabic and Persian digits normalize without dropping identifiers.");
Check(NameNormalizer.Normalize(null) == "", "Null is an empty normalized name.");
Check(NameNormalizer.ContainsArabic("شهدي") && !NameNormalizer.ContainsArabic("shohdi"), "Arabic script detection.");

var matcher = new NameLexicalMatcher();
Check(matcher.Compare(labseBilingualLimit, labseArabicLimit).EquivalentAliasMatch,
    "The LaBSE full-field limit example must be a verified bilingual alias so it exercises validation before alias selection.");
var lexicalCases = new (string First, string Second, double Minimum, double Maximum, bool Phonetic)[]
{
    ("shohdi", "شهدي", .98, 1, true),
    ("Mohamed", "محمد", .97, 1, true),
    ("Ahmed", "أحمد", .97, 1, true),
    ("SHOHDI", "shohdi", 1, 1, false),
    ("Ali", "علي", .97, 1, true),
    ("Gamal", "جمال", .97, 1, true),
    ("Magdy", "مجدي", .97, 1, true),
    ("Khalid", "خالد", .95, .97, false),
    ("شركة النور", "Al Nour Company", .97, 1, true),
    ("Ahmed", "حامد", 0, .73, false),
    ("Al Nour Company", "Al Noor Bank", 0, .8, false),
    ("Ahmed", "علي", 0, .4, false),
    ("محمد", "محمود", 0, .8, false),
    ("Hasan", "Hussein", 0, .8, false),
    ("Ahmed", "Ahmed Hassan", 0, .55, false),
    ("Ahmed Ahmed", "Ahmed", 0, .55, false),
    ("Company 12", "Company 13", 0, .8, false),
    ("Microsoft", "Apple", 0, .5, false),
    ("", "Ali", 0, 0, false)
};
foreach (var item in lexicalCases)
{
    var forward = matcher.Compare(item.First, item.Second);
    var reverse = matcher.Compare(item.Second, item.First);
    Check(forward.Similarity >= item.Minimum && forward.Similarity <= item.Maximum,
        $"Lexical score for {item.First} / {item.Second}: {forward.Similarity:R} outside [{item.Minimum}, {item.Maximum}].");
    Check(forward.CrossScriptPhoneticMatch == item.Phonetic,
        $"Phonetic match flag for {item.First} / {item.Second}.");
    Check(Math.Abs(forward.Similarity - reverse.Similarity) < 1e-12,
        $"Lexical matching must be symmetric for {item.First} / {item.Second}.");
}
Check(matcher.Compare("شَـهْدِي", "شهدي").ExactNormalizedMatch, "Normalized-equivalent Arabic names are exact.");
Check(matcher.Compare("Ahmed Hassan", "Hassan Ahmed").Similarity >= .99, "Word order is tolerated when all tokens match.");

// A bilingual source field repeats the full name; it must not be treated as
// eight independent name components, or accepted when its two halves conflict.
const string userEnglish = "Shohdy Mohamed Hussien ElSheemy";
const string userArabic = "شهدي محمد حسين الشيمي";
var bilingualName = userEnglish + " " + userArabic;
var nameRegressions = new (string First, string Second, double Minimum, double Maximum, string Description)[]
{
    (userEnglish, userArabic, .94, 1, "Full English/Arabic name transliteration"),
    (bilingualName, userArabic, .98, 1, "Bilingual full name versus its Arabic alias"),
    (bilingualName, userEnglish, .98, 1, "Bilingual full name versus its English alias"),
    (userArabic + " " + userEnglish, userArabic, .98, 1, "Arabic-first bilingual full name"),
    ("Ahmed Hassan أحمد حسن", "أحمد حسن", .98, 1, "Bilingual alias recognition also generalizes to another full name"),
    ("Ahmed Hassan 12 أحمد حسن ١٢", "أحمد حسن 12", .98, 1, "Consistent numeric identifiers survive bilingual alias selection"),
    (userEnglish, "Shohdy Mohamed Hussein ElSheemy", .90, 1, "Hussein/Hussien spelling variation"),
    (userEnglish, "Shohdy Mohamed Hussien El Sheemy", .90, 1, "Detached surname article"),
    (userEnglish, "Shohdy Mohamed Hussein Al Shimi", .90, 1, "Alternative surname transliteration"),
    (userEnglish, "شهدي محمد حسين الحسيني", 0, .92, "Different surnames must retain their penalty"),
    (userEnglish, "Ahmed Mohamed Hussien ElSheemy", 0, .92, "Shared surname does not erase a different first name"),
    ("Ahmed Hassan محمد علي", "محمد علي", 0, .92, "Contradictory concatenated aliases cannot collapse to the matching half"),
    ("Ahmed Hassan محمد علي", "Ahmed Hassan", 0, .92, "Contradictory aliases cannot collapse to the English half either"),
    (userEnglish + " شهدي محمد حسين الحسيني", userArabic, 0, .92, "A single conflicting surname prevents bilingual alias collapse"),
    (bilingualName + " 12", userArabic, 0, .92, "An identifier in only one bilingual half cannot disappear"),
    ("Ahmed Hassan 12 أحمد حسن 13", "أحمد حسن 13", 0, .92, "Conflicting identifiers inside the alias halves prevent collapse"),
    (userEnglish + " شهدي محمد حسين", userEnglish, 0, .92, "An incomplete Arabic half is not a complete alias of the English name"),
    (userEnglish + " شهدي محمد حسين", userArabic, 0, .92, "An incomplete bilingual field cannot discard the missing surname"),
    (bilingualName + " 12", userArabic + " 13", 0, .92, "Bilingual alias handling must preserve numeric identifiers"),
    ("Shohdy Mohamed Hussien", userArabic, 0, .85, "A missing surname remains a partial match")
};
foreach (var item in nameRegressions)
{
    var forward = matcher.Compare(item.First, item.Second);
    var reverse = matcher.Compare(item.Second, item.First);
    Check(forward.Similarity >= item.Minimum && forward.Similarity <= item.Maximum,
        $"{item.Description}: {forward.Similarity:R} outside [{item.Minimum}, {item.Maximum}].");
    Check(Math.Abs(forward.Similarity - reverse.Similarity) < 1e-12,
        $"{item.Description} must have a symmetric lexical score.");
    if (item.Maximum < .95)
        Check(!forward.CrossScriptPhoneticMatch && !reverse.CrossScriptPhoneticMatch &&
            !forward.EquivalentAliasMatch && !reverse.EquivalentAliasMatch,
            $"{item.Description} must not receive a complete-transliteration or equivalent-alias flag.");
}

var selectedEnglishAlias = matcher.Compare(bilingualName, userEnglish);
Check(selectedEnglishAlias.MatchedFirstName == userEnglish,
    "The selected English alias must preserve source casing for the cased embedding model.");
Check(selectedEnglishAlias.MatchedSecondName == userEnglish,
    "The model's corresponding target name must preserve source casing too.");

var fullNameScore = matcher.Compare(userEnglish, userArabic).Similarity;
Check(matcher.Compare("Shohdy Mohamed Hussien", userArabic).Similarity < fullNameScore,
    "Matching only part of the full name must score below matching all its components.");

const string organizationShort = "Cavalier Dynamics Technologies";
const string organizationEnglish = "Cavalier Dynamics for Technologies Company";
const string organizationArabic = "شركة كافاليير دايناميكس للتقنيات";
var organizationRecord = organizationEnglish + " " + organizationArabic;
var organizationMatcher = new OrganizationNameMatcher();
var organizationCases = new (string First, string Second, double Minimum, double Maximum, string Description)[]
{
    (organizationShort, organizationRecord, .90, 1, "Short organization name versus its complete bilingual source record"),
    (organizationShort, organizationEnglish, .90, 1, "English organization connectors and legal suffix"),
    (organizationShort, organizationArabic, .80, 1, "Arabic-only organization transliteration and sector translation"),
    (organizationShort, "Frontier Dynamics Technologies", 0, .69, "Shared organization terminology cannot erase a different core brand"),
    (organizationShort, "Atlantic Mercury Technologies", 0, .69, "A shared sector does not match unrelated brands"),
    (organizationShort, "Cavalier Dynamics Construction Company", 0, .89, "Different explicit sectors retain a penalty for the same brand"),
    (organizationShort + " 12", organizationEnglish + " 13", 0, .45, "Organization identifiers must retain mismatches")
};
foreach (var item in organizationCases)
{
    var forward = organizationMatcher.Compare(item.First, item.Second);
    var reverse = organizationMatcher.Compare(item.Second, item.First);
    Check(forward.Applicable && reverse.Applicable, $"Organization comparison is applicable: {item.Description}.");
    Check(forward.Similarity >= item.Minimum && forward.Similarity <= item.Maximum,
        $"{item.Description}: {forward.Similarity:R} outside [{item.Minimum}, {item.Maximum}].");
    Check(Math.Abs(forward.Similarity - reverse.Similarity) < 1e-12,
        $"{item.Description} must produce a symmetric organization signal.");
}
Check(organizationMatcher.Compare(organizationShort + " 12", organizationEnglish + " 13").ConflictingIdentifiers,
    "Conflicting organization identifiers must be explicitly retained in the matching result.");
Check(organizationMatcher.Compare(organizationShort, "Cavalier Dynamics Construction Company").ConflictingSectors,
    "An explicit technology/construction difference must remain visible to the final scorer.");

var conflictingCompanyBrand = organizationEnglish + " شركة فرونتير دايناميكس للتقنيات";
var conflictingCompanyIds = organizationEnglish + " 12 " + organizationArabic + " 13";
var conflictingCompanySector = organizationEnglish + " شركة كافاليير دايناميكس المصرفية";
var oneAliasCompanyId = organizationEnglish + " 12 " + organizationArabic;
var organizationAliasCases = new (string Source, double Maximum, bool Brand, bool Identifiers, bool Sectors, string Description)[]
{
    (conflictingCompanyBrand, .69, true, false, false, "Different distinctive brands inside the bilingual company field"),
    (conflictingCompanyIds, .45, false, true, false, "Conflicting identifiers inside bilingual company aliases"),
    (conflictingCompanySector, .79, false, false, true, "Technology and banking sectors inside the company aliases conflict"),
    (oneAliasCompanyId, .45, false, true, false, "An identifier present in only one company alias cannot disappear")
};
foreach (var item in organizationAliasCases)
{
    var forward = organizationMatcher.Compare(organizationShort, item.Source);
    var reverse = organizationMatcher.Compare(item.Source, organizationShort);
    Check(forward.Similarity <= item.Maximum && reverse.Similarity <= item.Maximum,
        $"{item.Description} must retain a penalty in either comparison direction.");
    Check(Math.Abs(forward.Similarity - reverse.Similarity) < 1e-12,
        $"{item.Description} must produce a symmetric score.");
    Check(!item.Brand || forward.ConflictingDistinctiveNames && reverse.ConflictingDistinctiveNames,
        $"{item.Description} must retain its distinctive-brand conflict flag.");
    Check(!item.Identifiers || forward.ConflictingIdentifiers && reverse.ConflictingIdentifiers,
        $"{item.Description} must retain its identifier conflict flag.");
    Check(!item.Sectors || forward.ConflictingSectors && reverse.ConflictingSectors,
        $"{item.Description} must retain its sector conflict flag.");
    var candidates = organizationMatcher.GetCandidateNames(item.Source);
    Check(candidates.Count == 1 && candidates[0].Name == item.Source && !candidates[0].IsVerifiedAlias,
        $"{item.Description} must preserve the entire original field for semantic inference.");
}
var consistentCompanyCandidates = organizationMatcher.GetCandidateNames(organizationRecord);
Check(consistentCompanyCandidates.Count == 2 && consistentCompanyCandidates.All(candidate => candidate.IsVerifiedAlias),
    "A verified bilingual company field exposes both complete aliases.");
Check(consistentCompanyCandidates.Any(candidate => candidate.Name == organizationEnglish &&
        candidate.SemanticName == "Cavalier Dynamics Technologies"),
    "The English company model input preserves the brand and sector while removing connectors and legal suffixes.");
Check(consistentCompanyCandidates.Any(candidate => candidate.Name == organizationArabic &&
        candidate.SemanticName == "كافاليير دايناميكس تقنيات"),
    "The Arabic company model input preserves the brand and sector while removing company wrappers and attached function prefixes.");
var missingSectorSource = organizationEnglish + " شركة كافاليير دايناميكس";
var bankTarget = "Cavalier Dynamics Bank";
var missingSectorMatch = organizationMatcher.Compare(bankTarget, missingSectorSource);
Check(missingSectorMatch.ConflictingSectors && missingSectorMatch.Similarity <= .79,
    "A sector-free alias cannot hide the technology/banking conflict in the original bilingual record.");
Check(organizationMatcher.Compare(missingSectorSource, bankTarget).ConflictingSectors &&
    organizationMatcher.Compare(missingSectorSource, bankTarget).Similarity <= .79,
    "A sector-free alias must preserve the same conflict when the company comparison is reversed.");
foreach (var function in new[] { "for", "of" })
{
    var markerFreeCompany = $"Cavalier Dynamics {function} كافاليير دايناميكس";
    var selected = organizationMatcher.Compare(markerFreeCompany, organizationShort);
    var candidates = organizationMatcher.GetCandidateNames(markerFreeCompany);
    Check(candidates.Count == 2 && candidates.All(candidate => candidate.IsVerifiedAlias),
        $"Marker-free bilingual organization aliases with '{function}' use consistent organization parsing.");
    Check(candidates.Any(candidate => candidate.Name == selected.MatchedFirstName),
        $"The selected marker-free organization alias with '{function}' must exist in the candidates supplied to semantic inference.");
    Check(candidates.All(candidate => !candidate.SemanticName.Split(' ').Contains(function)),
        $"Semantic organization inputs remove the '{function}' wrapper while retaining the complete source alias.");
}

var dense = new DenseProjection(Path.Combine(projectRoot, "Models", "dense.safetensors"));
using (var fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "dense-fixture.json"))))
{
    var input = fixture.RootElement.GetProperty("pooled_input").EnumerateArray().Select(x => x.GetSingle()).ToArray();
    var expected = fixture.RootElement.GetProperty("expected_embedding").EnumerateArray().Select(x => x.GetDouble()).ToArray();
    var actual = dense.Project(input);
    var maxError = actual.Zip(expected, (value, reference) => Math.Abs(value - reference)).Max();
    Check(actual.Length == 512, "Dense projection must reduce 768 pooled dimensions to 512.");
    Check(maxError < 1e-6, $"Independent Dense+tanh+L2 reference: maximum error {maxError:R}.");
    Check(Math.Abs(actual.Sum(x => (double)x * x) - 1) < 1e-6, "Dense output must have unit L2 norm.");
}
Reject<InvalidDataException>(() => dense.Project(new float[767]), "An invalid Dense input shape must reject.");

var labseDense = new DenseProjection(Path.Combine(projectRoot, "Models", "LaBSE", "dense.safetensors"), 768, 768);
using (var fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "labse-dense-fixture.json"))))
{
    // This independent reference checks the full 768-dimensional LaBSE projection,
    // including matrix layout, bias, tanh and normalization after CLS pooling.
    var input = fixture.RootElement.GetProperty("cls_input").EnumerateArray().Select(x => x.GetSingle()).ToArray();
    var expected = fixture.RootElement.GetProperty("expected_embedding").EnumerateArray().Select(x => x.GetDouble()).ToArray();
    var actual = labseDense.Project(input);
    var maxError = actual.Zip(expected, (value, reference) => Math.Abs(value - reference)).Max();
    Check(actual.Length == 768 && expected.Length == 768,
        "LaBSE Dense projection must preserve 768 dimensions after CLS pooling.");
    Check(maxError < 1e-6, $"Independent LaBSE Dense+tanh+L2 reference: maximum error {maxError:R}.");
    Check(Math.Abs(actual.Sum(x => (double)x * x) - 1) < 1e-6, "LaBSE Dense output must have unit L2 norm.");
    Console.WriteLine($"LaBSE projection reference maximum absolute error: {maxError:R}.");
}
Reject<InvalidDataException>(() => labseDense.Project(new float[767]), "An invalid LaBSE Dense input shape must reject.");

Console.WriteLine($"{checks - failures.Count}/{checks} checks passed.");
foreach (var failure in failures) Console.Error.WriteLine($"FAIL: {failure}");
return failures.Count == 0 ? 0 : 1;

static string FindProjectRoot()
{
    foreach (var start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
    {
        var directory = new DirectoryInfo(start);
        while (directory != null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "CSharpSimilarity.csproj"))) return directory.FullName;
            directory = directory.Parent;
        }
    }
    throw new DirectoryNotFoundException("Run checks inside the repository or pass its path as the first argument.");
}
