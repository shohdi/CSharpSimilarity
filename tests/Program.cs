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

Check(NameNormalizer.Normalize(" شَـهْدِي ") == "شهدي", "Arabic diacritics and tatweel are normalized.");
Check(NameNormalizer.Normalize("أحمد إِبراهيم") == "احمد ابراهيم", "Arabic alef variants normalize consistently.");
Check(NameNormalizer.Normalize("François  O’Connor") == "francois o connor", "Latin accents, case and punctuation normalize.");
Check(NameNormalizer.Normalize("شركة ١٢۳") == "شركة 123", "Arabic and Persian digits normalize without dropping identifiers.");
Check(NameNormalizer.Normalize(null) == "", "Null is an empty normalized name.");
Check(NameNormalizer.ContainsArabic("شهدي") && !NameNormalizer.ContainsArabic("shohdi"), "Arabic script detection.");

var matcher = new NameLexicalMatcher();
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
