using System.Globalization;
using System.Text.Encodings.Web;
using System.Text.Json.Serialization;
using CSharpSimilarity.Inference;
using CSharpSimilarity.Matching;

// Resolve settings and model files beside the executable, even when launched
// from a different working directory after publishing.
var builder = WebApplication.CreateBuilder(new WebApplicationOptions
{
    Args = args,
    ContentRootPath = AppContext.BaseDirectory
});
builder.Services.ConfigureHttpJsonOptions(options =>
    options.SerializerOptions.Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping);
builder.Services.AddProblemDetails();
builder.Services.AddSingleton<OnnxNameEncoder>();
builder.Services.AddSingleton<NameLexicalMatcher>();
builder.Services.AddSingleton<OrganizationNameMatcher>();
builder.Services.AddSingleton<NameSimilarityService>();

var app = builder.Build();
app.UseExceptionHandler();
// Load and verify the model before accepting requests. Never download at runtime.
_ = app.Services.GetRequiredService<OnnxNameEncoder>();

app.MapGet("/", () => Results.Ok(new
{
    service = "Local Arabic / English name similarity",
    endpoint = "/IsSame?first_name=shohdi&second_name=شهدي",
    model = OnnxNameEncoder.ModelName,
    percentageMeaning = "Name-match similarity score, not a calibrated identity probability."
}));
app.MapGet("/health", () => Results.Ok(new { status = "healthy", model = "loaded", inference = "local-cpu" }));

app.MapGet("/IsSame", async (HttpContext context, NameSimilarityService similarity) =>
{
    var errors = new Dictionary<string, string[]>();
    string? ReadName(string key)
    {
        var values = context.Request.Query[key];
        if (values.Count != 1 || string.IsNullOrWhiteSpace(values[0]))
        {
            errors[key] = ["Supply exactly one non-empty name."];
            return null;
        }

        var value = values[0]!;
        if (value.Length > 256)
            errors[key] = ["Names must be at most 256 characters."];
        else if (!value.Any(char.IsLetterOrDigit) || NameNormalizer.Normalize(value).Length == 0)
            errors[key] = ["Names must contain a letter or digit."];
        return value;
    }

    var first = ReadName("first_name");
    var second = ReadName("second_name");
    if (errors.Count > 0)
        return Results.ValidationProblem(errors);

    try
    {
        var percentage = await similarity.CompareAsync(first!, second!, context.RequestAborted);
        return Results.Ok(new SimilarityResponse(first!, second!, percentage.ToString(CultureInfo.InvariantCulture)));
    }
    catch (ArgumentException)
    {
        return Results.ValidationProblem(new Dictionary<string, string[]>
        {
            ["names"] = ["Names must fit within the model's 128-token limit and contain valid Unicode."]
        });
    }
});

app.Run();

public sealed record SimilarityResponse(
    [property: JsonPropertyName("first_name")] string FirstName,
    [property: JsonPropertyName("second_name")] string SecondName,
    [property: JsonPropertyName("Percentage")] string Percentage);
