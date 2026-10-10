using System.Globalization;
using System.Text;

namespace CSharpSimilarity.Matching;

/// <summary>Normalizes spelling without removing words or numeric identifiers.</summary>
public static class NameNormalizer
{
    public static string Normalize(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        var result = new StringBuilder(value.Length);
        var needsSpace = false;

        foreach (var rune in value.Normalize(NormalizationForm.FormKD).EnumerateRunes())
        {
            var category = Rune.GetUnicodeCategory(rune);
            if (category is UnicodeCategory.NonSpacingMark
                or UnicodeCategory.SpacingCombiningMark
                or UnicodeCategory.EnclosingMark || rune.Value == 0x0640)
            {
                // Combining accents, Arabic vowel marks, and tatweel are not name letters.
                continue;
            }

            var letter = rune.Value switch
            {
                0x0622 or 0x0623 or 0x0625 or 0x0671 => new Rune(0x0627), // alef
                0x0649 or 0x06CC => new Rune(0x064A), // Arabic/Persian ya
                0x06A9 => new Rune(0x0643), // Persian kaf
                >= 0x0660 and <= 0x0669 => new Rune('0' + rune.Value - 0x0660),
                >= 0x06F0 and <= 0x06F9 => new Rune('0' + rune.Value - 0x06F0),
                _ => Rune.ToLowerInvariant(rune)
            };

            if (Rune.IsLetterOrDigit(letter))
            {
                if (needsSpace && result.Length > 0)
                {
                    result.Append(' ');
                }

                result.Append(letter.ToString());
                needsSpace = false;
            }
            else
            {
                needsSpace = result.Length > 0;
            }
        }

        return result.ToString().Normalize(NormalizationForm.FormC);
    }

    public static bool ContainsArabic(string value) =>
        value.EnumerateRunes().Any(rune => Rune.IsLetter(rune) &&
            (rune.Value is >= 0x0600 and <= 0x06FF
                or >= 0x0750 and <= 0x077F
                or >= 0x08A0 and <= 0x08FF
                or >= 0xFB50 and <= 0xFDFF
                or >= 0xFE70 and <= 0xFEFF));
}
