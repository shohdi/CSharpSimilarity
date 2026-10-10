using System.Text;

namespace CSharpSimilarity.Matching;

/// <summary>A name-similarity signal, not a calibrated probability of identity.</summary>
public sealed record LexicalMatch(
    double Similarity,
    bool ExactNormalizedMatch,
    bool CrossScriptPhoneticMatch);

/// <summary>
/// Compares whole name tokens and Arabic/English transliterations locally.
/// Unmatched tokens reduce the score, including when one name is a subset of the other.
/// </summary>
public sealed class NameLexicalMatcher
{
    // Finite translations of organizational type words. These stay as tokens:
    // a shared company type is not enough to match different company names.
    private static readonly IReadOnlyDictionary<string, string> OrganizationTypes =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["company"] = "company", ["companies"] = "company", ["co"] = "company",
            ["corporation"] = "company", ["corp"] = "company",
            ["شركة"] = "company", ["شركه"] = "company", ["شركات"] = "company",
            ["bank"] = "bank", ["بنك"] = "bank", ["مصرف"] = "bank",
            ["foundation"] = "foundation", ["مؤسسة"] = "foundation", ["مؤسسه"] = "foundation",
            ["موسسة"] = "foundation", ["موسسه"] = "foundation",
            ["university"] = "university", ["جامعة"] = "university", ["جامعه"] = "university",
            ["association"] = "association", ["جمعية"] = "association", ["جمعيه"] = "association",
            ["hospital"] = "hospital", ["مستشفي"] = "hospital",
            ["center"] = "center", ["centre"] = "center", ["مركز"] = "center",
            ["group"] = "group", ["مجموعة"] = "group", ["مجموعه"] = "group",
            ["school"] = "school", ["مدرسة"] = "school", ["مدرسه"] = "school",
            ["holding"] = "holding", ["holdings"] = "holding", ["قابضة"] = "holding", ["قابضه"] = "holding",
            ["limited"] = "limited", ["ltd"] = "limited"
        };

    public LexicalMatch Compare(string? first, string? second)
    {
        var firstNormalized = NameNormalizer.Normalize(first);
        var secondNormalized = NameNormalizer.Normalize(second);
        if (firstNormalized.Length == 0 || secondNormalized.Length == 0)
        {
            return new LexicalMatch(0, false, false);
        }

        if (firstNormalized == secondNormalized)
        {
            return new LexicalMatch(1, true, false);
        }

        var firstTokens = PrepareTokens(firstNormalized);
        var secondTokens = PrepareTokens(secondNormalized);
        var size = Math.Max(firstTokens.Length, secondTokens.Length);
        var scores = new double[size, size];
        var crossScript = new bool[size, size];

        for (var row = 0; row < firstTokens.Length; row++)
        {
            for (var column = 0; column < secondTokens.Length; column++)
            {
                (scores[row, column], crossScript[row, column]) =
                    CompareToken(firstTokens[row], secondTokens[column]);
            }
        }

        var assignment = MaximumAssignment(scores);
        var total = 0d;
        var allStrong = firstTokens.Length == secondTokens.Length;
        var anyCrossScript = false;
        for (var row = 0; row < size; row++)
        {
            var column = assignment[row];
            total += scores[row, column];
            allStrong &= scores[row, column] >= 0.97;
            anyCrossScript |= crossScript[row, column];
        }

        return new LexicalMatch(Math.Clamp(total / size, 0, 1), false,
            allStrong && anyCrossScript);
    }

    private static (double Score, bool CrossScript) CompareToken(string first, string second)
    {
        if (first == second)
        {
            return (1, false);
        }

        // Numeric identifiers carry identity information, including inside words.
        // A spelling resemblance must not hide a different branch/account number.
        if (string.Concat(first.Where(char.IsDigit)) != string.Concat(second.Where(char.IsDigit)))
        {
            return (0, false);
        }

        var firstArabic = NameNormalizer.ContainsArabic(first);
        var secondArabic = NameNormalizer.ContainsArabic(second);
        var editSimilarity = EditSimilarity(first, second);

        if (firstArabic == secondArabic)
        {
            // Keep same-script comparisons conservative: similar sounds alone do not
            // distinguish common pairs such as Hasan/Hussein or Mohamed/Mahmoud.
            var score = editSimilarity;
            if (!firstArabic && IsLatin(first) && IsLatin(second))
            {
                if (FinalIToY(first) == FinalIToY(second) && first.Length >= 3)
                {
                    score = Math.Max(score, 0.96);
                }
                else if (editSimilarity >= 0.6 &&
                         LatinSkeleton(first) == LatinSkeleton(second) &&
                         LatinSkeleton(first).Length >= 2)
                {
                    score = Math.Max(score, 0.88);
                }
            }

            return (score >= 0.5 ? score : 0, false);
        }

        var arabic = firstArabic ? first : second;
        var latin = firstArabic ? second : first;
        if (!IsLatin(latin) || !IsArabicToken(arabic))
        {
            return (editSimilarity >= 0.5 ? editSimilarity : 0, false);
        }

        var arabicSkeleton = ArabicSkeleton(arabic);
        var latinSkeleton = LatinSkeleton(latin);
        if (arabicSkeleton.Length == 0 || latinSkeleton.Length == 0)
        {
            return (0, false);
        }

        if (arabicSkeleton == latinSkeleton && HasCompatibleAlefAnchors(arabic, latin))
        {
            return (ExactPhoneticScore(arabicSkeleton.Length), true);
        }

        // Short u/o can represent Arabic waw (e.g. Yusuf, Mahmud). Candidate
        // positions are preserved, so Mohamed does not collapse into Mahmoud.
        foreach (var (variant, mask) in LatinWawVariants(latin))
        {
            if (arabicSkeleton == variant && variant.Length >= 2 &&
                HasCompatibleAlefAnchors(arabic, latin, mask))
            {
                return (0.98, true);
            }
        }

        // Egyptian Arabic commonly writes ج where English uses g (Gamal,
        // Magdy). Gh remains a separate sound and is not changed by this rule.
        if (arabic.Contains('ج') && arabicSkeleton.Length >= 3 &&
            latinSkeleton.Replace('g', 'j') == arabicSkeleton &&
            HasCompatibleAlefAnchors(arabic, latin))
        {
            return (0.98, true);
        }

        // Short i is not necessarily written as Arabic ya: Khalid/خالد and
        // Tariq/طارق. Long ei/ee/ai digraphs stay distinct from an omitted i.
        if (arabicSkeleton.Length >= 3 && RemoveOptionalLatinI(latin) == arabicSkeleton &&
            HasCompatibleAlefAnchors(arabic, latin, omitI: true))
        {
            return (0.96, false);
        }

        // Ta marbuta may be written as h, t, or only the preceding vowel.
        if (arabic.EndsWith('ة'))
        {
            var withoutEnding = ArabicSkeleton(arabic[..^1]);
            if (withoutEnding.Length >= 3 && HasCompatibleAlefAnchors(arabic, latin) &&
                (latinSkeleton == withoutEnding || latinSkeleton == withoutEnding + "t"))
            {
                return (0.98, true);
            }

            // Fatima/Fatimah: the short i is not always present in Arabic spelling.
            if (withoutEnding.Length >= 3 && HasCompatibleAlefAnchors(arabic, latin, omitI: true) &&
                (RemoveOptionalLatinI(latin) == withoutEnding ||
                 RemoveOptionalLatinI(latin) == withoutEnding + "h"))
            {
                return (0.96, false);
            }
        }

        var phoneticSimilarity = EditSimilarity(arabicSkeleton, latinSkeleton);
        // Two different phonetic skeletons never get the near-certain score used
        // for equivalent transliterations, even when they differ by one letter.
        var approximate = Math.Min(0.92, phoneticSimilarity * 0.92);
        if (!HasCompatibleAlefAnchors(arabic, latin))
        {
            approximate = Math.Min(approximate, 0.72);
        }
        return (approximate >= 0.5 ? approximate : 0, false);
    }

    private static string[] PrepareTokens(string value)
    {
        var original = value.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var result = new List<string>(original.Length);
        for (var index = 0; index < original.Length; index++)
        {
            var token = original[index];
            if (index + 1 < original.Length && token is "al" or "el" or "ال")
            {
                // Preserve the article in the name rather than discarding it.
                token = (token == "el" ? "al" : token) + original[++index];
            }
            else if (index + 1 < original.Length && token == "عبد" && original[index + 1] == "الله")
            {
                token += original[++index];
            }

            if (OrganizationTypes.TryGetValue(token, out var canonical))
            {
                token = canonical;
            }
            else if (token.StartsWith("ال", StringComparison.Ordinal) &&
                     OrganizationTypes.TryGetValue(token[2..], out canonical))
            {
                token = canonical;
            }
            else if (token.StartsWith("al", StringComparison.Ordinal) &&
                     OrganizationTypes.TryGetValue(token[2..], out canonical))
            {
                token = canonical;
            }
            result.Add(token);
        }
        return result.ToArray();
    }

    private static bool HasCompatibleAlefAnchors(string arabic, string latin,
        int wawMask = 0, bool omitI = false)
    {
        // An explicit Arabic alef must have a corresponding Latin vowel at that
        // consonant position. Ahmed/أحمد matches, whereas Ahmed/حامد does not
        // receive .99 merely because both vowel-free skeletons are hmd.
        for (var position = 0; position < arabic.Length; position++)
        {
            if (arabic[position] != 'ا')
            {
                continue;
            }

            var consonantsBeforeAlef = ArabicSkeleton(arabic[..position]).Length;
            var foundVowel = false;
            for (var latinPosition = 0; latinPosition < latin.Length; latinPosition++)
            {
                if ((latin[latinPosition] is 'a' or 'e' or 'o' or 'u' ||
                     latinPosition == 0 && latin[latinPosition] == 'i') &&
                    LatinSkeleton(latin[..latinPosition], wawMask, omitI).Length == consonantsBeforeAlef)
                {
                    foundVowel = true;
                    break;
                }
            }
            if (!foundVowel)
            {
                return false;
            }
        }
        return true;
    }

    private static double ExactPhoneticScore(int length) => length switch
    {
        1 => 0.70,
        2 => 0.98,
        _ => 0.99
    };

    private static bool IsLatin(string value) =>
        value.All(character => character is >= 'a' and <= 'z' or >= '0' and <= '9');

    private static bool IsArabicToken(string value) =>
        value.All(character => character is >= '\u0600' and <= '\u06FF'
            or >= '0' and <= '9');

    private static string FinalIToY(string value) =>
        value.EndsWith('i') ? value[..^1] + "y" : value;

    private static string ArabicSkeleton(string value)
    {
        var result = new StringBuilder();
        foreach (var character in value)
        {
            var sound = character switch
            {
                'ا' or 'ء' or 'ع' => "",
                'ب' or 'پ' => "b",
                'ت' or 'ث' => "t",
                'ج' or 'چ' => "j",
                'ح' or 'ه' or 'ة' => "h",
                'خ' => "x",
                'د' or 'ذ' or 'ض' => "d",
                'ر' => "r",
                'ز' or 'ظ' => "z",
                'س' or 'ص' => "s",
                'ش' => "š",
                'ط' => "t",
                'غ' => "ġ",
                'ف' or 'ڤ' => "f",
                'ق' or 'ك' or 'گ' => "k",
                'ل' => "l",
                'م' => "m",
                'ن' => "n",
                'و' or 'ؤ' => "w",
                'ي' or 'ئ' => "y",
                _ => character.ToString()
            };
            result.Append(sound);
        }

        return CollapseRepeats(result.ToString());
    }

    private static string LatinSkeleton(string value, int wawMask = 0, bool omitI = false)
    {
        var result = new StringBuilder();
        var wawPosition = 0;
        for (var position = 0; position < value.Length; position++)
        {
            var character = value[position];
            var pair = position + 1 < value.Length ? value.Substring(position, 2) : "";
            var digraph = pair switch
            {
                "sh" or "ch" => "š",
                "kh" => "x",
                "gh" => "ġ",
                "ph" => "f",
                "th" => "t",
                "dh" => "d",
                "dj" => "j",
                "ou" or "oo" => "w",
                "ee" or "ei" or "ai" or "ey" => "y",
                _ => null
            };
            if (digraph is not null)
            {
                result.Append(digraph);
                position++;
                continue;
            }

            switch (character)
            {
                case 'a':
                case 'e':
                    break;
                case 'o':
                case 'u':
                    if (wawPosition < 5 && (wawMask & (1 << wawPosition)) != 0)
                    {
                        result.Append('w');
                    }
                    wawPosition++;
                    break;
                case 'i':
                    if (position > 0 && !omitI)
                    {
                        result.Append('y');
                    }
                    break;
                case 'q':
                case 'c':
                    result.Append('k');
                    break;
                default:
                    result.Append(character);
                    break;
            }
        }

        return CollapseRepeats(result.ToString());
    }

    private static IEnumerable<(string Skeleton, int Mask)> LatinWawVariants(string value)
    {
        // Bound variant generation even for unusually long, vowel-heavy inputs.
        var vowelCount = Math.Min(5, value.Count(character => character is 'o' or 'u'));
        for (var mask = 1; mask < (1 << vowelCount); mask++)
        {
            yield return (LatinSkeleton(value, mask), mask);
        }
    }

    private static string RemoveOptionalLatinI(string value) => LatinSkeleton(value, omitI: true);

    private static string CollapseRepeats(string value)
    {
        var result = new StringBuilder(value.Length);
        foreach (var character in value)
        {
            if (result.Length == 0 || result[^1] != character || char.IsDigit(character))
            {
                result.Append(character);
            }
        }
        return result.ToString();
    }

    private static double EditSimilarity(string first, string second)
    {
        if (first == second)
        {
            return 1;
        }

        var previousPrevious = new int[second.Length + 1];
        var previous = Enumerable.Range(0, second.Length + 1).ToArray();
        var current = new int[second.Length + 1];
        for (var row = 1; row <= first.Length; row++)
        {
            current[0] = row;
            for (var column = 1; column <= second.Length; column++)
            {
                var substitution = first[row - 1] == second[column - 1] ? 0 : 1;
                current[column] = Math.Min(Math.Min(previous[column] + 1,
                    current[column - 1] + 1), previous[column - 1] + substitution);
                if (row > 1 && column > 1 && first[row - 1] == second[column - 2] &&
                    first[row - 2] == second[column - 1])
                {
                    current[column] = Math.Min(current[column], previousPrevious[column - 2] + 1);
                }
            }

            (previousPrevious, previous, current) = (previous, current, previousPrevious);
        }

        return 1d - (double)previous[second.Length] / Math.Max(first.Length, second.Length);
    }

    private static int[] MaximumAssignment(double[,] scores)
    {
        // Hungarian assignment maximizes total token similarity while matching
        // each token at most once; repeated words cannot inflate coverage.
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
                    {
                        continue;
                    }

                    var cost = 1 - scores[currentRow - 1, candidate - 1] -
                        rowPotential[currentRow] - columnPotential[candidate];
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
                    {
                        minimum[candidate] -= delta;
                    }
                }
                column = nextColumn;
            } while (matching[column] != 0);

            do
            {
                var previousColumn = route[column];
                matching[column] = matching[previousColumn];
                column = previousColumn;
            } while (column != 0);
        }

        var result = new int[size];
        for (var column = 1; column <= size; column++)
        {
            result[matching[column] - 1] = column - 1;
        }
        return result;
    }
}
