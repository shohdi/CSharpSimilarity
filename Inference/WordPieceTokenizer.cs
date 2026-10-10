using System.Globalization;
using System.Text;
using System.Text.Json;

namespace CSharpSimilarity.Inference;

/// <summary>
/// The cased BERT WordPiece tokenizer used by LaBSE and the reference fixtures.
/// Its vocabulary and configuration come from the local Hugging Face tokenizer.json.
/// </summary>
internal sealed class WordPieceTokenizer
{
    private readonly Dictionary<string, long> _vocabulary = new(StringComparer.Ordinal);
    private readonly List<(string Text, long Id)> _specialTokens = [];
    private readonly string _continuationPrefix;
    private readonly int _maxCharactersPerWord;
    private readonly long _unknownId;
    private readonly long _clsId;
    private readonly long _sepId;

    public WordPieceTokenizer(string tokenizerPath)
    {
        using var stream = File.OpenRead(tokenizerPath);
        using var document = JsonDocument.Parse(stream);
        var root = document.RootElement;
        var normalizer = root.GetProperty("normalizer");
        var model = root.GetProperty("model");
        var stripAccents = normalizer.GetProperty("strip_accents");

        if (normalizer.GetProperty("type").GetString() != "BertNormalizer"
            || !normalizer.GetProperty("clean_text").GetBoolean()
            || !normalizer.GetProperty("handle_chinese_chars").GetBoolean()
            || normalizer.GetProperty("lowercase").GetBoolean()
            || (stripAccents.ValueKind != JsonValueKind.Null && stripAccents.GetBoolean())
            || root.GetProperty("pre_tokenizer").GetProperty("type").GetString() != "BertPreTokenizer"
            || model.GetProperty("type").GetString() != "WordPiece")
        {
            throw new InvalidDataException("The model requires a cased BERT WordPiece tokenizer with accents preserved.");
        }

        foreach (var token in model.GetProperty("vocab").EnumerateObject())
        {
            _vocabulary.Add(token.Name, token.Value.GetInt64());
        }

        _continuationPrefix = model.GetProperty("continuing_subword_prefix").GetString()
            ?? throw new InvalidDataException("The WordPiece continuation prefix is missing.");
        _maxCharactersPerWord = model.GetProperty("max_input_chars_per_word").GetInt32();
        _unknownId = RequiredToken(model.GetProperty("unk_token").GetString()!);
        _clsId = RequiredToken("[CLS]");
        _sepId = RequiredToken("[SEP]");

        foreach (var token in root.GetProperty("added_tokens").EnumerateArray())
        {
            // This model's added tokens are exact, unnormalised special strings.
            if (!token.GetProperty("special").GetBoolean()
                || token.GetProperty("normalized").GetBoolean()
                || token.GetProperty("single_word").GetBoolean()
                || token.GetProperty("lstrip").GetBoolean()
                || token.GetProperty("rstrip").GetBoolean())
            {
                throw new InvalidDataException("Unsupported added-token behavior in tokenizer.json.");
            }

            var text = token.GetProperty("content").GetString()!;
            var id = token.GetProperty("id").GetInt64();
            if (text.Length == 0 || RequiredToken(text) != id)
            {
                throw new InvalidDataException("Invalid special token in tokenizer.json.");
            }

            _specialTokens.Add((text, id));
        }
    }

    /// <summary>Returns input IDs with CLS and SEP, without padding or truncation.</summary>
    public long[] Encode(string text, int maxTokens = 128)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (maxTokens < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(maxTokens), "At least two tokens are needed for CLS and SEP.");
        }

        var ids = new List<long>(Math.Min(maxTokens, text.Length + 2)) { _clsId };
        var offset = 0;
        while (offset < text.Length)
        {
            var specialOffset = text.Length;
            (string Text, long Id)? nextSpecial = null;
            foreach (var special in _specialTokens)
            {
                var index = text.IndexOf(special.Text, offset, StringComparison.Ordinal);
                if (index >= 0 && (index < specialOffset
                    || (index == specialOffset && special.Text.Length > (nextSpecial?.Text.Length ?? 0))))
                {
                    specialOffset = index;
                    nextSpecial = special;
                }
            }

            foreach (var word in PreTokenize(text.AsSpan(offset, specialOffset - offset)))
            {
                AppendWordPieces(word, ids);
                CheckLength(ids.Count, maxTokens);
            }

            if (nextSpecial is not { } found)
            {
                break;
            }

            ids.Add(found.Id);
            CheckLength(ids.Count, maxTokens);
            offset = specialOffset + found.Text.Length;
        }

        ids.Add(_sepId);
        return ids.ToArray();
    }

    private long RequiredToken(string token) => _vocabulary.TryGetValue(token, out var id)
        ? id : throw new InvalidDataException($"The tokenizer vocabulary is missing {token}.");

    private static void CheckLength(int countWithoutSep, int maxTokens)
    {
        if (countWithoutSep >= maxTokens)
        {
            throw new ArgumentException($"The name exceeds the model limit of {maxTokens - 2} name tokens.", "text");
        }
    }

    private static List<string> PreTokenize(ReadOnlySpan<char> text)
    {
        var words = new List<string>();
        var word = new StringBuilder();
        foreach (var rune in text.EnumerateRunes())
        {
            var category = Rune.GetUnicodeCategory(rune);
            // Hugging Face's BertNormalizer removes Unicode category C, except TAB/LF/CR.
            if (rune.Value is 0 or 0xfffd
                || (rune.Value is not (9 or 10 or 13) && IsOther(category)))
            {
                continue;
            }

            if (Rune.IsWhiteSpace(rune))
            {
                FlushWord(word, words);
            }
            else if (IsChineseCharacter(rune.Value) || IsPunctuation(rune.Value, category))
            {
                FlushWord(word, words);
                words.Add(rune.ToString());
            }
            else
            {
                word.Append(rune.ToString());
            }
        }

        FlushWord(word, words);
        return words;
    }

    private static void FlushWord(StringBuilder word, List<string> words)
    {
        if (word.Length == 0)
        {
            return;
        }

        words.Add(word.ToString());
        word.Clear();
    }

    private void AppendWordPieces(string word, List<long> ids)
    {
        // Store UTF-16 boundaries so a supplementary character is never split in half.
        var boundaries = new List<int> { 0 };
        var length = 0;
        foreach (var rune in word.EnumerateRunes())
        {
            length += rune.Utf16SequenceLength;
            boundaries.Add(length);
        }

        if (boundaries.Count - 1 > _maxCharactersPerWord)
        {
            ids.Add(_unknownId);
            return;
        }

        var initialCount = ids.Count;
        var start = 0;
        while (start < boundaries.Count - 1)
        {
            var end = boundaries.Count - 1;
            var matched = false;
            while (end > start)
            {
                var piece = word[boundaries[start]..boundaries[end]];
                if (start != 0)
                {
                    piece = _continuationPrefix + piece;
                }

                if (_vocabulary.TryGetValue(piece, out var id))
                {
                    ids.Add(id);
                    matched = true;
                    break;
                }

                end--;
            }

            if (!matched)
            {
                // One unrecognised subword makes the entire word unknown.
                ids.RemoveRange(initialCount, ids.Count - initialCount);
                ids.Add(_unknownId);
                return;
            }

            start = end;
        }
    }

    private static bool IsOther(UnicodeCategory category) => category is
        UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.Surrogate
        or UnicodeCategory.PrivateUse or UnicodeCategory.OtherNotAssigned;

    private static bool IsPunctuation(int value, UnicodeCategory category) =>
        value is >= 33 and <= 47 or >= 58 and <= 64 or >= 91 and <= 96 or >= 123 and <= 126
        || category is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation
        or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation
        or UnicodeCategory.InitialQuotePunctuation or UnicodeCategory.FinalQuotePunctuation
        or UnicodeCategory.OtherPunctuation;

    // Matches Hugging Face tokenizers' BertNormalizer, including supplementary CJK ranges.
    private static bool IsChineseCharacter(int value) => value is
        >= 0x4e00 and <= 0x9fff or >= 0x3400 and <= 0x4dbf
        or >= 0x20000 and <= 0x2a6df or >= 0x2a700 and <= 0x2b73f
        or >= 0x2b740 and <= 0x2b81f or >= 0x2b920 and <= 0x2ceaf
        or >= 0xf900 and <= 0xfaff or >= 0x2f800 and <= 0x2fa1f;
}
