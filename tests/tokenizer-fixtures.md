`tokenizer-fixtures.json` contains unpadded input IDs captured independently from
Hugging Face Transformers.js 3.8.1's `DistilBertTokenizer` with special tokens enabled
and truncation disabled. The tokenizer loaded `Models/tokenizer.json` directly;
no model inference or Python was used to generate these fixtures.

Tokenizer source: [Xenova/distiluse-base-multilingual-cased-v2](https://huggingface.co/Xenova/distiluse-base-multilingual-cased-v2/tree/cad454171d918d9873a2701ba245054b6c1760dd).
Oracle: [Transformers.js 3.8.1 tokenizers](https://github.com/huggingface/transformers.js/blob/3.8.1/src/tokenizers.js).

The corpus covers Arabic and English names, Arabic diacritics, preserved case and
accents, punctuation, whitespace, removed controls, exact special strings, Chinese
characters, an unknown emoji, and WordPiece's 100-character word limit.
The empty and whitespace-only cases test tokenizer behavior; the HTTP endpoint
should reject empty names separately.

Supplementary CJK splitting follows the native
[Hugging Face BertNormalizer](https://github.com/huggingface/tokenizers/blob/v0.15.2/tokenizers/src/normalizers/bert.rs),
rather than Transformers.js 3.8.1's UTF-16 character loop. The C# implementation
walks Unicode runes and never splits surrogate pairs during WordPiece matching.
