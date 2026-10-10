# Local model assets

The runtime model is [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE), whose model card describes a shared embedding space for 109 languages, including Arabic and English. Run this command from the project root to install or repair the pinned assets:

```powershell
.\scripts\download-model.ps1
```

The default download contains about 488 MB in `Models/LaBSE`. The C# API verifies these files at startup and copies them into the publish output. Runtime inference uses only local CPU resources and requires no Python or network connection.

The embedding pipeline follows the trained LaBSE architecture: cased WordPiece tokenization, BERT encoder, CLS-token representation, trained `768 → 768` dense projection with tanh, and L2 normalization. Cosine similarity compares the resulting unit-length embeddings. The service limits submitted names to 128 tokens, including CLS/SEP.

| Local asset | Bytes | SHA256 |
| --- | ---: | --- |
| `LaBSE/model_quantized.onnx` | 471,527,949 | `2d209704455c8e5718dbef9c46ca22c823eb35256b74b4122d90845310c7ef82` |
| `LaBSE/tokenizer.json` | 13,631,919 | `09216b42d2697b7b4a26ac05ff09ba8bf52dc19b896c5ceee8bbff9f39055322` |
| `LaBSE/dense.safetensors` | 2,362,528 | `f866c945fa2147ce4940c8ead0b4b20af4217e0f4664bd35f41c2a0967c4fe77` |

The encoder and tokenizer come from Xenova/LaBSE revision `147002d501fc76ecd313c631999ea796520fea74`: [ONNX encoder](https://huggingface.co/Xenova/LaBSE/resolve/147002d501fc76ecd313c631999ea796520fea74/onnx/model_quantized.onnx), [tokenizer](https://huggingface.co/Xenova/LaBSE/resolve/147002d501fc76ecd313c631999ea796520fea74/tokenizer.json). The trained projection comes from sentence-transformers/LaBSE revision `836121a0533e5664b21c7aacc5d22951f2b8b25b`: [dense weights](https://huggingface.co/sentence-transformers/LaBSE/resolve/836121a0533e5664b21c7aacc5d22951f2b8b25b/2_Dense/model.safetensors).

For independent C# fixture checks, use `.\scripts\download-model.ps1 -IncludeTestAssets`. This additionally downloads the legacy DistilUSE tokenizer and `768 → 512` dense projection into the `Models` root. They exercise the shared tokenizer and projection reader and are not used by the default runtime model or deployed with it. No legacy ONNX model is required for those checks.

The upstream LaBSE model is [Apache 2.0 licensed](https://huggingface.co/sentence-transformers/LaBSE/blob/836121a0533e5664b21c7aacc5d22951f2b8b25b/README.md). This project's MIT license applies to the application code and does not replace the model license. Model assets are excluded from Git.
