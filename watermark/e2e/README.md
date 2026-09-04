# E2E-LLM-Watermark model files

This integration uses the encoder/detector checkpoint released by the
official [E2E-LLM-Watermark repository](https://github.com/KahimWong/E2E-LLM-Watermark).

Download `35000.pth` and place it at the path configured by
`checkpoint_path` in `config/E2E.json`:

```bash
mkdir -p watermark/e2e/model
wget -O watermark/e2e/model/35000.pth \
  https://github.com/KahimWong/E2E-LLM-Watermark/raw/master/ckpt/35000.pth
```

Expected SHA-256:

```text
3dc85de9d81fda064de179731e57c6c7c9d3b3d868ebe7a616edc08e23686174
```

The released checkpoint was trained with `facebook/opt-1.3b` embeddings.
Generation can use a different Hugging Face causal language model; candidate
text is converted through the configured reference tokenizer when necessary.
