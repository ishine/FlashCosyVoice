<div align="center">

<img src="assets/flashcosyvoice.jpg" alt="Description" width="25%" />

# FlashCosyVoice

<p><em>A lightweight vLLM implementation built from scratch for CosyVoice.</em></p>

</div>

## Key Features

- ⚡️ **Blazing-fast offline inference without complex dependencies**
  - Built-in Prefix caching, Torch compilation, CUDA graph, etc.
  - Achieve equal or even better inference speed without installing vLLM.
- 📖 **Readable codebase with easier hackability**
  - Clean & minimal implementation of CosyVoice in pure Python code.
  - Allow you to DIY any part, such as implementing [ras_sample](./flashcosyvoice/modules/sampler.py) (repetition aware sample) that is hard to be supported in the standard vLLM library.

## Installation

If you don't need any modification:

```sh
pip install git+https://github.com/xingchensong/FlashCosyVoice
```

If you want to do some DIY:

```sh
git clone https://github.com/xingchensong/FlashCosyVoice
cd FlashCosyVoice
pip install -e .
```

## Model Download

```sh
# you might need `sudo apt-get install git-lfs` before download this model
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git
```

## Example Data Format

```json
{"key": "uttid_1", "prompt_text": "你好，我是小明。", "text": "你好，我是小红。", "prompt_wav": "/mnt/data/audio/00000000.wav", "wav": "/mnt/data/audio_synthetic/uttid_1.wav"}
...
{"key": "uttid_2", "prompt_text": "你好，我是小红。", "text": "你好，我是小明。", "prompt_wav": "/mnt/data/audio/00000001.wav", "wav": "/mnt/data/audio_synthetic/uttid_2.wav"}
...
```

- `key` is the key of this sample.
- `prompt_text` is the text used for prompt.
- `text` is the text to be generated.
- `prompt_wav` is the audio used for prompt.
- `wav` is the path to save the generated audio (we highly recommend to pre-define the save path before running the script).

## Example Usage

FlashCosyVoice is built for distributed offline batch inference, enjoy ultra speed!

```sh
# 1 node 1 gpu, try to decrease `batch_size_dataloader` & `batch_size_flow` if OOM
torchrun --nproc_per_node=1 --nnodes=1 \
     --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    `which flashcosyvoice` \
        --model_path "path to your CosyVoice2-0.5B" \
        --data_list "path to your data.jsonl" \
        --batch_size_dataloader 1024 \
        --batch_size_flow 32 \
        --num_workers 8 \
        --fp16_flow \
        --prefetch 32

# 1 node 8 gpu, try to decrease `batch_size_dataloader` & `batch_size_flow` if OOM
torchrun --nproc_per_node=8 --nnodes=1 \
     --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    `which flashcosyvoice` \
        --model_path "path to your CosyVoice2-0.5B" \
        --data_list "path to your data.jsonl" \
        --batch_size_dataloader 1024 \
        --batch_size_flow 32 \
        --num_workers 8 \
        --fp16_flow \
        --prefetch 32
```

## Performance Benchmark

|  Method  | RTF | Relative speed up | WERs on [CV3-Eval](https://github.com/FunAudioLLM/CV3-Eval) Zero-Shot Test Set (zh/en/ja/ko) |
|:------:|:--------------:|:-----:|:--------------:|
|  CV2 results reported in CV3 paper (Table 5)  |   N/A   |   N/A         | 4.08 / 6.32 / 9.13 / 19.7 |
|  [[cosyvoice/pytorch_example.py]](https://github.com/FunAudioLLM/CosyVoice?tab=readme-ov-file#cosyvoice2-usage) (fp32 llm + fp32 flow) |   0.487    |    1x     |   4.17 / 6.25 / 12.85 / 8.25 |
|  [[cosyvoice/pytorch_example.py]](https://github.com/FunAudioLLM/CosyVoice?tab=readme-ov-file#cosyvoice2-usage) (fp16 llm + fp16 flow) |   0.554    |    0.9x  |   3.95 / 6.21 / 9.82 / 9.44  |
|  [[cosyvoice/vllm_example.py]](https://github.com/FunAudioLLM/CosyVoice/blob/main/vllm_example.py) (bf16 llm + fp16 flow)              |   0.167    |    ~3x     |   4.28 / 6.49 / 8.40 / 9.59  |
|  FlashCosyVoice (bf16 llm + fp32 flow)  |   0.081   |   ~6x    |  3.89 / 6.11 / 8.50 / 10.27 |
|  FlashCosyVoice (bf16 llm + fp16 flow)  |   0.055   |   ~9x    | 3.88 / 6.11 / 8.47  / 10.33 |

Conclusion

- Compared with native PyTorch inference, FlashCosyVoice achieves 9x speed-up while maintaining similar WERs.
- Due to the lack of `ras_sample`, The `cosyvoice/vllm_example.py` is more unstable and more harmful to WERs than FlashCosyVoice in common languages (Chinese/English).

Test Configuration

- Hardware: 1 * H800 (80GB)
- Model: CosyVoice2-0.5B
- Total Requests: 2000 (500 for each in [zh, en, ja, ko])
- seed: 1986
- text_frontend: False
- batch_size: 1024(dataloader)/32(flow) for FlashCosyVoice and 1 for others, (When testing FlashCosyVoice, we repeated the request 100 times to obtain a more accurate RTF.)

## TODO

- [ ] Support online generation for RL training
- [ ] Support Ray for ultra-large-scale speech generation
- [ ] CosyVoice3 (when it is released, hhh)

## Acknowledge

- This repo is highly motivated by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). We drew on the design of the LLM engine and made necessary adaptations for CosyVoice.
- This repo also benefits from [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer), [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), [CV3-Eval](https://github.com/FunAudioLLM/CV3-Eval)
