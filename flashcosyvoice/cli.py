# Copyright (c) 2025 Tsinghua Univ. (authors: Xingchen Song)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Example Usage

"""

import argparse
import json
import onnxruntime
import os

import torch
import torch.distributed as dist
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
import regex

import ttsfrd
from cosyvoice_ttsfrd import get_resource_path
from transformers import AutoTokenizer

import s3tokenizer

from flashcosyvoice.cosyvoice2 import CosyVoice2
from flashcosyvoice.utils.audio import mel_spectrogram


def is_only_punctuation(text):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r'^[\p{P}\p{S}]*$'
    return bool(regex.fullmatch(punctuation_pattern, text))


class AudioDataset(Dataset):

    def __init__(self, data_list, model_path):
        self.datas = []
        self.text_norm = ttsfrd.TtsFrontendEngine()
        self.text_norm.initialize(get_resource_path())
        self.text_norm.set_lang_type('pinyinvg')

        """Example data_list:
        ```
        {"key": "uttid_1", "prompt_text": "你好，我是小明。", "text": "你好，我是小红。", "prompt_wav": "data/audio/00000000.wav"}
        {"key": "uttid_2", "prompt_text": "你好，我是小红。", "text": "你好，我是小明。", "prompt_wav": "data/audio/00000001.wav"}
        ```
        """
        missing = 0
        with open(data_list, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)
            for line in tqdm(lines, desc='Loading data'):
                data = json.loads(line.strip())
                valid = True
                for k in ['key', 'prompt_text', 'text', 'prompt_wav']:
                    if k not in data:
                        valid = False
                        break
                    if data[k] is None:
                        valid = False
                        break
                if not os.path.exists(data['prompt_wav']):
                    valid = False
                if valid:
                    texts = [i["text"] for i in json.loads(self.text_norm.do_voicegen_frd(data['text'].strip()))["sentences"]]
                    texts = [i for i in texts if not is_only_punctuation(i)]
                    key, suffix = data['key'], 0
                    for text in texts:
                        data['key'] = f"{key}/part{suffix}"
                        data['text'] = text
                        suffix += 1
                        self.datas.append(data)
                else:
                    missing += 1
        print(f'Loaded {total_lines} lines, found {missing} missing lines, split valid lines into {len(self.datas)} samples (due to text normalization).')

        self.special_tokens = {
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "additional_special_tokens": [
                "<|im_start|>", "<|im_end|>", "<|endofprompt|>", "[breath]", "<strong>", "</strong>",
                "[noise]", "[laughter]", "[cough]", "[clucking]", "[accent]", "[quick_breath]",
                "<laughter>", "</laughter>", "[hissing]", "[sigh]", "[vocalized-noise]", "[lipsmack]", "[mn]",
            ],
        }
        self.text_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/CosyVoice-BlankEN")
        self.text_tokenizer.add_special_tokens(self.special_tokens)

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(f"{model_path}/campplus.onnx", sess_options=option,
                                                      providers=["CPUExecutionProvider"])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]

        # 1. feature for s3tokenizer
        audio = s3tokenizer.load_audio(data['prompt_wav'], sr=16000)
        if audio.shape[0] / 16000 > 30:
            print(
                f'do not support extract speech token for audio longer than 30s, file_path: {data["prompt_wav"]}'  # noqa
            )
            log_mel = torch.zeros(128, 0)
        else:
            log_mel = s3tokenizer.log_mel_spectrogram(audio)

        # 2. feature for speaker embedding
        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = self.spk_model.run(
            None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
        )[0].flatten().tolist()

        # 3. feature for flow
        audio = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)(audio.unsqueeze(0))
        mel = mel_spectrogram(audio).squeeze(dim=0).transpose(0, 1).unsqueeze(dim=0)
        mel_len = mel.shape[1]

        # 4. feature for llm
        prompt_texts = [i["text"] for i in json.loads(self.text_norm.do_voicegen_frd(data['prompt_text'].strip()))["sentences"]]
        prompt_text = ''.join(prompt_texts)
        prompt_text_ids = self.text_tokenizer([prompt_text], return_tensors="pt")["input_ids"][0].cpu().tolist()
        texts = [i["text"] for i in json.loads(self.text_norm.do_voicegen_frd(data['text'].strip()))["sentences"]]
        text = ''.join(texts)
        text_ids = self.text_tokenizer([text], return_tensors="pt")["input_ids"][0].cpu().tolist()
        item = {
            "input_ids": prompt_text_ids + text_ids, "input_len": len(prompt_text_ids) + len(text_ids),
            "spk_emb": spk_emb, "mel": mel, "mel_len": mel_len, "log_mel": log_mel, "info": data,
        }
        return item


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    input_lens = [item["input_len"] for item in batch]
    spk_embs = [item["spk_emb"] for item in batch]
    mels = [item["mel"] for item in batch]
    mels_lens = [item["mel_len"] for item in batch]
    log_mels = [item["log_mel"] for item in batch]
    log_mels, log_mels_lens = s3tokenizer.padding(log_mels)
    infos = [item["info"] for item in batch]
    return {
        "input_ids": input_ids, "input_lens": input_lens, "spk_embs": spk_embs,
        "mels": mels, "mels_lens": mels_lens, "log_mels": log_mels, "log_mels_lens": log_mels_lens, "infos": infos,
    }


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    print(f'Inference on multiple gpus, this gpu {local_rank}, rank {rank}, world_size {world_size}')
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank


def get_args():
    parser = argparse.ArgumentParser(description='FlashCosyVoice')
    parser.add_argument('--model_path',
                        required=True,
                        type=str,
                        help='model path')
    parser.add_argument('--data_list',
                        required=True,
                        type=str,
                        help='data list')
    parser.add_argument('--output_dir',
                        required=True,
                        type=str,
                        help='dir to save result')
    parser.add_argument('--batch_size',
                        required=True,
                        type=int,
                        help='batch size (per-device) for inference')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='workers for dataloader')
    parser.add_argument('--prefetch',
                        type=int,
                        default=5,
                        help='prefetch for dataloader')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    assert (torch.cuda.is_available())
    world_size, local_rank, rank = init_distributed()

    device = torch.device(args.device)

    model = CosyVoice2(
        model_path=args.model_path,
        device=device,
    )

    dataset = AudioDataset(args.data_list, args.model_path)
    sampler = DistributedSampler(dataset,
                                 num_replicas=world_size,
                                 rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                            sampler=sampler, shuffle=False, prefetch_factor=args.prefetch, collate_fn=collate_fn)

    total_steps = len(dataset)

    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Processing", unit="wavs")

    writer = open(f"{args.output_dir}/part_{rank + 1}_of_{world_size}", "w")

    for batch in dataloader:
        hifigan_outputs = model(**batch)
        for i in range(len(hifigan_outputs)):
            batch['infos'][i]['wav'] = f"{args.output_dir}/{batch['infos'][i]['key']}.wav"
            torchaudio.save(batch['infos'][i]['wav'], hifigan_outputs[i].unsqueeze(0), 24000)
            writer.write(f"{json.dumps(batch['infos'][i], ensure_ascii=False)}\n")
        if rank == 0:
            progress_bar.update(world_size * len(batch["input_ids"]))

    if rank == 0:
        progress_bar.close()
    writer.close()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
