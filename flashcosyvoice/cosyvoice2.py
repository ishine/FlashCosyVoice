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
import torch
import time
import s3tokenizer

from flashcosyvoice.config import Config, SamplingParams
from flashcosyvoice.engine.llm_engine import LLMEngine
from flashcosyvoice.modules.flow import CausalMaskedDiffWithXvec
from flashcosyvoice.modules.hifigan import HiFTGenerator


class CosyVoice2(torch.nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        self.config = Config() if config is None else config

        self.audio_tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz").cuda().eval()

        self.llm = LLMEngine(**self.config.__dict__)

        self.flow = CausalMaskedDiffWithXvec()
        self.flow.load_state_dict(torch.load(f"{self.config.model}/flow.pt", map_location="cpu"), strict=True)
        self.flow.cuda().eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{self.config.model}/hift.pt", map_location="cpu").items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.cuda().eval()

    @torch.inference_mode()
    def forward(
        self, prompt_mels_for_llm: torch.Tensor, prompt_mels_lens_for_llm: torch.Tensor,
        prompt_text_tokens_for_llm: list[list[int]], text_tokens_for_llm: list[list[int]],
        prompt_mels_for_flow: torch.Tensor, prompt_mels_lens_for_flow: torch.Tensor,
        spk_emb_for_flow: torch.Tensor,
        sampling_params: SamplingParams | list[SamplingParams] = None,
    ):
        timing_stats = {}

        # Audio tokenization
        start_time = time.time()
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(
            prompt_mels_for_llm.cuda(), prompt_mels_lens_for_llm.cuda()
        )
        timing_stats['audio_tokenization'] = time.time() - start_time

        batch_size = prompt_speech_tokens.shape[0]
        assert len(prompt_text_tokens_for_llm) == batch_size

        # Prepare LLM inputs
        start_time = time.time()
        valid_prompt_speech_tokens = []
        inputs = []
        for i in range(batch_size):
            speech_tokens_i = prompt_speech_tokens[i, :prompt_speech_tokens_lens[i].item()].tolist()
            valid_prompt_speech_tokens.append(speech_tokens_i)
            inputs.append([self.config.hf_config.speech_vocab_size] + prompt_text_tokens_for_llm[i] + text_tokens_for_llm[i]
                           + [self.config.hf_config.speech_vocab_size + 1] + speech_tokens_i)
        timing_stats['prepare_llm_inputs'] = time.time() - start_time

        # LLM generation
        start_time = time.time()
        llm_outputs = self.llm.generate(inputs, self.config.sampling_params if sampling_params is None else sampling_params)
        timing_stats['llm_generation'] = time.time() - start_time

        # Prepare Flow inputs
        start_time = time.time()
        flow_inputs = []
        flow_inputs_lens = []
        for i, o in enumerate(llm_outputs):
            generated_speech_tokens = o['token_ids'][:-1]  # ignore last eos
            prompt_speech_tokens = valid_prompt_speech_tokens[i]
            flow_inputs.append(torch.tensor(prompt_speech_tokens + generated_speech_tokens))
            flow_inputs_lens.append(len(prompt_speech_tokens) + len(generated_speech_tokens))
        flow_inputs = torch.nn.utils.rnn.pad_sequence(flow_inputs, batch_first=True, padding_value=0)
        flow_inputs_lens = torch.tensor(flow_inputs_lens)
        timing_stats['prepare_flow_inputs'] = time.time() - start_time

        # Flow generation
        start_time = time.time()
        generated_mels, generated_mels_lens = self.flow(
            flow_inputs.cuda(), flow_inputs_lens.cuda(),
            prompt_mels_for_flow.cuda(), prompt_mels_lens_for_flow.cuda(), spk_emb_for_flow.cuda(),
            streaming=False, finalize=True
        )
        timing_stats['flow_generation'] = time.time() - start_time

        # HiFi-GAN generation
        start_time = time.time()
        generated_wavs = []
        for i in range(batch_size):
            mel = generated_mels[i, :, prompt_mels_lens_for_flow[i].item():generated_mels_lens[i].item()].unsqueeze(0)
            wav, _ = self.hift(speech_feat=mel)
            generated_wavs.append(wav)
        timing_stats['hifigan_generation'] = time.time() - start_time

        # Calculate total time
        timing_stats['total'] = sum(timing_stats.values())

        return generated_wavs, timing_stats
