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
import s3tokenizer

from flashcosyvoice.config import Config
from flashcosyvoice.engine.llm_engine import LLMEngine
from flashcosyvoice.modules.flow import CausalMaskedDiffWithXvec
from flashcosyvoice.modules.hifigan import HiFTGenerator


class CosyVoice2(torch.nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        self.config = Config() if config is None else config

        self.audio_tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz").cuda().eval()

        self.llm = LLMEngine(self.config)

        self.flow = CausalMaskedDiffWithXvec()
        self.flow.load_state_dict(torch.load(f"{self.config.model}/flow.pt", map_location="cpu"), strict=True)
        self.flow.cuda().eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{self.config.model}/hift.pt", map_location="cpu").items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.cuda().eval()

    def forward(self, input_ids: list[list[int]], input_lens: list[int], spk_embs: torch.Tensor,
                mels: torch.Tensor, mels_lens: torch.Tensor, log_mels: torch.Tensor, log_mels_lens: torch.Tensor,
                infos: list[dict]):
        speech_ids, speech_ids_lens = self.audio_tokenizer(log_mels.cuda(), log_mels_lens.cuda())
        batch_size = speech_ids.shape[0]
        assert len(input_ids) == batch_size

        prompts = []
        for i in range(batch_size):
            speech_ids_i = speech_ids[i, :speech_ids_lens[i].item()].tolist()
            prompts.append(input_ids[i].extend(speech_ids_i))

        llm_outputs = self.llm.generate(prompts, self.config.sampling_params)

        generated_speech_ids = []
        generated_speech_ids_lens = []
        for o in llm_outputs:
            generated_speech_ids.append(o['token_ids'])
            generated_speech_ids_lens.append(len(o['token_ids']))

        flow_outputs, _ = self.flow(
            generated_speech_ids, generated_speech_ids_lens,
            speech_ids, speech_ids_lens,
            mels, mels_lens,
            spk_embs,
            streaming=False, finalize=True
        )

        hifigan_outputs, _ = self.hift(speech_feat=flow_outputs)

        return hifigan_outputs
