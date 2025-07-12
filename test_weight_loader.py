import torch
import torch.distributed as dist
from safetensors import safe_open

from flashcosyvoice.config import Config
from flashcosyvoice.modules.qwen2 import Qwen2ForCausalLM
from flashcosyvoice.utils.loader import load_model
from flashcosyvoice.modules.flow import CausalMaskedDiffWithXvec
from flashcosyvoice.modules.hifigan import HiFTGenerator


file = "/mnt/user-ssd/songxingchen/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN/model.safetensors"
with safe_open(file, "pt", "cpu") as f:
    for weight_name in f.keys():
        print(weight_name)
        print(f.get_tensor(weight_name).shape)
        print("-" * 100)

print("=" * 100)

file = "/mnt/user-ssd/songxingchen/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B/llm.pt"
other_weights = torch.load(file, map_location="cpu")
for k, v in other_weights.items():
    print(k)
    print(v.shape)
    print("-" * 100)

print("=" * 100)

flow = CausalMaskedDiffWithXvec()
flow.load_state_dict(torch.load("/mnt/user-ssd/songxingchen/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B/flow.pt", map_location="cpu"), strict=True)
total_params = sum(p.numel() for p in flow.parameters())
print(f"total_params (flow): {total_params / 1000000}M")
# print(flow)

hift = HiFTGenerator()
hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load("/mnt/user-ssd/songxingchen/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B/hift.pt", map_location="cpu").items()}
hift.load_state_dict(hift_state_dict, strict=True)
total_params = sum(p.numel() for p in hift.parameters())
print(f"total_params (hift): {total_params / 1000000}M")
# print(hift)

dist.init_process_group(
    "nccl", "tcp://localhost:2333", world_size=1, rank=0
)
torch.cuda.set_device(0)
default_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
config = Config(model="/mnt/user-ssd/songxingchen/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B")
qwen2 = Qwen2ForCausalLM(config.hf_config)
load_model(qwen2, config.model, config.hf_config)
total_params = sum(p.numel() for p in qwen2.parameters())
print(f"total_params (qwen2): {total_params / 1000000}M")
print(qwen2)
