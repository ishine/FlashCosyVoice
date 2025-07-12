import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

from flashcosyvoice.config import CosyVoice2LLMConfig


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str, hf_config: CosyVoice2LLMConfig):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 1. load speech embedding + sos/taskid embedding + lm head
    embedding_weights = {}
    tmp_weights = torch.load(f"{path}/llm.pt", map_location="cpu")
    for k, v in tmp_weights.items():
        if k == "speech_embedding.weight":  # torch.Size([6564, 896])
            speech_embedding_size = hf_config.speech_vocab_size  # 6592
            # NOTE(xcsong): padding to 6592 for vllm tensor parallel
            if speech_embedding_size != v.shape[0]:  # [6564, 896] -> [6592, 896]
                assert speech_embedding_size >= v.shape[0], f"speech_embedding_size should be greater than or equal to {v.shape[0]}, but got {speech_embedding_size}"
                padded_v = torch.zeros(speech_embedding_size, v.shape[1], dtype=v.dtype, device=v.device)
                padded_v[:v.shape[0], :] = v
                v = padded_v
            embedding_weights["speech_embedding.weight"] = v
        elif k == "llm_embedding.weight":  # torch.Size([2, 896])
            assert v.shape[0] == 2, f"llm_embedding.weight should be of shape [2, 896], but got {v.shape}"
            embedding_weights["llm_embedding.weight"] = v
        elif k == "llm_decoder.weight":  # torch.Size([6564, 896])
            lm_head_size = hf_config.speech_vocab_size  # 6592
            # NOTE(xcsong): padding to 6592 for vllm tensor parallel
            if lm_head_size != v.shape[0]:  # [6564, 896] -> [6592, 896]
                assert lm_head_size >= v.shape[0], f"lm_head_size should be greater than or equal to {v.shape[0]}, but got {lm_head_size}"
                padded_v = torch.zeros(lm_head_size, v.shape[1], dtype=v.dtype, device=v.device)
                padded_v[:v.shape[0], :] = v
                v = padded_v
            param = model.get_parameter("lm_head.weight")
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, v)
        elif k == "llm_decoder.bias":  # torch.Size([6564])
            lm_head_size = hf_config.speech_vocab_size  # 6592
            # NOTE(xcsong): padding to 6592 for vllm tensor parallel
            if lm_head_size != v.shape[0]:  # [6564] -> [6592]
                assert lm_head_size >= v.shape[0], f"lm_head_size should be greater than or equal to {v.shape[0]}, but got {lm_head_size}"
                padded_v = torch.zeros(lm_head_size, dtype=v.dtype, device=v.device)
                padded_v[:v.shape[0]] = v
                v = padded_v
            param = model.get_parameter("lm_head.bias")
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, v)
        else:
            continue

    # 2. load Qwen2 backbone from CosyVoice-BlankEN
    for file in glob(os.path.join(f"{path}/CosyVoice-BlankEN", "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if weight_name == "model.embed_tokens.weight":
                    # NOTE(xcsong): merge text embedding, sos/taskid embedding, and speech embedding
                    text_embedding_weight = f.get_tensor(weight_name).cpu()  # [151936, 896]
                    sos_taskid_embedding_weight = embedding_weights["llm_embedding.weight"].cpu()  # [2, 896]
                    speech_embedding_weight = embedding_weights["speech_embedding.weight"].cpu()  # [6592, 896]
                    # print(f"text_embedding_weight: {text_embedding_weight.shape} {text_embedding_weight.device}")
                    # print(f"sos_taskid_embedding_weight: {sos_taskid_embedding_weight.shape} {sos_taskid_embedding_weight.device}")
                    # print(f"speech_embedding_weight: {speech_embedding_weight.shape} {speech_embedding_weight.device}")
                    final_embedding_weight = torch.cat([text_embedding_weight, sos_taskid_embedding_weight, speech_embedding_weight], dim=0)  # [158530, 896]
                    # print(f"final_embedding_weight: {final_embedding_weight.shape} {final_embedding_weight.device}")
                    param = model.get_parameter("model.embed_tokens.weight")
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, final_embedding_weight)
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        try:
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            break
                        except Exception as e:
                            print(e)
                            print(f"skip parameter (1): {weight_name}")
                            continue
                else:
                    try:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                    except Exception as e:
                        print(e)
                        print(f"skip parameter (2): {weight_name}")
                        continue
