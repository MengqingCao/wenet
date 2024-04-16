import wenet
# from torchaudio._extension import _FFMPEG_EXT, _SOX_INITIALIZED
import torch_npu  # noqa: F401

model = wenet.load_model('chinese', device="npu:0")
result = model.transcribe(
    './dataset/common_language/common_voice_kpd/Chinese_China/dev/chch_dev_sp_2/common_voice_zh-CN_22053898.wav'
)
print(result['text'])
# import torch
# import torch_npu

# a = torch.tensor(1).to("npu:0")
