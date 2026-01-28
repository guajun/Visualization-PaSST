from hear21passt.base import get_basic_model,get_model_passt
import torch
# get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
model = get_basic_model(mode="logits")
print(model.mel) # Extracts mel spectrogram from raw waveforms.
print(model.net) # the transformer network.

# example inference
model.eval()
model = model.cuda()
with torch.no_grad():
    # audio_wave has the shape of [batch, seconds*32000] sampling rate is 32k
    # example audio_wave of batch=3 and 10 seconds
    audio = torch.ones((3, 32000 * 10))*0.5
    audio_wave = audio.cuda()
    logits=model(audio_wave) 

