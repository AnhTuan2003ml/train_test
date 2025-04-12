import os
import torch
import librosa
import numpy as np
from utils.wav_io import save_wav
from models.clap_encoder import CLAP_Encoder
import lightning.pytorch as pl
from utils import load_ss_model, parse_yaml

def inference(checkpoint_path, audio_file, caption, config_yaml='config/audiosep_base.yaml', device="cuda"):
    # Load configurations from YAML
    configs = parse_yaml(config_yaml)

    # Load model
    query_encoder = CLAP_Encoder().eval()

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
        query_encoder=query_encoder
    ).to(device)

    print(f'Loading audio file: {audio_file}')

    # Load the audio mixture file (mono)
    mixture, fs = librosa.load(audio_file, sr=16000, mono=True)

    # Declipping if necessary
    max_value = np.max(np.abs(mixture))
    if max_value > 1:
        mixture *= 0.9 / max_value

    # Create conditions for the model (using the caption)
    conditions = pl_model.query_encoder.get_query_embed(
        modality='text',
        text=[caption],
        device=device
    )
    
    # Prepare the input for the model
    input_dict = {
        "mixture": torch.Tensor(mixture)[None, None, :].to(device),
        "condition": conditions,
    }

    # Run the model to get the separated segment
    sep_segment = pl_model.ss_model(input_dict)["waveform"]
    
    # Convert back to numpy for saving the separated audio
    sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()

    # Save the separated audio to output file
    output_file = os.path.splitext(audio_file)[0] + '_separated.wav'
    save_wav(sep_segment, output_file)
    print(f'Separated audio saved to: {output_file}')


if __name__ == '__main__':
    # Set the path to the checkpoint
    checkpoint_path = 'audiosep_16k,baseline,step=200000.ckpt'

    # Specify the input audio file and caption
    audio_file = 'input_mixture.wav'  # Replace with your audio file path
    caption = 'a man is speaking'  # Replace with the description of the target source

    # Perform inference
    inference(checkpoint_path, audio_file, caption, device="cuda")
