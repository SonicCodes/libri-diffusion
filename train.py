import sys
import os
import glob
from types import SimpleNamespace

import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize

from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
from tqdm.auto import tqdm
import numpy as np
# Add the necessary path to import DiffusionForcingAudio
sys.path.append("/home/ubuntu/libri-diff/diffusion-forcing")
from algorithms.diffusion_forcing import DiffusionForcingAudio
from datasets import load_dataset
from diffusers import AudioLDM2Pipeline

# Constants
MAX_WAV_VALUE = 32768.0

# Utility functions
def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(0), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(0)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True).real.float()
    spec = torch.abs(spec)
    
    spec = torch.einsum('oi,ib->bo', mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

NUM_SAMPLES = 2_000_000
NUM_BATCHES = 32

# Dataset and DataLoader
class LibriSpeechDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        # self.dataset = 
        self.reset()

    def __iter__(self):
        return self

    # def __len__(self):
    #     return NUM_SAMPLES

    def __next__(self):
        try:
            item = next(self.iterator)
            audio = (item["audio"]["array"][:16000*10])  # First 10 seconds
            # audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            audio = torch.FloatTensor(audio)
            mel = mel_spectrogram(audio, 1024, 64, 16000, 160, 1024, 0, 8000, center=False)
            return mel
        except StopIteration:
            # self.iterator = iter(self.dataset)  # Reset the iterator
            self.reset()
            
            raise StopIteration

    def reset(self):
        # self.dataset
        print("Epoch ended, resetting")
        self.iterator = iter(self.dataset.shuffle())

def handle_dataset_item(item):
    audio = (item["audio"]["array"][:16000*10])  # First 10 seconds
    # audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95
    audio = torch.FloatTensor(audio)
    mel = mel_spectrogram(audio, 1024, 64, 16000, 160, 1024, 0, 8000, center=False)
    return {
        "mel": mel
    }
def create_dataloader(batch_size):
    # audio_files = glob.glob("/home/ubuntu/libri-diff/LibriSpeech/**/*.wav", recursive=True)
    dataset = load_dataset("parler-tts/mls_eng_10k", streaming=True, split="train").shuffle().take(NUM_SAMPLES)
    print("Streaming ds with n_shards: ", dataset.n_shards)
    dataset = dataset.map(handle_dataset_item, remove_columns=["audio"])
    
    # dataset = dataset.to_iterable_dataset(num_shards=batch_size)
    def batch_collate_fn(batch):
        # max_length = max(max([len(x) for x in batch]), 320)
        mels = [x["mel"] for x in batch]
        # _eos_token = torch.randn(1, 64).to(batch[0].device) 
        # batch = [torch.cat([mel, _eos_token.repeat((max_length - mel.size(0)) + 1, 1)], dim=0) for mel in batch]
        batch = torch.stack(mels, dim=0)
        return batch
    
    return DataLoader(dataset, batch_size=batch_size, num_workers=batch_size, collate_fn=batch_collate_fn)

# Model configuration
def get_model_config():
    return SimpleNamespace(
        x_shape=(8, 8, 16),
        frame_stack=1,
        frame_skip=0,
        data_mean=0,
        data_std=1,
        external_cond_dim=0,
        context_frames=6,
        weight_decay=2e-3,
        warmup_steps=10000,
        optimizer_beta=[0.9, 0.99],
        uncertainty_scale=1,
        guidance_scale=0.0,
        chunk_size=1,
        scheduling_matrix="autoregressive",
        noise_level="random_all",
        causal=True,
        n_frames=31,
        diffusion=SimpleNamespace(
            objective="pred_v",
            beta_schedule="sigmoid",
            schedule_fn_kwargs={},
            clip_noise=7.0,
            use_snr=False,
            use_cum_snr=False,
            use_fused_snr=True,
            snr_clip=5.0,
            cum_snr_decay=0.96,
            timesteps=1000,
            sampling_timesteps=200,
            ddim_sampling_eta=0.0,
            stabilization_level=15,
            architecture=SimpleNamespace(
                network_size=64,
                attn_heads=16,
                attn_dim_head=64,
                dim_mults=[1, 4, 8, 16],
                resolution=16,
                attn_resolutions=[16, 32, 64, 98],
                use_init_temporal_attn=True,
                use_linear_attn=False,
                time_emb_type="rotary"
            )
        ),
        debug=True,
        metrics=[]
    )

def main():
    # Accelerator
    accelerator = Accelerator(
        mixed_precision='fp16',
        log_with="wandb"
    )
    
    # Set random seed for reproducibility
    set_seed(42)

    # Initialize wandb
    accelerator.init_trackers(
        project_name="diffusion-forcing-audio",
        config=vars(get_model_config())
    )

    # Create model, optimizer, and dataloader
    model = DiffusionForcingAudio(cfg=get_model_config())

    # model.load_state_dict(torch.load("checkpoints/epoch_last/weights.pt"))
    
    # Define optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=8e-5,
        weight_decay=0.001,  # Adjust this value as needed

    )
    
    dataloader = create_dataloader(batch_size=NUM_BATCHES)

    # Load AudioLDM2 pipeline
    repo_id = "anhnct/audioldm2_gigaspeech"
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = pipe.to(accelerator.device)

    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training loop
    num_epochs = 200  # Adjust as needed
    grad_norm_clip = 1.0  # Adjust as needed

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(
            total= (NUM_SAMPLES//NUM_BATCHES),
        desc=f"Epoch {epoch + 1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        nm_losses = 0
        for step, mel_batch in enumerate(dataloader):
            B, T, D = mel_batch.shape
            nm_losses += 1
            segment_size = 32
            final_augmented_size = mel_batch.shape[1] - (mel_batch.shape[1] % segment_size)
            mel_batch = mel_batch[:,:final_augmented_size, :].reshape(-1, 1, segment_size, 64).half()

            with torch.no_grad():
                mel_batch = pipe.vae.encode(mel_batch).latent_dist.sample() * pipe.vae.config.scaling_factor
            
            mel_batch = mel_batch.reshape(B, -1, *mel_batch.shape[1:])

            deets = model.training_step([mel_batch], step)
            loss = deets["loss"]

            accelerator.backward(loss)
            
            if grad_norm_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), grad_norm_clip)
            
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

            # Log to wandb
            accelerator.log({"train_loss": loss.item(), "epoch": epoch, "step": step})

        # End of epoch
        avg_loss = total_loss / nm_losses
        accelerator.print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if accelerator.is_main_process:
            checkpoint_dir = f"checkpoints/epoch_last"
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(model.state_dict(), f"{checkpoint_dir}/weights.pt")
            accelerator.save_state(checkpoint_dir)
            # accelerator.save_checkpoint(checkpoint_dir, model, optimizer, epoch, step)
            # wandb.save(f"{checkpoint_dir}/*")

    # End of training
    accelerator.end_training()

if __name__ == "__main__":
    main()