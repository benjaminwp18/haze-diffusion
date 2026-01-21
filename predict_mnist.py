import torch
# Import required classes from the DDPM module of TorchDiff
from torchdiff.ddpm import ReverseDDPM, VarianceSchedulerDDPM, SampleDDPM
# Import utility functions from the TorchDiff utils module
from torchdiff.utils import NoisePredictor

MODEL_PATH = 'test_ddpm/ddpm_epoch_6.pth'

# Initialize DDPM variance-scheduler for the noise schedule
variance_scheduler_ddpm = VarianceSchedulerDDPM(
    num_steps=500,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=False, # Whether the beta schedule is trainable
    beta_method="linear"
)

# Initialize a new NoisePredictor with the same architecture as the trained model
new_noise_predictor = NoisePredictor(
    in_channels=1,  # Single channel for grayscale images
    down_channels=[16, 32],
    mid_channels=[32, 32],
    up_channels=[32, 16],
    down_sampling=[True, True],
    time_embed_dim=32,
    y_embed_dim=32,
    num_down_blocks=2,
    num_mid_blocks=2,
    num_up_blocks=2,
    down_sampling_factor=2
)

# Initialize DDPM hyperparameters for the noise schedule
new_variance_scheduler_ddpm = VarianceSchedulerDDPM(
    num_steps=500,
    beta_start=1e-4,
    beta_end=0.02,
    beta_method="linear"
)

# Set up the new reverse diffusion process for sampling
new_reverse_ddpm = ReverseDDPM(variance_scheduler_ddpm)


# Load the trained model checkpoint from file
checkpoint = torch.load(MODEL_PATH, map_location='cpu')

# Load the trained parameters into the new NoisePredictor and set to evaluation mode
new_noise_predictor.load_state_dict(checkpoint['model_state_dict_noise_predictor'])
new_noise_predictor.eval()

new_reverse_ddpm.variance_scheduler.load_state_dict(checkpoint['variance_scheduler_model'])

# sampling process
sampler = SampleDDPM(
    reverse_diffusion=new_reverse_ddpm,
    noise_predictor=new_noise_predictor,
    image_shape=(28, 28),
    conditional_model=None,
    tokenizer="bert-base-uncased",
    max_token_length=77,
    batch_size=10, # generate 10 new images
    in_channels=1,
    device="cpu",
    image_output_range=(-1, 1)
)

gen_imgs = sampler(save_images=True, save_path="ddpm_generated") # the created images will be stored in the given directory