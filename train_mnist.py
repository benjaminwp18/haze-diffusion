import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
# Import required classes from the DDPM module of TorchDiff
from torchdiff.ddpm import ForwardDDPM, ReverseDDPM, VarianceSchedulerDDPM, TrainDDPM
# Import utility functions from the TorchDiff utils module
from torchdiff.utils import NoisePredictor, Metrics


# Define transformations: convert images to tensors and normalize (mean=0.5, std=0.5) for grayscale images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print('Loading datasets')
# Load the FashionMNIST training dataset (28x28 grayscale images)
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Load the FashionMNIST test dataset (28x28 grayscale images)
test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print('Making loaders')
train_subset_indices = torch.randperm(len(train_dataset))[:200]
test_subset_indices = torch.randperm(len(test_dataset))[:10]
train_subset = Subset(train_dataset, train_subset_indices)
test_subset = Subset(test_dataset, test_subset_indices)
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(test_subset, batch_size=10, shuffle=False, drop_last=False)

print('Making noise predictor')
# Initialize the NoisePredictor for the DDPM model with parameters for grayscale images
noise_predictor = NoisePredictor(
        in_channels=1,  # Single channel for grayscale images in the training data
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

# Set up the AdamW optimizer for the NoisePredictor parameters with a learning rate of 1e-3
optimizer = torch.optim.AdamW(
    [p for p in noise_predictor.parameters()], lr=1e-4
)

# Initialize the Mean Squared Error (MSE) loss function
loss = nn.MSELoss()

# Configure the Metrics class for evaluation on CPU (GPUs are recommended for actual training)
metrics = Metrics(
    device="cpu",  # Using CPU for this tutorial, but GPUs are recommended for training diffusion models
    fid=False,
    metrics=True,
    lpips_=True
)

print('Scheduling variance')
# Initialize DDPM variance-scheduler for the noise schedule
variance_scheduler_ddpm = VarianceSchedulerDDPM(
    num_steps=500,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=False, # Whether the beta schedule is trainable
    beta_method="linear"
)

print('Creating forward/backward passes')
# Set up the forward and reverse diffusion process
forward_ddpm = ForwardDDPM(variance_scheduler_ddpm)
reverse_ddpm = ReverseDDPM(variance_scheduler_ddpm)

print('Making trainer')
# Configure the DDPM trainer for model training
train_ddpm = TrainDDPM(
    noise_predictor=noise_predictor,
    forward_diffusion=forward_ddpm,
    reverse_diffusion=reverse_ddpm,
    data_loader=train_loader,
    optimizer=optimizer,
    objective=loss,
    val_loader=val_loader,
    max_epochs=50,
    device="cpu",
    conditional_model=None,
    metrics_=metrics,
    bert_tokenizer=None,
    max_token_length=77,
    store_path="test_ddpm",
    val_frequency=6,
    image_output_range=(-1.0, 1.0),
    normalize_output=True,
    use_ddp=False,
    grad_accumulation_steps=2,
    log_frequency=1,
    use_compilation=False
)

print('Training')
# start trining
train_losses, best_val_loss = train_ddpm()

