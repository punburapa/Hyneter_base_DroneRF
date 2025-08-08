import torch
from torch import nn, einsum
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

       
    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x    


class CNN(nn.Module):
    def __init__(self, in_channels, embed_dim, stride1=1):
        super().__init__()
        padding_3x3 = 1
        padding_5x5 = 2

        self.out_channels = embed_dim//3

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=stride1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU()
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1), # 1x1 to potentially reduce channels
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=stride1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 3: 1x1 Convolution followed by two 3x3 Convolutions (effective 5x5 receptive field)
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1), # 1x1 to potentially reduce channels
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1), # First 3x3
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=stride1), # Second 3x3
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        x = torch.cat((branch1x1, branch3x3, branch5x5), dim=1)

        return x


class MultiGranularitySummingBlock(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, stride1: int = 1):
        super().__init__()
        padding_3x3 = 1
        padding_5x5 = 2

        self.out_channels = embed_dim//3

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=stride1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU()
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=stride1, padding=padding_3x3, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU()
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=5, stride=stride1, padding=padding_5x5, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU()
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        x = torch.cat((branch1x1, branch3x3, branch5x5), dim=1)

        return x
    

class DualSwitch_SwapOnly(nn.Module):
    def __init__(self):
        super(DualSwitch_SwapOnly, self).__init__()

    def _switch_adjacent(self, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Switches adjacent elements along a specified dimension.
        If the dimension size is odd, the last element remains untouched.
        Returns a new tensor.
        """
        size = input_tensor.shape[dim]
        output_tensor = input_tensor.clone() # Start with a copy of the input

        # Determine the largest even size that can be fully swapped
        swappable_size = (size // 2) * 2

        if swappable_size > 0:
            # Create slices for even and odd indices within the swappable part
            slices_even_part = [slice(None)] * input_tensor.ndim
            slices_odd_part = [slice(None)] * input_tensor.ndim
            
            slices_even_part[dim] = slice(0, swappable_size, 2)  # 0, 2, 4, ...
            slices_odd_part[dim] = slice(1, swappable_size + 1, 2) # 1, 3, 5, ...

            # Perform the swap on the output_tensor
            output_tensor[slices_even_part] = input_tensor[slices_odd_part]
            output_tensor[slices_odd_part] = input_tensor[slices_even_part]
            
        return output_tensor

    def _switch_interlaced(self, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Switches interlaced blocks of 2 elements along a specified dimension.
        If the dimension size is not a multiple of 4, the trailing elements remain untouched.
        Returns a new tensor.
        """
        size = input_tensor.shape[dim]
        
        indices = torch.arange(size, device=input_tensor.device)
        new_indices = indices.clone() # Initialize with identity permutation

        # Determine the largest size that is a multiple of 4 and can be fully swapped
        swappable_size = (size // 4) * 4

        for i in range(0, swappable_size, 4):
            # Swap blocks of 2: [i, i+1] goes to [i+2, i+3] positions
            new_indices[i:i+2] = indices[i+2:i+4]
            # And [i+2, i+3] goes to [i, i+1] positions
            new_indices[i+2:i+4] = indices[i:i+2]
        
        # Apply the permutation using advanced indexing, which creates a new tensor
        all_slices = [slice(None)] * input_tensor.ndim
        all_slices[dim] = new_indices # Apply the reordered indices to the specified dimension
        return input_tensor[all_slices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a sequence of column and row switching operations on the feature map.

        Following the convention:
        - Columns refer to the Height (H) dimension (dim=2).
        - Rows refer to the Width (W) dimension (dim=3).
        
        The operations are:
        1. Adjacent column switching (on H).
        2. Adjacent row switching (on W).
        3. Interlaced column switching (on H, swaps blocks of 2).
        4. Interlaced row switching (on W, swaps blocks of 2).

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).
                              No internal dimension checks are performed;
                              trailing elements in odd/non-multiple-of-4 dimensions
                              will be left untouched by the respective operations.

        Returns:
            torch.Tensor: The feature map after all switching operations.
                          The output shape is identical to the input shape.
        """

        # Step 1: Adjacent columns switch (Columns is H, so dim=2)
        x = self._switch_adjacent(x, dim=2)
        
        # Step 2: Adjacent rows switch (Rows is W, so dim=3)
        x = self._switch_adjacent(x, dim=3)
        
        # Step 3: Interlaced columns switch (Columns is H, so dim=2)
        x = self._switch_interlaced(x, dim=2)

        # Step 4: Interlaced rows switch (Rows is W, so dim=3)
        x = self._switch_interlaced(x, dim=3)
        
        return x
    

class HyneterModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, Conv_layers,TB_layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        self.mg = MultiGranularitySummingBlock(in_channels=in_channels, embed_dim=hidden_dimension, stride1=1)

        self.Conv_layers = nn.ModuleList([])
        for _ in range(Conv_layers):
            self.Conv_layers.append(
                CNN(in_channels=hidden_dimension,embed_dim=hidden_dimension)
            )

        

        self.TB_layers = nn.ModuleList([])
        for _ in range(TB_layers):
            self.TB_layers.append(
                TransformerBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
               )          

    def forward(self, x):
        print("\nHyneterModule: HNB: Patch -> Conv -> TB")
        print("HyneterModule forward pass")
        
        
        print(f"\nInput shape before Multigranularity CNN: {x.shape}") # Batch size, Channels, Height, Width
        x = self.mg(x)
        print(f"Input shape after Multigranularity CNN: {x.shape}") # Batch size, Channels, Height, Width

        x_conv_path = x
        x_tb_path = x

        x_tb_path = x_tb_path.permute(0, 2, 3, 1) # Change to (batch_size, Height, Width, Channels)

        print("X(CNN) shape before Conv layers:", x_conv_path.shape) # Batch size, Channels, Height, Width
        for block in self.Conv_layers:
            x_conv_path = block(x_conv_path)
        print(f"X(CNN) shape after Conv layers: {x_conv_path.shape}") # Batch size, Channels, Height, Width


        print("X(TB) shape before Transformer Blocks:", x_tb_path.shape) # Batch size, Height, Width, Channels
        for block in self.TB_layers:
            x_tb_path = block(x_tb_path)
        print(f"Input shape after Transformer Blocks: {x_tb_path.shape}") # Batch size, Height, Width, Channels
        x_tb_path = x_tb_path.permute(0, 3, 1, 2) # Change to (batch_size, channels, height, width)


        print("########### going to calculate Z ###########")
        print("X(CNN) shape:", x_conv_path.shape) # B, C, H, W
        print("X(TB) shape:", x_tb_path.shape) # B, C, H, W

        
        output = x_tb_path + torch.tanh(x_conv_path * x_tb_path)
        print("output shape after adding Z:", x.shape) # B, C, H, W
        print("output value:", output)
        return output
    

class HyneterModule_DualSwitch(nn.Module):
    def __init__(self, in_channels, hidden_dimension, Conv_layers,TB_layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        self.mg = MultiGranularitySummingBlock(in_channels=in_channels, embed_dim=hidden_dimension, stride1=1)

        self.Conv_layers = nn.ModuleList([])
        for _ in range(Conv_layers):
            self.Conv_layers.append(
                CNN(in_channels=hidden_dimension,embed_dim=hidden_dimension)
            )


        self.DualSwitching = DualSwitch_SwapOnly()

        self.TB_layers = nn.ModuleList([])
        for _ in range(TB_layers):
            self.TB_layers.append(
                TransformerBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
               )
            

    def forward(self, x):
        print("\nHyneterModule: HNB: Patch -> Conv -> TB")
        print("HyneterModule forward pass")
        
        
        print(f"\nInput shape before Multigranularity CNN: {x.shape}") # Batch size, Channels, Height, Width
        x = self.mg(x)
        print(f"Input shape after Multigranularity CNN: {x.shape}") # Batch size, Channels, Height, Width

        x_conv_path = x
        x_tb_path = x


        print("X(CNN) shape before Conv layers:", x_conv_path.shape) # Batch size, Channels, Height, Width
        for block in self.Conv_layers:
            x_conv_path = block(x_conv_path)
        print(f"X(CNN) shape after Conv layers: {x_conv_path.shape}") # Batch size, Channels, Height, Width
    
        x_tb_path = self.DualSwitching(x_tb_path)

        x_tb_path = x_tb_path.permute(0, 2, 3, 1) # Change to (batch_size, Height, Width, Channels)
        print("X(TB) shape before Transformer Blocks:", x_tb_path.shape) # Batch size, Height, Width, Channels
        for block in self.TB_layers:
            x_tb_path = block(x_tb_path)
        print(f"Input shape after Transformer Blocks: {x_tb_path.shape}") # Batch size, Height, Width, Channels
        x_tb_path = x_tb_path.permute(0, 3, 1, 2) # Change to (batch_size, channels, height, width)


        print("########### going to calculate Z ###########")
        print("X(CNN) shape:", x_conv_path.shape) # B, C, H, W
        print("X(TB) shape:", x_tb_path.shape) # B, C, H, W

        output = x_tb_path+torch.tanh(x_conv_path * x_tb_path)

        return output
    

import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from collections import OrderedDict


class Hyneter(nn.Module):
    def __init__(self, *, hidden_dim, Conv_layers, TB_layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()


        self.out_channels = hidden_dim * 8

        self.stage1 = HyneterModule(in_channels=channels, hidden_dimension=hidden_dim, Conv_layers=Conv_layers[0], TB_layers=TB_layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = HyneterModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, Conv_layers=Conv_layers[1], TB_layers=TB_layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = HyneterModule_DualSwitch(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, Conv_layers=Conv_layers[2], TB_layers=TB_layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = HyneterModule_DualSwitch(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, Conv_layers=Conv_layers[3], TB_layers=TB_layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )


    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)
        


def hyneter_base(hidden_dim=96, Conv_layers=(2, 2, 2, 2), TB_layers=(2, 2, 2, 2), heads=(3, 6, 12, 24), **kwargs):
    return Hyneter(hidden_dim=hidden_dim, Conv_layers=Conv_layers, TB_layers=TB_layers, heads=heads, **kwargs)


def hyneter_plus(hidden_dim=96, Conv_layers=(2, 2, 3, 2), TB_layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return Hyneter(hidden_dim=hidden_dim, Conv_layers=Conv_layers, TB_layers=TB_layers, heads=heads, **kwargs)


def hyneter_max(hidden_dim=128, Conv_layers=(2, 2, 6, 2), TB_layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return Hyneter(hidden_dim=hidden_dim, Conv_layers=Conv_layers, TB_layers=TB_layers, heads=heads, **kwargs)



from torchvision.datasets import ImageNet
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import random
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
import time
import os


# NUM_EPOCHS = 300
NUM_EPOCHS = int(input("Enter Epoch (Default: 300) : "))
IMAGE_SIZE = 224
# BATCH_SIZE = 1024
BATCH_SIZE = int(input("Enter Batch Size (Default: 1024) : "))
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 20
NUM_WORKERS = 8  # Number of workers for DataLoader
NUM_CLASSES = 4
DATA_DIR = 'DroneRF'  # Update with your ImageNet dataset path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Training for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} and learning rate {LEARNING_RATE}"
      f"\nImage size: {IMAGE_SIZE}, Weight decay: {WEIGHT_DECAY}, Warmup epochs: {WARMUP_EPOCHS}, "
      f"Number of workers: {NUM_WORKERS}, Number of classes: {NUM_CLASSES}")


model = Hyneter(hidden_dim=96, Conv_layers=(2, 2, 2, 2), TB_layers=(2, 2, 2, 2), heads=(3, 6, 12, 24), num_classes=NUM_CLASSES)
print("---------Hyneter Base model initialized----------")
print("Model size:", sum(p.numel() for p in model.parameters() if p.requires_grad))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")
print(f"Model is on device: {next(model.parameters()).device}")


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, momentum=0.9)

print("Optimizer and scheduler initialized with learning rate:", LEARNING_RATE, "and weight decay:", WEIGHT_DECAY)


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])


print("--- Data transformations initialized ---")
print("--- Loade Dataset ---")
dataset = ImageFolder(root=DATA_DIR, split='train', transform=transform)

total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size
NUM_CLASSES = len(dataset.classes)
print(f"Total dataset size: {total_size}, Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

# Split dataset
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)


print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
print(f"Train DataLoader created with batch size {BATCH_SIZE} and {NUM_WORKERS} workers")


CHECKPOINT_DIR = './checkpoints' # Directory to save checkpoints
os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Create the directory if it doesn't exist


print("Starting training...")

def train_model(model, train_loader, val_loader, epochs=5):
    start_time = time.time()
    for epoch in range(epochs):
        # --- Train ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds")

def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Top-1 prediction
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    top1_acc = 100 * correct / total
    print(f"Top-1 Accuracy on Test Set: {top1_acc:.2f}%")


# ========================
# 5. RUN TRAINING
# ========================
train_model(model, train_loader, val_loader, epochs=NUM_EPOCHS)
print("\n\n\n")
test_model(model, test_loader)
