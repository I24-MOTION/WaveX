import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from PIL import Image
import numpy as np
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from i24_dataloader import WavexDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy
ddp = DDPStrategy(process_group_backend="gloo")


print(torch.cuda.device_count())
torch.set_float32_matmul_precision('medium')
        
class UNet(nn.Module):
    def __resolution_in_list(self, resolution, resolution_list):
        for entry in resolution_list:
            if ((entry[0] == resolution[0]) and (entry[1] == resolution[1])):
                return True
        return False

    def __init__(self, unet_input_channels, unet_output_channels, unet_original_resolution, unet_block_channels, unet_block_resolutions, encoder_attn_resolutions, decoder_attn_resolutions, bottleneck_attn, num_subblocks, time_steps=500):
        super().__init__()
        self.unet_input_channels = unet_input_channels
        self.unet_output_channels = unet_output_channels
        self.unet_original_resolution = unet_original_resolution
        self.unet_block_channels = unet_block_channels
        self.unet_block_resolutions = unet_block_resolutions
        self.encoder_attn_resolutions = encoder_attn_resolutions
        self.decoder_attn_resolutions = decoder_attn_resolutions
        self.bottleneck_attn = bottleneck_attn
        self.num_subblocks = num_subblocks
        self.time_steps = time_steps
        self.encoder_blocks = nn.ModuleList([])
        # We are assuming that the last two channels are gonna be used for the bottleneck
        for i in range(len(unet_block_channels)):
            prev_channels = unet_input_channels if (i == 0) else unet_block_channels[i - 1]
            next_channels = unet_block_channels[i]
            output_resolution = unet_block_resolutions[i]
            block = nn.ModuleList([])
            block.append(encoder_block(prev_channels, next_channels, output_resolution=output_resolution, time_steps=time_steps, num_subblocks=num_subblocks))
            if self.__resolution_in_list(output_resolution, self.encoder_attn_resolutions):
                block.append(AttnBlock(next_channels))
            self.encoder_blocks.append(block)
        self.bottleneck_block_conv = conv_block(unet_block_channels[-1], unet_block_channels[-1], time_steps=self.time_steps)
        self.bottleneck_block_attn = AttnBlock(unet_block_channels[-1]) if self.bottleneck_attn else nn.Identity()
        self.decoder_blocks = nn.ModuleList([])
        for i in range(len(unet_block_channels) - 1, -1, -1):
            prev_channels = unet_block_channels[i]
            next_channels = unet_block_channels[i - 1] if (i > 0) else unet_input_channels # Keep it symmetrical at the end until we do a simple conv2d mapping to the output channel .....
            output_resolution = unet_block_resolutions[i - 1] if (i > 0) else unet_original_resolution
            #print(prev_channels, next_channels, output_resolution)
            block = nn.ModuleList([])
            block.append(decoder_block(prev_channels, next_channels, output_resolution=output_resolution, time_steps=time_steps, num_subblocks=num_subblocks))
            if self.__resolution_in_list(output_resolution, self.decoder_attn_resolutions):
                block.append(AttnBlock(next_channels))
            self.decoder_blocks.append(block)
        self.output_block = nn.Conv2d(unet_input_channels, unet_output_channels, kernel_size=1, padding=0)

    def forward(self, inputs, t=None):
        s_list = []
        p_list = []
        for entry in self.encoder_blocks:
            prev_p = p_list[-1] if len(p_list) > 0 else inputs
            s, p = entry[0](prev_p, t)
            s_list.append(s)
            # Attention mechanism
            if (len(entry) >= 2):
                p = entry[1](p)
            p_list.append(p)
        b = self.bottleneck_block_conv(p_list[-1], t)
        b = self.bottleneck_block_attn(b)
        outputs = b
        for entry in self.decoder_blocks:
            s = s_list.pop()
            #print(outputs.shape, s.shape, t.shape)
            outputs = entry[0](outputs, s, t)
            # Attention mechanism
            if (len(entry) >= 2):
                outputs = entry[1](outputs)
        outputs = self.output_block(outputs)
        return outputs

class AttnBlock(nn.Module):
    def __init__(self, embedding_dims, num_heads=4) -> None:
        super().__init__()

        self.embedding_dims = embedding_dims
        self.ln = nn.LayerNorm(embedding_dims)
        self.mhsa = MultiHeadSelfAttention(embedding_dims=embedding_dims, num_heads=num_heads)
        self.ff = nn.Sequential(
            nn.LayerNorm(self.embedding_dims),
            nn.Linear(self.embedding_dims, self.embedding_dims),
            nn.GELU(),
            nn.Linear(self.embedding_dims, self.embedding_dims),
        )

    def forward(self, x):
        bs, c, sz, _ = x.shape
        x = x.view(bs, c, -1).swapaxes(1, 2)  # Make tensors contiguous
        x = x.contiguous()  # Ensure the tensor is contiguous
        x_ln = self.ln(x)
        _, attention_value = self.mhsa(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(bs, c, sz, sz).contiguous()  # Ensure the tensor is contiguous


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dims, num_heads=4) -> None:
        super().__init__()
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        assert self.embedding_dims % self.num_heads == 0, f"{self.embedding_dims} not divisible by {self.num_heads}"
        self.head_dim = self.embedding_dims // self.num_heads
        self.wq = nn.Linear(self.head_dim, self.head_dim)
        self.wk = nn.Linear(self.head_dim, self.head_dim)
        self.wv = nn.Linear(self.head_dim, self.head_dim)
        self.wo = nn.Linear(self.embedding_dims, self.embedding_dims)

    def attention(self, q, k, v):
        # no need for a mask
        attn_weights = F.softmax((q @ k.transpose(-1, -2)) / self.head_dim ** 0.5, dim=-1)
        return attn_weights, attn_weights @ v

    def forward(self, q, k, v):
        bs, img_sz, c = q.shape
        q = q.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v of the shape (bs, self.num_heads, img_sz**2, self.head_dim)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        attn_weights, o = self.attention(q, k, v)  # of shape (bs, num_heads, img_sz**2, c)

        o = o.transpose(1, 2).contiguous().view(bs, img_sz, self.embedding_dims)
        o = self.wo(o)
        return attn_weights, o
        
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, output_resolution, time_steps, num_subblocks, activation = "relu"):
        super().__init__()
        self.conv = nn.ModuleList([conv_block(in_c, out_c, time_steps = time_steps, activation = activation, embedding_dims = out_c)])
        for i in range(num_subblocks - 1):
            self.conv.append(conv_block(out_c, out_c, time_steps = time_steps, activation = activation, embedding_dims = out_c))
        self.skip = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0) if (in_c != out_c) else nn.Identity()
        self.output_resolution = output_resolution

    def forward(self, inputs, time = None):
        x = inputs
        for entry in self.conv:
            x = entry(x, time)
        x = x + self.skip(inputs)
        p = F.interpolate(x, self.output_resolution)
        return x, p

# Decoder Block for upsampling
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, output_resolution, time_steps, num_subblocks, activation = "relu"):
        super().__init__()
        self.output_resolution = output_resolution
        self.up = nn.Upsample(output_resolution)
        self.conv = nn.ModuleList([conv_block(2 * in_c, out_c, time_steps = time_steps, activation = activation, embedding_dims = out_c)])
        for i in range(num_subblocks - 1):
            self.conv.append(conv_block(out_c, out_c, time_steps = time_steps, activation = activation, embedding_dims = out_c))
        self.skip = nn.Conv2d(2 * in_c, out_c, kernel_size=1, stride=1, padding=0) if ((2 * in_c) != out_c) else nn.Identity()
    def forward(self, inputs, skip, time = None):
        inputs = self.up(inputs)
        inputs = torch.cat([inputs, skip], axis=1)
        x = inputs
        for entry in self.conv:
            x = entry(x, time)
        x = x + self.skip(inputs)
        return x        

class GammaEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU()

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return self.act(self.linear(encoding))


# Double Conv Block
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps=500, activation="relu", embedding_dims=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.embedding_dims = embedding_dims if embedding_dims else out_c

        # self.embedding = nn.Embedding(time_steps, embedding_dim = self.embedding_dims)
        self.embedding = GammaEncoding(self.embedding_dims)
        # switch to nn.Embedding if you want to pass in timestep instead; but note that it should be of dtype torch.long
        self.act = nn.ReLU() if activation == "relu" else nn.SiLU()

    def forward(self, inputs, time=None):
        time_embedding = self.embedding(time).view(-1, self.embedding_dims, 1, 1)
        # print(f"time embed shape => {time_embedding.shape}")
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = x + time_embedding
        return x


class DiffusionModel(nn.Module):
    def __init__(self, time_steps,
                 beta_start=10e-4,
                 beta_end=0.02):
        super().__init__()
        self.time_steps = time_steps
        self.unet_input_channels = 4
        self.unet_output_channels = 1
        self.unet_original_resolution = (200,200)
        self.unet_block_channels = [64,128,256,512,1024]
        self.unet_block_resolutions = [
            (100,100),
            (50,50),
            (25,25),
            (12,12),
            (6,6)
        ]
        self.encoder_attn_resolutions = [(25, 25), (12, 12)]
        self.decoder_attn_resolutions = [(25, 25), (12, 12)]
        self.bottleneck_attn = True
        self.num_subblocks = 3
        self.betas = torch.linspace(beta_start, beta_end, self.time_steps)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=-1)
        self.model = UNet(self.unet_input_channels, self.unet_output_channels, self.unet_original_resolution, self.unet_block_channels, self.unet_block_resolutions, self.encoder_attn_resolutions, self.decoder_attn_resolutions, self.bottleneck_attn, self.num_subblocks, time_steps=self.time_steps)

    def add_noise(self, x, ts):
        # 'x' and 'ts' are expected to be batched
        noise = torch.randn_like(x)
        # print(x.shape, noise.shape)
        noised_examples = []
        for i, t in enumerate(ts):
            alpha_hat_t = self.alpha_hats[t]
            noised_examples.append(torch.sqrt(alpha_hat_t) * x[i] + torch.sqrt(1 - alpha_hat_t) * noise[i])
        return torch.stack(noised_examples), noise

    def forward(self, x, t):
        return self.model(x, t)
        
from torchvision import transforms

class LightningDiffusionModel(L.LightningModule):
    def __init__(self, lr=2e-4, time_steps=500):
        super().__init__()
        self.time_steps = time_steps
        self.lr = lr
        self.save_hyperparameters()
        self.diffusion_model = DiffusionModel(time_steps=time_steps)
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, x, y, ts):
        x, y = x.to(self.device), y.to(self.device)
        gamma = self.diffusion_model.alpha_hats[ts].to(self.device)
        predicted_noise = self.diffusion_model(y, gamma)
        return predicted_noise
        
    # def get_loss(self, x, y, m=0.8):
        # bs = y.shape[0]
        # ts = torch.randint(low=1, high=self.time_steps, size=(bs,))
        # mask_over = y > 0.2
        # mask_under = ~mask_over
        # y, target_noise = self.diffusion_model.add_noise(y, ts)
        # y = torch.cat([x, y], dim=1)
        # predicted_noise = self(x, y, ts)
        # loss_over = self.criterion(target_noise[mask_over], predicted_noise[mask_over])
        # loss_under = self.criterion(target_noise[mask_under], predicted_noise[mask_under])
        # loss = (1-m) * loss_over + m * loss_under
        # return loss
        
    def get_loss(self, x, y, m=0.8):
        bs = y.shape[0]
        ts = torch.randint(low=1, high=self.time_steps, size=(bs,))
        mask_over = y > 0.15
        mask_under = ~mask_over
        stacked_noise, target_noise = self.diffusion_model.add_noise(y, ts)
        stacked_noise = torch.cat([x, stacked_noise], dim=1)
        predicted_noise = self(x, stacked_noise, ts)
        loss_over = self.criterion(target_noise[mask_over], predicted_noise[mask_over])
        loss_under = self.criterion(target_noise[mask_under], predicted_noise[mask_under])
        loss = (1-m) * loss_over + m * loss_under
        return loss
        
    def get_val_loss(self, x, y):
        bs = y.shape[0]
        ts = torch.randint(low=1, high=self.time_steps, size=(bs,))
        y, target_noise = self.diffusion_model.add_noise(y, ts)
        y = torch.cat([x, y], dim=1)
        predicted_noise = self(x, y, ts)
        loss = self.criterion(target_noise, predicted_noise)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.get_loss(x, y)
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if (self.global_rank == 0) and (batch_idx == 0):
            self.current_val_image = x[0]
            self.current_val_truth = y[0]
        loss = self.get_val_loss(x, y)
        self.log("val/loss", loss, sync_dist=True)
        return loss

    def postprocess_image(self, tensor):
        print(tensor.shape)
        if tensor.shape[1] == 1:
            tensor = tensor.squeeze(0).squeeze(0).cpu().numpy()
        else:
            tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Clamp tensor values to [0, 1] range
        tensor = np.clip(tensor, 0, 1)
        
        # Replace NaN and infinite values
        tensor = np.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure values are within [0, 255] range
        tensor = (tensor * 255).astype(np.uint8)
        
        image = Image.fromarray(tensor)
        image = image.resize((128, 128), Image.BICUBIC)  # Ensure the image is resized to 128x128
        totensor = transforms.ToTensor()
        return totensor(image)

    def on_validation_epoch_end(self):
        if (self.global_rank != 0):
            return
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')

        test_image = self.current_val_image.unsqueeze(0).to(self.device)
        y = torch.randn((1,1,200,200), device=self.device)
        for t in range(self.time_steps - 1, 0, -1):
            alpha_t = self.diffusion_model.alphas[t]
            alpha_t_hat = self.diffusion_model.alpha_hats[t]
            beta_t = self.diffusion_model.betas[t]
            t = torch.tensor([t], device=self.device).long()
            pred_noise = self.diffusion_model(torch.cat([test_image, y], dim=1), alpha_t_hat.view(-1).to(self.device))
            y = (torch.sqrt(1 / alpha_t)) * (y - (1 - alpha_t) / torch.sqrt(1 - alpha_t_hat) * pred_noise)
            if t > 1:
                noise = torch.randn_like(y)
                y = y + torch.sqrt(beta_t) * noise
        test_image = self.postprocess_image(test_image[:,[0],:,:]).unsqueeze(0)
        y = self.postprocess_image(y).unsqueeze(0)
        y_truth = self.postprocess_image(self.current_val_truth.unsqueeze(0)).unsqueeze(0)
        tb_logger.add_images("val/generated", y, self.current_epoch)
        tb_logger.add_images("val/rds", test_image, self.current_epoch)
        tb_logger.add_images("val/motion", y_truth, self.current_epoch)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt

from torch.utils.data import Dataset, DataLoader, Subset

def train_ddpm(time_steps=500, epochs=20, batch_size=16):
    model = LightningDiffusionModel(time_steps=time_steps)
    train_ds = WavexDataset("wavex/train/rds.npy", "wavex/train/motion.npy")
    val_ds = WavexDataset("wavex/val/rds.npy", "wavex/val/motion.npy")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="experiment/lightning_logs", name="r2", version=None)
    
    # Define ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        filename='best',
        save_top_k=10,
        mode='min'
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        logger=tb_logger,
        accelerator="gpu",
        devices=2,
        strategy=ddp,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback]  # Add checkpoint callback here
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return model

if __name__ == '__main__':
    model = train_ddpm(time_steps=500,epochs=500, batch_size=32)