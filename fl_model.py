import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from opacus.utils.module_modification import convert_batchnorm_modules

class MedSAM_FL(nn.Module):
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.sam = convert_batchnorm_modules(self.sam)  # Replace BatchNorm with GroupNorm
        # Disable inplace=True in ReLU layers
        for module in self.sam.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.sam.to(device)
        self.device = device

    def forward(self, x):
        # x: (B, 3, 1024, 1024)
        image_embedding = self.sam.image_encoder(x)  # (B, 256, 64, 64)
        return image_embedding

    def get_prompt_embeddings(self, boxes):
        # boxes: (B, 4) normalized
        box_torch = torch.as_tensor(boxes, dtype=torch.float, device=self.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=box_torch, masks=None
        )
        return sparse_embeddings, dense_embeddings

    def decode(self, image_embeddings, sparse_embeddings, dense_embeddings):
        low_res_logits, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_logits

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_step(model, optimizer, images, masks, boxes, device):
    model.train()
    images = images.to(device)
    masks = masks.to(device)
    boxes = boxes.to(device)
    optimizer.zero_grad()
    image_embeddings = model(images)
    sparse_emb, dense_emb = model.get_prompt_embeddings(boxes)
    logits = model.decode(image_embeddings, sparse_emb, dense_emb)
    pred = F.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)
    loss = dice_loss(pred, masks)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    num_samples = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            # Assume boxes are fixed for simplicity, e.g., full image
            boxes = torch.tensor([[0, 0, 1, 1]] * images.size(0), device=device)
            image_embeddings = model(images)
            sparse_emb, dense_emb = model.get_prompt_embeddings(boxes)
            logits = model.decode(image_embeddings, sparse_emb, dense_emb)
            pred = F.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)
            loss = dice_loss(pred, masks)
            pred_mask = (torch.sigmoid(pred) > 0.5).float()
            dice = 1 - dice_loss(pred_mask, masks)
            total_loss += loss.item() * images.size(0)
            total_dice += dice.item() * images.size(0)
            num_samples += images.size(0)
    return total_loss / num_samples, total_dice / num_samples
