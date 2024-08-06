import torch
import numpy as np
from utils import compute_auc, compute_pauc, logger


def train_epoch(
    epoch,
    model,
    optimizer,
    criterion,
    dev_dataloader,
    lr_scheduler,
    accelerator,
    log_interval=100,
):
    model.train()
    train_loss = []
    total_steps = len(dev_dataloader)
    for step, (images, labels, index) in enumerate(dev_dataloader):
        optimizer.zero_grad()
        logits = model(images)
        probs = torch.sigmoid(logits)
        loss = criterion(probs, labels.unsqueeze(1))
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        loss_value = accelerator.gather(loss).item()
        train_loss.append(loss_value)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        if (step == 0) or ((step + 1) % log_interval == 0):
            logger.info(
                f"Epoch: {epoch} | Step: {step + 1}/{total_steps} |"
                f" Loss: {loss_value:.5f} | Smooth loss: {smooth_loss:.5f}"
            )
    train_loss = np.mean(train_loss)
    return train_loss


def get_trans(img, iteration):
    if iteration >= 6:
        img = img.transpose(2, 3)
    if iteration % 6 == 0:
        return img
    elif iteration % 6 == 1:
        return torch.flip(img, dims=[2])
    elif iteration % 6 == 2:
        return torch.flip(img, dims=[3])
    elif iteration % 6 == 3:
        return torch.rot90(img, 1, dims=[2, 3])
    elif iteration % 6 == 4:
        return torch.rot90(img, 2, dims=[2, 3])
    elif iteration % 6 == 5:
        return torch.rot90(img, 3, dims=[2, 3])


def val_epoch(
    epoch,
    model,
    criterion,
    val_dataloader,
    accelerator,
    n_tta,
    log_interval=100,
):
    model.eval()
    val_probs = []
    val_labels = []
    val_loss = []
    total_steps = len(val_dataloader)
    with torch.no_grad():
        for step, (images, labels, index) in enumerate(val_dataloader):
            logits = 0
            probs = 0
            for i in range(n_tta):
                logits_iter = model(get_trans(images, i))
                logits += logits_iter
                probs += torch.sigmoid(logits_iter)
            logits /= n_tta
            probs /= n_tta

            labels = labels.unsqueeze(1)
            loss = criterion(probs, labels)
            val_loss.append(loss.detach().cpu().numpy())

            probs, labels = accelerator.gather((probs, labels))
            val_probs.append(probs)
            val_labels.append(labels)

            if (step == 0) or ((step + 1) % log_interval == 0):
                logger.info(f"Epoch: {epoch} | Step: {step + 1}/{total_steps}")

    val_loss = np.mean(val_loss)
    val_probs = torch.cat(val_probs).cpu().numpy()
    val_labels = torch.cat(val_labels).cpu().numpy()
    val_auc = compute_auc(val_labels, val_probs)
    val_pauc = compute_pauc(val_labels, val_probs, min_tpr=0.8)
    return (
        val_loss,
        val_auc,
        val_pauc,
        val_probs,
        val_labels
    )
