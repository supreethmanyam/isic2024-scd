import numpy as np
import torch
from dataset import all_labels, malignant_idx
from utils import compute_auc, compute_pauc, logger


def train_epoch(
    epoch,
    model,
    optimizer,
    criterion,
    dev_dataloader,
    lr_scheduler,
    accelerator,
    target_mode,
    log_interval=100,
):
    model.train()
    train_loss = []
    total_steps = len(dev_dataloader)
    for step, (images, targets) in enumerate(dev_dataloader):
        optimizer.zero_grad()
        if target_mode == "binary":
            logits = model(images)
            probs = torch.sigmoid(logits)
            targets = targets.float().unsqueeze(1)
            loss = criterion(probs, targets)
        elif target_mode == "multi":
            logits = model(images)
            targets = targets.long()
            loss = criterion(logits, targets)
        else:
            raise ValueError(f"Invalid target_mode: {target_mode}")
        accelerator.backward(loss)
        if "medvit" in model.model_name.lower():
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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
    target_mode,
    log_interval=10,
):
    model.eval()
    val_probs = []
    val_targets = []
    val_loss = []
    total_steps = len(val_dataloader)
    with torch.no_grad():
        for step, (images, targets) in enumerate(val_dataloader):
            if target_mode == "binary":
                logits = 0
                probs = 0
                for i in range(n_tta):
                    logits_iter = model(get_trans(images, i))
                    logits += logits_iter
                    probs += torch.sigmoid(logits_iter)
                logits /= n_tta
                probs /= n_tta

                targets = targets.float().unsqueeze(1)
                loss = criterion(probs, targets)
            elif target_mode == "multi":
                out_dim = len(all_labels)
                logits = torch.zeros((images.shape[0], out_dim)).to(accelerator.device)
                probs = torch.zeros((images.shape[0], out_dim)).to(accelerator.device)
                for idx in range(n_tta):
                    logits_iter = model(get_trans(images, idx))
                    logits += logits_iter
                    probs += logits_iter.softmax(1)
                logits /= n_tta
                probs /= n_tta

                targets = targets.long()
                loss = criterion(logits, targets)
            val_loss.append(loss.detach().cpu().numpy())

            probs, targets = accelerator.gather((probs, targets))
            val_probs.append(probs)
            val_targets.append(targets)

            if (step == 0) or ((step + 1) % log_interval == 0):
                logger.info(f"Epoch: {epoch} | Step: {step + 1}/{total_steps}")

    val_loss = np.mean(val_loss)
    val_probs = torch.cat(val_probs).cpu().numpy()
    val_targets = torch.cat(val_targets).cpu().numpy()
    if target_mode == "multi":
        val_probs = val_probs[:, malignant_idx].sum(1)
        val_targets = (
            (val_targets == malignant_idx[0])
            | (val_targets == malignant_idx[1])
            | (val_targets == malignant_idx[2])
        )
    val_auc = compute_auc(val_targets, val_probs)
    val_pauc = compute_pauc(val_targets, val_probs, min_tpr=0.8)
    return (
        val_loss,
        val_auc,
        val_pauc,
        val_probs,
        val_targets,
    )


def predict(model, test_dataloader, accelerator, n_tta, target_mode, log_interval=10):
    model.eval()
    test_probs = []
    total_steps = len(test_dataloader)
    with torch.no_grad():
        for step, images in enumerate(test_dataloader):
            if target_mode == "binary":
                logits = 0
                probs = 0
                for i in range(n_tta):
                    logits_iter = model(get_trans(images, i))
                    logits += logits_iter
                    probs += torch.sigmoid(logits_iter)
                logits /= n_tta
                probs /= n_tta
            elif target_mode == "multi":
                out_dim = len(all_labels)
                logits = torch.zeros((images.shape[0], out_dim)).to(accelerator.device)
                probs = torch.zeros((images.shape[0], out_dim)).to(accelerator.device)
                for idx in range(n_tta):
                    logits_iter = model(get_trans(images, idx))
                    logits += logits_iter
                    probs += logits_iter.softmax(1)
                logits /= n_tta
                probs /= n_tta

            probs = accelerator.gather(probs)
            test_probs.append(probs)

            if (step == 0) or ((step + 1) % log_interval == 0):
                print(f"Step: {step + 1}/{total_steps}")

    test_probs = torch.cat(test_probs).cpu().numpy()
    if target_mode == "multi":
        test_probs = test_probs[:, malignant_idx].sum(1)
    return test_probs
