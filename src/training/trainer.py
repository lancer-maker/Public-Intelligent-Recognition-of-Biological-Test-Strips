import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.load_config import get_main_config
from src.utils.logger import get_logger
from .train_utils import get_weighted_bce_loss, get_optimizer

config = get_main_config().get("training", {})
logger = get_logger()
stage1_config = config.get("stage1", {})
stage2_config = config.get("stage2", {})


class StageTrainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: str,
        pos_weight: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.criterion = get_weighted_bce_loss(pos_weight, device)

    def fit(
        self, 
        fold_info: str = "",
        phase1_epochs: Optional[int] = None,
        phase2_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        phase1_lr: Optional[float] = None,
        phase2_lr: Optional[float] = None,
        weight_decay: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.loss_history = []

        phase1_epochs = phase1_epochs if phase1_epochs is not None else int(stage1_config.get("epochs", 20))
        phase2_epochs = phase2_epochs if phase2_epochs is not None else int(stage2_config.get("epochs", 10))
        patience = patience if patience is not None else config.get("early_stopping_patience")
        if patience is not None:
            patience = int(patience)
        phase1_lr = phase1_lr if phase1_lr is not None else float(stage1_config.get("learning_rate", 1e-4))
        phase2_lr = phase2_lr if phase2_lr is not None else float(stage2_config.get("learning_rate", 1e-5))
        weight_decay = weight_decay if weight_decay is not None else float(config.get("weight_decay", 1e-3))

        logger.info("Stage 1: Freezing 2D Stream (Training 1D CNN & Head)")
        self.model.freeze_image_stream()
        optimizer1 = get_optimizer(self.model, lr=phase1_lr, weight_decay=weight_decay)
        self._run_epochs(
            optimizer1,
            epochs=phase1_epochs,
            patience=patience,
            stage_name="Stage1",
            fold_info=fold_info,
        )

        logger.info("Stage 2: Unfreezing All Streams (Fine-tuning)")
        self.model.unfreeze_image_stream()
        optimizer2 = get_optimizer(self.model, lr=phase2_lr, weight_decay=weight_decay)
        best_preds, best_labels = self._run_epochs(
            optimizer2,
            epochs=phase2_epochs,
            patience=patience,
            stage_name="Stage2",
            fold_info=fold_info,
        )

        self._save_loss_history(fold_info=fold_info, report_dir="outputs/reports")
        return best_preds, best_labels

    def _save_loss_history(self, fold_info: str = "", report_dir: Optional[str] = None):

        if report_dir is None:
            report_dir = Path(__file__).resolve().parents[2] / "outputs" / "reports"
        else:
            report_dir = Path(report_dir)

        report_dir.mkdir(parents=True, exist_ok=True)

        history_df = pd.DataFrame(self.loss_history)
        if history_df.empty:
            logger.warning("没有损失历史记录，跳过 CSV 导出。")
            return

        file_name = f"loss_history_{fold_info.replace('/', '_')}.csv" if fold_info else "loss_history.csv"
        output_path = report_dir / file_name
        history_df.to_csv(output_path, index=False)
        logger.info("Saved loss history CSV: %s", output_path)

    def _run_epochs(
        self, 
        optimizer: torch.optim.Optimizer, 
        epochs: int, 
        patience: int, 
        stage_name: str,
        fold_info: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:

        best_val_loss = float('inf')
        best_preds, best_labels = None, None
        patience_counter = 0

        prefix = f"[{fold_info}] " if fold_info else ""
        desc_title = f"{prefix}{stage_name}"

        epoch_pbar = tqdm(
            range(1, epochs + 1), 
            desc=desc_title, 
            leave=True, 
            dynamic_ncols=True
        )

        for epoch in epoch_pbar:
            self.model.train()
            train_loss = 0.0
            batch_pbar = tqdm(
                self.train_loader, 
                desc=f"  └─ Epoch {epoch}/{epochs}", 
                leave=False, 
                dynamic_ncols=True
            )

            for imgs, projs, labels in batch_pbar:
                imgs = imgs.to(self.device)
                projs = projs.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                logits = self.model(imgs, projs)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                batch_pbar.set_postfix({"b_loss": f"{loss.item():.4f}"})
            train_loss /= len(self.train_loader)

            val_loss, preds, labels = self._validate()

            patience_status = "disabled" if patience is None else f"{patience_counter}/{patience}"
            epoch_pbar.set_postfix({
                "t_loss": f"{train_loss:.4f}",
                "v_loss": f"{val_loss:.4f}",
                "best_v_loss": f"{best_val_loss:.4f}" if best_val_loss != float('inf') else "N/A",
                "patience": patience_status
            })

            self.loss_history.append({
                "fold_info": fold_info,
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_preds, best_labels = preds, labels
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pth"))
            else:
                patience_counter += 1
                if patience is not None and patience_counter >= patience:
                    tqdm.write(f"  ⚠️ {prefix}{stage_name} 触发早停 (Epoch {epoch}/{epochs})")
                    logger.info("%s%s 触发早停 (Epoch %s/%s)", prefix, stage_name, epoch, epochs)
                    break

        best_ckpt = os.path.join(self.save_dir, "best_model.pth")
        if os.path.exists(best_ckpt):
            self.model.load_state_dict(torch.load(best_ckpt, weights_only=True))
            
        return best_preds, best_labels

    def _validate(self) -> Tuple[float, np.ndarray, np.ndarray]:

        self.model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, projs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                projs = projs.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                logits = self.model(imgs, projs)
                loss = self.criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(self.val_loader)
        return val_loss, np.array(all_preds).flatten(), np.array(all_labels).flatten()
