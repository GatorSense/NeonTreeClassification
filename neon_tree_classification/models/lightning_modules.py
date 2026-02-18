"""
PyTorch Lightning modules for training tree classification models.

Provides base class with common training logic and modality-specific extensions.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from typing import Dict, Any, Optional, Union
import torchmetrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from pytorch_lightning.loggers import CometLogger

from .rgb_models import create_rgb_model
from .hsi_models import create_hsi_model
from .lidar_models import create_lidar_model
from torchvision import transforms


class BaseTreeClassifier(L.LightningModule):
    """
    Base Lightning module for tree species classification.

    Provides common training logic that works with our clean data format:
    batch = {"modality": tensor, "species_idx": tensor}
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "plateau",
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize base classifier.

        Args:
            model: PyTorch model (from rgb_models.py, hsi_models.py, etc.)
            num_classes: Number of tree species classes
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            scheduler: Scheduler type ('plateau', 'cosine', 'step')
            weight_decay: Weight decay for optimizer
            class_weights: Optional class weights for imbalanced datasets
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "class_weights"])

        self.model = model
        self.num_classes = num_classes
        self.class_weights = class_weights

        # Common metrics for all modalities
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        # Confusion matrix for validation
        self.val_confmat = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

        # Storage for test predictions and labels (for confusion matrix)
        self.test_predictions = []
        self.test_labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str):
        """Shared step logic for train/val/test."""
        # Extract labels
        targets = batch["species_idx"]

        # Get modality data (to be implemented by subclasses)
        inputs = self._extract_modality_data(batch)

        # Forward pass
        logits = self.forward(inputs)

        # Compute loss
        if self.class_weights is not None:
            loss = F.cross_entropy(
                logits, targets, weight=self.class_weights.to(self.device)
            )
        else:
            loss = F.cross_entropy(logits, targets)

        # Get predictions
        preds = torch.argmax(logits, dim=1)

        return loss, preds, targets, logits

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        loss, preds, targets, logits = self._shared_step(batch, "train")

        # Update metrics
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_acc, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        loss, preds, targets, logits = self._shared_step(batch, "val")

        # Update metrics
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_confmat(preds, targets)

        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True)

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss, preds, targets, logits = self._shared_step(batch, "test")

        # Store predictions and targets for confusion matrix
        self.test_predictions.append(preds.detach().cpu())
        self.test_labels.append(targets.detach().cpu())

        # Update metrics
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)

        # Log metrics
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_accuracy", self.test_acc, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)

        return loss

    def _get_species_names_and_labels(self):
        """Extract species names and create display labels for confusion matrix."""
        try:
            # Get the idx_to_label mapping from the dataset
            idx_to_label = self.trainer.datamodule.full_dataset.idx_to_label
            num_classes = self.trainer.datamodule.full_dataset.num_classes

            # Create ordered list of species names (idx 0, 1, 2, ...)
            species_names = [idx_to_label[i] for i in range(num_classes)]

            # Create display labels: "Species_Name (0)", "Species_Name (1)", etc.
            display_labels = [
                f"{species} ({i})" for i, species in enumerate(species_names)
            ]

            # Get the actual unique labels present in the test data for classification report
            predictions = torch.cat(self.test_predictions).cpu().numpy()
            true_labels = torch.cat(self.test_labels).cpu().numpy()
            label_ints_for_report = sorted(list(set(true_labels)))

            return species_names, display_labels, label_ints_for_report

        except (AttributeError, KeyError) as e:
            print(f"Could not extract species names: {e}")
            # Fallback to generic labels
            num_classes = self.hparams.num_classes
            species_names = [f"Species_{i}" for i in range(num_classes)]
            display_labels = [f"Species_{i} ({i})" for i in range(num_classes)]

            predictions = torch.cat(self.test_predictions).cpu().numpy()
            true_labels = torch.cat(self.test_labels).cpu().numpy()
            label_ints_for_report = sorted(list(set(true_labels)))

            return species_names, display_labels, label_ints_for_report

    def on_test_epoch_end(self):
        """Generate and log confusion matrix and classification report after test epoch."""

        # Log standard metrics
        self.log_dict(
            {
                "test_acc_final": self.test_acc.compute(),
                "test_f1_final": self.test_f1.compute(),
            },
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        if not self.test_predictions or not self.test_labels:
            print(
                "No test predictions or labels recorded, skipping confusion matrix and report."
            )
            return

        # Get results directory
        results_dir = self.trainer.default_root_dir

        # Convert predictions and labels to numpy
        predictions = torch.cat(self.test_predictions).cpu().numpy()
        true_labels = torch.cat(self.test_labels).cpu().numpy()

        # Debug: Print total test samples
        print(f"\n📊 Test Set Statistics:")
        print(f"   Total test samples: {len(true_labels)}")
        print(f"   Predictions collected from {len(self.test_predictions)} batches")

        # Get species names and labels
        species_names, display_labels, label_ints_for_report = (
            self._get_species_names_and_labels()
        )

        print(f"\nGenerating Test Results Summary in: {results_dir}")
        print(f"   Number of classes in test set: {len(label_ints_for_report)}")

        # Generate and save confusion matrix
        try:
            cm_display = ConfusionMatrixDisplay.from_predictions(
                true_labels,
                predictions,
                labels=label_ints_for_report,
                display_labels=[display_labels[i] for i in label_ints_for_report],
                xticks_rotation="vertical",
                cmap="Blues",
            )
            cm_fig_path = os.path.join(results_dir, "test_confusion_matrix.png")
            cm_display.figure_.savefig(cm_fig_path, dpi=150, bbox_inches="tight")
            print(f"Confusion Matrix figure saved to: {cm_fig_path}")

            # Get raw confusion matrix for logging
            conf_matrix = confusion_matrix(
                true_labels, predictions, labels=label_ints_for_report
            )
            print("\nConfusion Matrix:\n", conf_matrix)

            # Log confusion matrix to CometML
            try:
                self.logger.experiment.log_confusion_matrix(
                    matrix=conf_matrix,
                    labels=[display_labels[i] for i in label_ints_for_report],
                )
                print("Confusion Matrix logged to CometML.")
            except Exception as e:
                print(f"Could not log confusion matrix to CometML: {e}")

            # Close figure to free memory
            plt.close(cm_display.figure_)

        except Exception as e:
            print(f"Could not generate confusion matrix: {e}")
            print(
                "Confusion Matrix (sklearn):\n",
                confusion_matrix(
                    true_labels, predictions, labels=label_ints_for_report
                ),
            )

        # Generate and save classification report
        try:
            report = classification_report(
                true_labels,
                predictions,
                labels=label_ints_for_report,
                target_names=[display_labels[i] for i in label_ints_for_report],
                zero_division=0,
                output_dict=True,
            )
            report_df = pd.DataFrame(report).transpose().round(4)
            report_csv_path = os.path.join(
                results_dir, "test_classification_report.csv"
            )
            report_df.to_csv(report_csv_path)
            print(f"Classification Report saved to: {report_csv_path}")
            print("\nClassification Report:\n", report_df)

            # Log classification report to CometML
            try:
                self.logger.experiment.log_table(
                    filename="test_classification_report.csv", dataframe=report_df
                )
                print("Classification Report logged to CometML.")
            except Exception as e:
                print(f"Could not log classification report to CometML: {e}")

        except Exception as e:
            print(f"Could not generate classification report: {e}")

        # Clear predictions and labels for next test run
        self.test_predictions.clear()
        self.test_labels.clear()

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Optimizer
        if self.hparams.optimizer == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        # Scheduler
        if self.hparams.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        elif self.hparams.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=100)
            return [optimizer], [scheduler]
        elif self.hparams.scheduler == "step":
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def _extract_modality_data(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract modality-specific data from batch.

        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _extract_modality_data")


class RGBClassifier(BaseTreeClassifier):
    """
    Lightning module for RGB-based tree species classification.

    Includes RGB-specific evaluation features like image logging.
    """

    def __init__(
        self,
        model_type: str = "simple",
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "plateau",
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        log_images: bool = False,
        idx_to_label: Optional[Dict[int, str]] = None,
        rgb_norm_method: str = "imagenet",
        **model_kwargs,
    ):
        """
        Initialize RGB classifier.

        Args:
            model_type: Type of RGB model ("simple", "resnet")
            num_classes: Number of tree species classes
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            scheduler: Scheduler type ('plateau', 'cosine', 'step')
            weight_decay: Weight decay for optimizer
            class_weights: Optional class weights for imbalanced datasets
            log_images: Whether to log sample images during validation
            idx_to_label: Optional label mapping {0: "Species1", 1: "Species2", ...}
                         for DeepForest CropModel compatibility
            rgb_norm_method: Normalization method used during training ('imagenet' or '0_1')
            **model_kwargs: Additional arguments for model creation
        """
        # Create RGB model
        model = create_rgb_model(
            model_type=model_type, num_classes=num_classes, **model_kwargs
        )

        super().__init__(
            model=model,
            num_classes=num_classes,
            learning_rate=learning_rate,
            optimizer=optimizer,
            scheduler=scheduler,
            weight_decay=weight_decay,
            class_weights=class_weights,
        )

        self.log_images = log_images
        self.logged_images_this_epoch = False
        self.rgb_norm_method = rgb_norm_method

        # Set label_dict for DeepForest CropModel compatibility
        if idx_to_label is not None:
            self.set_label_dict(idx_to_label)
        else:
            self.label_dict = None
            self.numeric_to_label_dict = None

    def _extract_modality_data(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract RGB data from batch."""
        return batch["rgb"]

    def normalize(self):
        """Return normalization transform matching the training configuration.

        Required for DeepForest CropModel integration. Returns a transform
        consistent with the rgb_norm_method used during training.

        Returns:
            torchvision.transforms.Normalize object
        """
        if self.rgb_norm_method == "imagenet":
            return transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        elif self.rgb_norm_method == "0_1":
            # Scale to [0,1]: equivalent to dividing by 255 in ToTensor,
            # represented as zero-mean, unit-std (no-op standardization)
            return transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        else:
            raise ValueError(f"Unknown rgb_norm_method: {self.rgb_norm_method}")

    def set_label_dict(self, idx_to_label: Dict[int, str]):
        """Set label dictionaries from idx_to_label mapping.

        Creates both label_dict and numeric_to_label_dict as required by DeepForest CropModel.

        Args:
            idx_to_label: Dictionary mapping class indices to class names
        """
        # label_dict: {"Class1": 0, "Class2": 1} - used by DeepForest for class lookup
        self.label_dict = {label: idx for idx, label in idx_to_label.items()}
        # numeric_to_label_dict: {0: "Class1", 1: "Class2"} - used by DeepForest for prediction output
        self.numeric_to_label_dict = dict(idx_to_label)

    def get_label_dict(self) -> Optional[Dict[str, int]]:
        """Get label dictionary in DeepForest CropModel format.

        Returns:
            Dictionary mapping class names to indices, or None if not set
        """
        return self.label_dict

    def on_save_checkpoint(self, checkpoint):
        """Save label dictionaries to checkpoint for DeepForest CropModel compatibility."""
        checkpoint["label_dict"] = self.label_dict
        checkpoint["numeric_to_label_dict"] = self.numeric_to_label_dict

    def on_load_checkpoint(self, checkpoint):
        """Restore label dictionaries from checkpoint."""
        self.label_dict = checkpoint.get("label_dict", None)
        self.numeric_to_label_dict = checkpoint.get("numeric_to_label_dict", None)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step with optional image logging."""
        loss = super().validation_step(batch, batch_idx)

        # Log sample images periodically
        if (
            self.log_images
            and not self.logged_images_this_epoch
            and batch_idx == 0
            and self.trainer.current_epoch % 10 == 0
        ):
            self._log_sample_images(batch)
            self.logged_images_this_epoch = True

        return loss

    def on_validation_epoch_start(self):
        """Reset image logging flag at start of validation epoch."""
        self.logged_images_this_epoch = False

    def _log_sample_images(self, batch: Dict[str, torch.Tensor]):
        """Log sample RGB images with predictions."""
        if not hasattr(self.logger, "experiment"):
            return

        # Get a few samples
        rgb_images = batch["rgb"][:4]  # First 4 images
        targets = batch["species_idx"][:4]

        # Get predictions
        with torch.no_grad():
            logits = self.forward(rgb_images)
            preds = torch.argmax(logits, dim=1)

        # Normalize images for display (assuming they're in [0, 1] range)
        images_np = rgb_images.cpu().numpy()

        # Log to comet/tensorboard (implementation depends on logger)
        try:
            from lightning.pytorch.loggers import CometLogger

            if isinstance(self.logger, CometLogger):
                for i in range(len(images_np)):
                    img = images_np[i].transpose(1, 2, 0)  # CHW -> HWC
                    caption = f"True: {targets[i].item()}, Pred: {preds[i].item()}"
                    self.logger.experiment.log_image(
                        img,
                        name=f"val_image_{i}_epoch_{self.trainer.current_epoch}",
                        step=self.trainer.current_epoch,
                    )
        except ImportError:
            pass  # comet not available


class HSIClassifier(BaseTreeClassifier):
    """
    Lightning module for hyperspectral-based tree species classification.

    Includes HSI-specific evaluation features like spectral analysis.
    Supports multi-output models (e.g., Hang2020) with deep supervision.
    """

    def __init__(
        self,
        model_type: str = "simple",
        num_bands: int = 426,
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "plateau",
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        aux_loss_weight: float = 0.4,
        **model_kwargs,
    ):
        """
        Initialize HSI classifier.

        Args:
            model_type: Type of HSI model ("simple", "spectral_cnn", "hypernet", "hang2020")
            num_bands: Number of hyperspectral bands
            num_classes: Number of tree species classes
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            scheduler: Scheduler type ('plateau', 'cosine', 'step')
            weight_decay: Weight decay for optimizer
            class_weights: Optional class weights for imbalanced datasets
            aux_loss_weight: Weight for auxiliary losses in multi-output models (0.0-1.0)
            **model_kwargs: Additional arguments for model creation
        """
        # Create HSI model
        model = create_hsi_model(
            model_type=model_type,
            num_bands=num_bands,
            num_classes=num_classes,
            **model_kwargs,
        )

        super().__init__(
            model=model,
            num_classes=num_classes,
            learning_rate=learning_rate,
            optimizer=optimizer,
            scheduler=scheduler,
            weight_decay=weight_decay,
            class_weights=class_weights,
        )

        self.num_bands = num_bands
        self.aux_loss_weight = aux_loss_weight

        # Detect if model supports multi-output training (e.g., Hang2020)
        self.is_multi_output = hasattr(self.model, "forward_with_aux")

        if self.is_multi_output:
            print(
                f"✓ Multi-output model detected - using deep supervision with aux_weight={aux_loss_weight}"
            )

    def _extract_modality_data(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract HSI data from batch."""
        return batch["hsi"]

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str):
        """
        Shared step with support for multi-output models.

        Overrides base class to handle models with auxiliary outputs (e.g., Hang2020).
        """
        # Extract labels
        targets = batch["species_idx"]
        inputs = self._extract_modality_data(batch)

        # Check if we're training/validating and using multi-output model
        if self.is_multi_output and stage in ["train", "val"]:
            # Multi-output forward pass (Hang2020 style)
            final_logits, aux_logits = self.model.forward_with_aux(inputs)

            # Main loss on final output
            if self.class_weights is not None:
                main_loss = F.cross_entropy(
                    final_logits, targets, weight=self.class_weights.to(self.device)
                )
            else:
                main_loss = F.cross_entropy(final_logits, targets)

            # Auxiliary losses (deep supervision on intermediate outputs)
            aux_losses = []
            for aux_logit in aux_logits:
                if self.class_weights is not None:
                    aux_loss = F.cross_entropy(
                        aux_logit, targets, weight=self.class_weights.to(self.device)
                    )
                else:
                    aux_loss = F.cross_entropy(aux_logit, targets)
                aux_losses.append(aux_loss)

            # Combined loss: main + weighted average of auxiliary losses
            total_aux_loss = torch.stack(aux_losses).mean()
            loss = main_loss + self.aux_loss_weight * total_aux_loss

            # Log individual losses
            if stage == "train":
                self.log("train_main_loss", main_loss, on_epoch=True)
                self.log("train_aux_loss", total_aux_loss, on_epoch=True)
            elif stage == "val":
                self.log("val_main_loss", main_loss, on_epoch=True)
                self.log("val_aux_loss", total_aux_loss, on_epoch=True)

            # Use final logits for predictions
            logits = final_logits
        else:
            # Single-output forward pass (standard models or test stage)
            logits = self.forward(inputs)

            # Compute loss
            if self.class_weights is not None:
                loss = F.cross_entropy(
                    logits, targets, weight=self.class_weights.to(self.device)
                )
            else:
                loss = F.cross_entropy(logits, targets)

        # Get predictions
        preds = torch.argmax(logits, dim=1)

        return loss, preds, targets, logits

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step with HSI-specific analysis."""
        loss = super().test_step(batch, batch_idx)

        # Add HSI-specific metrics if needed
        # TODO: Add spectral band importance analysis
        # TODO: Add spectral confusion analysis

        return loss


class LiDARClassifier(BaseTreeClassifier):
    """
    Lightning module for LiDAR-based tree species classification.

    Includes LiDAR-specific evaluation features like height analysis.
    """

    def __init__(
        self,
        model_type: str = "simple",
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "plateau",
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        **model_kwargs,
    ):
        """
        Initialize LiDAR classifier.

        Args:
            model_type: Type of LiDAR model ("simple", "height_cnn", "structural")
            num_classes: Number of tree species classes
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            scheduler: Scheduler type ('plateau', 'cosine', 'step')
            weight_decay: Weight decay for optimizer
            class_weights: Optional class weights for imbalanced datasets
            **model_kwargs: Additional arguments for model creation
        """
        # Create LiDAR model
        model = create_lidar_model(
            model_type=model_type, num_classes=num_classes, **model_kwargs
        )

        super().__init__(
            model=model,
            num_classes=num_classes,
            learning_rate=learning_rate,
            optimizer=optimizer,
            scheduler=scheduler,
            weight_decay=weight_decay,
            class_weights=class_weights,
        )

    def _extract_modality_data(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract LiDAR data from batch."""
        return batch["lidar"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step with LiDAR-specific analysis."""
        loss = super().test_step(batch, batch_idx)

        # Add LiDAR-specific metrics if needed
        # TODO: Add height distribution analysis
        # TODO: Add structural pattern analysis

        return loss
