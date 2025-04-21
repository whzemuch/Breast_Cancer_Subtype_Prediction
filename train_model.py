#!/usr/bin/env python
# train_model.py

import argparse
import torch
from pathlib import Path
from utils import Config

# Local imports
from data_loader import create_dataloaders
from data_loader import MultiOmicsDataset
from models import MultiOmicsClassifier
from losses import MultiOmicsLoss
from trainers import MultiOmicsTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Omics Model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results and checkpoints')
    return parser.parse_args()


def main():
    # Parse arguments and load config
    args = parse_args()
    config = Config.from_yaml(args.config)
    config_dict = config.to_dict()
    # Set up device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config.seed)

    # Create dataset and dataloaders
    print("Initializing dataset...")
    dataset = MultiOmicsDataset(config_dict)
    train_loader, val_loader, test_loader = create_dataloader(
        dataset=dataset,
        config=config.data.to_dict(),  # Convert Config to dict
        num_workers=config.data.num_workers
    )

    # Initialize model using config object
    print("Creating model...")
    model = MultiOmicsClassifier(
        mirna_dim=config.model.mirna_dim,
        rna_exp_dim=config.model.rna_exp_dim,
        methy_shape=tuple(config.model.methy_shape),
        latent_dim=config.model.latent_dim,
        num_classes=config.model.num_classes
    ).to(device)

    # Initialize loss and optimizer
    criterion = MultiOmicsLoss(
        beta=config.train.kl_weight,
        class_weights=config.train.get('class_weights', None),
        annealing_steps=config.train.annealing_steps
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay
    )

    # Create trainer with converted config dict
    trainer = MultiOmicsTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config.to_dict(),  # Convert entire config to dict
        checkpoint_dir=args.output_dir
    )

    # Start training
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.train.n_epochs
    )

    # Final evaluation
    print("\nRunning final evaluation...")
    test_loss, test_acc = trainer.test(test_loader)
    print(f"Test Results - Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")

    # Save final model
    final_path = Path(args.output_dir) / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()