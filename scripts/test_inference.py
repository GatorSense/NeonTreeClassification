#!/usr/bin/env python3
"""
Test inference module with sample data from HDF5.

This script:
1. Loads a checkpoint
2. Extracts sample images from HDF5
3. Runs inference
4. Validates predictions

Usage:
    python scripts/test_inference.py \
        --checkpoint path/to/best.ckpt \
        --csv_path path/to/combined_dataset.csv \
        --hdf5_path path/to/neon_dataset.h5 \
        --taxonomic_level species \
        --num_samples 5
"""

import argparse
import sys
from pathlib import Path
import torch
import h5py
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neon_tree_classification.inference import TreeClassifier
from neon_tree_classification.inference.utils import print_prediction_summary


def test_inference(
    checkpoint_path: str,
    csv_path: str,
    hdf5_path: str,
    taxonomic_level: str = 'species',
    num_samples: int = 5,
    top_k: int = 5,
):
    """Test inference on sample data."""
    
    print("=" * 80)
    print("NEON TREE CLASSIFICATION - INFERENCE TEST")
    print("=" * 80)
    
    # Step 1: Load model
    print("\n📦 Step 1: Loading model...")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Level: {taxonomic_level}")
    
    classifier = TreeClassifier.from_checkpoint(
        checkpoint_path=checkpoint_path,
        taxonomic_level=taxonomic_level,
        model_type='resnet',
    )
    
    print(f"\n✅ Model loaded: {classifier}")
    
    # Step 2: Load sample data
    print(f"\n📊 Step 2: Loading {num_samples} random samples from HDF5...")
    df = pd.read_csv(csv_path)
    
    # Sample random crown IDs
    sample_df = df.sample(n=num_samples, random_state=42)
    print(f"   Selected samples:")
    
    # Step 3: Run inference on each sample
    print(f"\n🔍 Step 3: Running inference...")
    
    with h5py.File(hdf5_path, 'r') as hf:
        for idx, (i, row) in enumerate(sample_df.iterrows(), 1):
            crown_id = str(row['crown_id'])
            gt_species = row['species']
            gt_name = row['species_name']
            
            print(f"\n{'='*80}")
            print(f"Sample {idx}/{num_samples}")
            print(f"{'='*80}")
            print(f"Crown ID: {crown_id}")
            print(f"Site: {row['site']}, Year: {row['year']}")
            
            # Extract genus from species name
            gt_genus = gt_name.split()[0]
            
            if taxonomic_level == 'species':
                print(f"Ground Truth: {gt_species} - {gt_name}")
            else:
                print(f"Ground Truth Genus: {gt_genus}")
            
            # Load RGB image from HDF5
            if crown_id not in hf['rgb']:
                print(f"   ⚠️  Crown ID {crown_id} not found in HDF5, skipping")
                continue
            
            rgb_data = hf['rgb'][crown_id][:]  # Shape: (H, W, 3), values 0-255
            print(f"Image shape: {rgb_data.shape}, dtype: {rgb_data.dtype}")
            print(f"Value range: [{rgb_data.min()}, {rgb_data.max()}]")
            
            # Run prediction
            result = classifier.predict(rgb_data, top_k=top_k)
            
            # Display results
            print(f"\n🎯 Predictions (top {top_k}):")
            print(f"   Confidence: {result['top_probability']:.2%}")
            print(f"   Entropy: {result['entropy']:.3f}")
            
            for j, pred in enumerate(result['predictions'], 1):
                if taxonomic_level == 'species':
                    code = pred['species_code']
                    name = pred['species_name']
                    is_correct = "✓" if code == gt_species else " "
                    print(f"   {is_correct} {j}. [{pred['probability']:6.2%}] {code:10s} - {name[:50]}")
                else:
                    genus = pred['genus']
                    is_correct = "✓" if genus == gt_genus else " "
                    print(f"   {is_correct} {j}. [{pred['probability']:6.2%}] {genus}")
            
            # Check if ground truth is in top-k
            if taxonomic_level == 'species':
                top_codes = [p['species_code'] for p in result['predictions']]
                if gt_species in top_codes:
                    rank = top_codes.index(gt_species) + 1
                    print(f"\n   ✅ Ground truth found at rank {rank}")
                else:
                    print(f"\n   ❌ Ground truth not in top-{top_k}")
            else:
                top_genera = [p['genus'] for p in result['predictions']]
                if gt_genus in top_genera:
                    rank = top_genera.index(gt_genus) + 1
                    print(f"\n   ✅ Ground truth genus found at rank {rank}")
                else:
                    print(f"\n   ❌ Ground truth genus not in top-{top_k}")
    
    # Step 4: Test batch prediction
    print(f"\n{'='*80}")
    print(f"🔄 Step 4: Testing batch prediction...")
    print(f"{'='*80}")
    
    batch_samples = df.sample(n=3, random_state=123)
    batch_images = []
    batch_ids = []
    
    with h5py.File(hdf5_path, 'r') as hf:
        for _, row in batch_samples.iterrows():
            crown_id = str(row['crown_id'])
            if crown_id in hf['rgb']:
                batch_images.append(hf['rgb'][crown_id][:])
                batch_ids.append(crown_id)
    
    if len(batch_images) > 0:
        print(f"Running batch prediction on {len(batch_images)} images...")
        batch_results = classifier.predict_batch(batch_images, top_k=3)
        
        for i, (crown_id, result) in enumerate(zip(batch_ids, batch_results), 1):
            top_pred = result['predictions'][0]
            if taxonomic_level == 'species':
                label = f"{top_pred['species_code']} - {top_pred['species_name'][:40]}"
            else:
                label = top_pred['genus']
            print(f"   {i}. Crown {crown_id}: {label} ({result['top_probability']:.2%})")
        
        print(f"✅ Batch prediction successful!")
    
    # Step 5: Test get_class_probabilities
    print(f"\n{'='*80}")
    print(f"📊 Step 5: Testing get_class_probabilities()...")
    print(f"{'='*80}")
    
    with h5py.File(hdf5_path, 'r') as hf:
        test_crown_id = str(sample_df.iloc[0]['crown_id'])
        if test_crown_id in hf['rgb']:
            test_image = hf['rgb'][test_crown_id][:]
            probs = classifier.get_class_probabilities(test_image)
            
            print(f"Probability distribution:")
            print(f"   Shape: {probs.shape}")
            print(f"   Sum: {probs.sum():.6f} (should be ~1.0)")
            print(f"   Max: {probs.max():.4f}")
            print(f"   Min: {probs.min():.6f}")
            print(f"✅ Probability distribution valid!")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ INFERENCE TEST COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll tests passed successfully!")
    print(f"Model: {checkpoint_path}")
    print(f"Level: {taxonomic_level} ({classifier.num_classes} classes)")
    print(f"Device: {classifier.device}")
    print(f"\nInference module is ready for use! 🎉")


def main():
    parser = argparse.ArgumentParser(description="Test inference module")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to combined_dataset.csv'
    )
    parser.add_argument(
        '--hdf5_path',
        type=str,
        required=True,
        help='Path to neon_dataset.h5'
    )
    parser.add_argument(
        '--taxonomic_level',
        type=str,
        default='species',
        choices=['species', 'genus'],
        help='Taxonomic level for classification'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to test'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of top predictions to show'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not Path(args.csv_path).exists():
        print(f"❌ Error: CSV not found: {args.csv_path}")
        sys.exit(1)
    
    if not Path(args.hdf5_path).exists():
        print(f"❌ Error: HDF5 not found: {args.hdf5_path}")
        sys.exit(1)
    
    # Run test
    test_inference(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv_path,
        hdf5_path=args.hdf5_path,
        taxonomic_level=args.taxonomic_level,
        num_samples=args.num_samples,
        top_k=args.top_k,
    )


if __name__ == '__main__':
    main()
