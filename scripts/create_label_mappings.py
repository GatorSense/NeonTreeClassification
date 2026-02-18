#!/usr/bin/env python3
"""
Create label mapping JSON files for inference.

Extracts species-level (167 classes) and genus-level (60 classes) label mappings
from the training CSV and saves them as JSON files for use in inference.

Usage:
    python scripts/create_label_mappings.py --csv_path path/to/labels.csv
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_species_label_mapping(csv_path: str) -> dict:
    """
    Create species-level label mapping from CSV.

    Format: {
        "idx_to_code": {0: "PSMEM", 1: "TSHE", ...},
        "idx_to_name": {0: "Pseudotsuga menziesii...", ...},
        "code_to_idx": {"PSMEM": 0, ...},
        "name_to_idx": {"Pseudotsuga menziesii...": 0, ...},
        "metadata": {...}
    }
    """
    df = pd.read_csv(csv_path)

    # Get unique species (code, name pairs)
    species_df = df[["species", "species_name"]].drop_duplicates()

    # Sort by species code for consistency
    species_df = species_df.sort_values("species").reset_index(drop=True)

    # Create mappings
    idx_to_code = {idx: row["species"] for idx, row in species_df.iterrows()}
    idx_to_name = {idx: row["species_name"] for idx, row in species_df.iterrows()}
    code_to_idx = {row["species"]: idx for idx, row in species_df.iterrows()}
    name_to_idx = {row["species_name"]: idx for idx, row in species_df.iterrows()}

    # Count samples per species
    species_counts = df["species"].value_counts().to_dict()
    idx_to_count = {
        idx: species_counts.get(code, 0) for idx, code in idx_to_code.items()
    }

    # Metadata
    metadata = {
        "taxonomic_level": "species",
        "num_classes": len(idx_to_code),
        "total_samples": len(df),
        "source_csv": Path(csv_path).name,
        "description": "NEON tree species classification - Species level (USDA plant codes)",
        "label_format": "USDA plant symbol codes (e.g., PSMEM for Pseudotsuga menziesii)",
    }

    return {
        "idx_to_code": idx_to_code,
        "idx_to_name": idx_to_name,
        "code_to_idx": code_to_idx,
        "name_to_idx": name_to_idx,
        "idx_to_count": idx_to_count,
        "metadata": metadata,
    }


def create_genus_label_mapping(csv_path: str) -> dict:
    """
    Create genus-level label mapping from CSV.

    Format: {
        "idx_to_genus": {0: "Acer", 1: "Pinus", ...},
        "genus_to_idx": {"Acer": 0, ...},
        "genus_to_species": {"Acer": ["ACRU", "ACSAS", ...], ...},
        "metadata": {...}
    }
    """
    df = pd.read_csv(csv_path)

    # Extract genus from species_name (first word)
    df["genus"] = df["species_name"].apply(lambda x: str(x).split()[0])

    # Get unique genera sorted alphabetically
    unique_genera = sorted(df["genus"].unique())

    # Create mappings
    idx_to_genus = {idx: genus for idx, genus in enumerate(unique_genera)}
    genus_to_idx = {genus: idx for idx, genus in enumerate(unique_genera)}

    # Map genus to species codes
    genus_to_species = {}
    for genus in unique_genera:
        species_list = df[df["genus"] == genus]["species"].unique().tolist()
        genus_to_species[genus] = sorted(species_list)

    # Count samples per genus
    genus_counts = df["genus"].value_counts().to_dict()
    idx_to_count = {
        idx: genus_counts.get(genus, 0) for idx, genus in idx_to_genus.items()
    }

    # Count species per genus
    genus_to_species_count = {
        genus: len(species_list) for genus, species_list in genus_to_species.items()
    }

    # Metadata
    metadata = {
        "taxonomic_level": "genus",
        "num_classes": len(idx_to_genus),
        "total_samples": len(df),
        "source_csv": Path(csv_path).name,
        "description": "NEON tree species classification - Genus level",
        "label_format": "Genus names (first word of scientific name)",
        "extraction_method": "genus = species_name.split()[0]",
    }

    return {
        "idx_to_genus": idx_to_genus,
        "genus_to_idx": genus_to_idx,
        "genus_to_species": genus_to_species,
        "genus_to_species_count": genus_to_species_count,
        "idx_to_count": idx_to_count,
        "metadata": metadata,
    }


def save_json(data: dict, output_path: Path, compact: bool = False):
    """Save data as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        if compact:
            json.dump(data, f)
        else:
            json.dump(data, f, indent=2)

    print(f"✅ Saved: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """Create label mapping JSON files."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create label mapping JSON files for inference"
    )
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to combined_dataset.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: neon_tree_classification/inference/label_mappings/)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("CREATE LABEL MAPPING FILES FOR INFERENCE")
    print("=" * 80)

    # Paths
    project_root = Path(__file__).parent.parent
    csv_path = args.csv_path
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            project_root / "neon_tree_classification" / "inference" / "label_mappings"
        )
    )

    if not Path(csv_path).exists():
        print(f"❌ Error: CSV not found: {csv_path}")
        sys.exit(1)

    print(f"\n📂 Input CSV: {csv_path}")
    print(f"📁 Output directory: {output_dir}")

    # Create species mapping
    print("\n" + "=" * 80)
    print("1. SPECIES-LEVEL MAPPING (167 classes)")
    print("=" * 80)

    species_mapping = create_species_label_mapping(csv_path)
    print(f"\nCreated species mapping:")
    print(f"  • Classes: {species_mapping['metadata']['num_classes']}")
    print(f"  • Samples: {species_mapping['metadata']['total_samples']:,}")
    print(f"  • Format: {species_mapping['metadata']['label_format']}")

    print(f"\nExample mappings:")
    for idx in range(min(5, len(species_mapping["idx_to_code"]))):
        code = species_mapping["idx_to_code"][idx]
        name = species_mapping["idx_to_name"][idx]
        count = species_mapping["idx_to_count"][idx]
        print(f"  {idx:3d} → {code:8s} → {name[:50]:50s} ({count:5,} samples)")

    # Save species mapping
    species_output = output_dir / "species_labels.json"
    save_json(species_mapping, species_output)

    # Create genus mapping
    print("\n" + "=" * 80)
    print("2. GENUS-LEVEL MAPPING (60 classes)")
    print("=" * 80)

    genus_mapping = create_genus_label_mapping(csv_path)
    print(f"\nCreated genus mapping:")
    print(f"  • Classes: {genus_mapping['metadata']['num_classes']}")
    print(f"  • Samples: {genus_mapping['metadata']['total_samples']:,}")
    print(f"  • Format: {genus_mapping['metadata']['label_format']}")

    print(f"\nExample mappings:")
    for idx in range(min(5, len(genus_mapping["idx_to_genus"]))):
        genus = genus_mapping["idx_to_genus"][idx]
        count = genus_mapping["idx_to_count"][idx]
        species_list = genus_mapping["genus_to_species"][genus]
        print(
            f"  {idx:3d} → {genus:15s} ({count:5,} samples, {len(species_list)} species)"
        )
        print(
            f"       Species: {', '.join(species_list[:5])}{'...' if len(species_list) > 5 else ''}"
        )

    # Save genus mapping
    genus_output = output_dir / "genus_labels.json"
    save_json(genus_mapping, genus_output)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✅ Created 2 label mapping files:")
    print(f"   1. {species_output}")
    print(f"      - {species_mapping['metadata']['num_classes']} species classes")
    print(f"      - USDA plant codes (e.g., PSMEM)")
    print(f"   2. {genus_output}")
    print(f"      - {genus_mapping['metadata']['num_classes']} genus classes")
    print(f"      - Genus names (e.g., Pseudotsuga)")

    print(f"\n📊 Class Distribution:")
    print(f"   Species level: {species_mapping['metadata']['num_classes']} classes")
    print(f"   Genus level:   {genus_mapping['metadata']['num_classes']} classes")
    print(
        f"   Reduction:     {species_mapping['metadata']['num_classes'] / genus_mapping['metadata']['num_classes']:.1f}x"
    )

    print("\n" + "=" * 80)
    print("✅ LABEL MAPPING CREATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Use these files in inference module")
    print("2. Load with: json.load(open('species_labels.json'))")
    print("3. Access mappings: data['idx_to_code'], data['idx_to_name'], etc.")
    print("\nUsage example:")
    print(f"  python {Path(__file__).name} --csv_path /path/to/combined_dataset.csv")


if __name__ == "__main__":
    main()
