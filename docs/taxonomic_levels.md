# Taxonomic Level Classification

Train tree species classification models at different taxonomic levels (species or genus) with the same codebase.

## Quick Start

```python
from neon_tree_classification.core.datamodule import NeonCrownDataModule

# Species-level classification (167 classes - more challenging)
datamodule = NeonCrownDataModule(
    csv_path="data/metadata/combined_dataset.csv",
    hdf5_path="data/combined_dataset.h5",
    modalities=["rgb"],
    taxonomic_level="species",  # Default
    batch_size=32,
)

# Genus-level classification (60 classes - easier, better for initial experiments)
datamodule = NeonCrownDataModule(
    csv_path="data/metadata/combined_dataset.csv",
    hdf5_path="data/combined_dataset.h5",
    modalities=["rgb"],
    taxonomic_level="genus",  # Extract genus from species names
    batch_size=32,
)
```

## Taxonomic Levels

### Species Level (Default)
- **Classes**: 167 unique species
- **Label format**: USDA plant codes (e.g., "ACRU", "PSMEM")
- **Full names**: e.g., "Acer rubrum L.", "Pseudotsuga menziesii"
- **Use when**: You need fine-grained species identification

### Genus Level
- **Classes**: 60 unique genera  
- **Label format**: Genus names (e.g., "Acer", "Pseudotsuga")
- **Extraction**: First word from species_name column
- **Use when**: 
  - Initial model development and testing (~3x fewer classes)
  - Evaluating model architectures
  - Limited training data or compute
  - Ecological studies at genus level

## Class Distribution

| Level | Classes | Top Class | Samples | Rare Classes (< 10 samples) |
|-------|---------|-----------|---------|----------------------------|
| **Species** | 167 | Acer rubrum | 5,684 (11.8%) | 14 (8.4%) |
| **Genus** | 60 | Quercus | 7,479 (15.6%) | 5 (8.3%) |

**Expected Performance Difference**: Genus-level accuracy typically 10-20% higher than species-level due to:
- Fewer classes (60 vs 167)
- More samples per class (average ~800 vs ~287)
- Less inter-class confusion

## Data Quality Check

**⚠️ IMPORTANT**: Always inspect your labels before training at genus level!

### Step 1: Run Label Inspection

```bash
python processing/misc/inspect_labels.py --csv_path path/to/your/labels.csv
```

This will show:
- All 60 genus names with sample counts
- Complete genus → species mappings
- Special cases and potential issues
- Edge cases (Unknown, Pinaceae, etc.)

### Step 2: Review Output

Look for potential issues:

✅ **Normal cases** (59 genera):
```
Acer          6,635 samples    10 species    (Maples)
Quercus       7,479 samples    27 species    (Oaks)
Pinus         6,600 samples    19 species    (Pines)
```

⚠️ **Edge cases to be aware of**:

1. **Unknown species** (147 samples, 0.31%)
   - Label: "Unknown plant", "Unknown softwood plant"
   - Genus extracted: "Unknown"
   - **Status**: Valid class representing unidentified species
   - **Action**: Keep or filter - your choice

2. **Pinaceae** (26 samples, 0.05%)
   - Label: "Pinaceae sp."
   - Genus extracted: "Pinaceae" (actually a **family name**, not genus)
   - Represents truly unidentified conifers from WREF site
   - **Status**: Minor edge case, negligible impact
   - **Action**: Keep (recommended) or filter

### Step 3: Filtering (Optional)

If you want taxonomically pure genus-level training:

```python
# Option A: Include only specific species codes (all others are excluded)
datamodule = NeonCrownDataModule(
    ...,
    taxonomic_level="genus",
    species_filter=["PSMEM", "TSHE"],  # Include only these USDA codes
)

# Option B: Build an inclusion list after inspecting
# See inspect_labels.py output for USDA codes present in your data
# species_filter keeps only rows WHERE species IS IN the list
all_codes = [...]  # full list from inspect_labels.py
species_to_include = [c for c in all_codes if c not in ["PINACE", "2PLANT", "2PLANT-S"]]
datamodule = NeonCrownDataModule(
    ...,
    taxonomic_level="genus",
    species_filter=species_to_include,
)
```

## Genus Extraction Method

The genus extraction is simple and robust:

```python
genus = species_name.split()[0]
```

**Examples**:
```
"Acer rubrum L."                                          → "Acer"
"Pseudotsuga menziesii (Mirb.) Franco var. menziesii"   → "Pseudotsuga"
"Betula papyrifera Marshall"                             → "Betula"
"Pinaceae sp."                                           → "Pinaceae" (family name, but treated as genus)
```

This method:
- ✅ Works for all 167 species in the dataset
- ✅ Handles varieties and subspecies automatically
- ✅ Requires no manual mapping or preprocessing
- ✅ Validated against 47,971 samples with 99.7% consistency

## Training Examples

### Basic Training

```python
import lightning as L
from neon_tree_classification.core.datamodule import NeonCrownDataModule
from neon_tree_classification.models.lightning_modules import RGBClassifier

# Setup data at genus level
datamodule = NeonCrownDataModule(
    csv_path="data/metadata/combined_dataset.csv",
    hdf5_path="data/combined_dataset.h5",
    modalities=["rgb"],
    taxonomic_level="genus",  # 60 classes
    batch_size=64,
)

# Create model (num_classes will be auto-set by Lightning from datamodule)
model = RGBClassifier(
    model_type="resnet50",  # Use pretrained ResNet50
    num_classes=60,  # Will match datamodule
    learning_rate=1e-3,
)

# Train
trainer = L.Trainer(max_epochs=50, accelerator="gpu")
trainer.fit(model, datamodule)
```

### With Filtering

```python
# Clean genus-level training (include only true genera, omit edge cases)
# species_filter keeps only rows where species code is in the list
all_codes = [...]  # get from inspect_labels.py output
clean_codes = [c for c in all_codes if c not in ["PINACE"]]  # drop Pinaceae
datamodule = NeonCrownDataModule(
    csv_path="data/metadata/combined_dataset.csv",
    hdf5_path="data/combined_dataset.h5",
    modalities=["rgb"],
    taxonomic_level="genus",
    species_filter=clean_codes,  # include all except Pinaceae
    batch_size=64,
)
# Now training on 59 true genera only
```

### Progressive Training Strategy

```python
# Phase 1: Genus-level baseline (fast iteration)
genus_datamodule = NeonCrownDataModule(..., taxonomic_level="genus")
genus_model = RGBClassifier(model_type="resnet50", num_classes=60)
trainer.fit(genus_model, genus_datamodule)
# Expected: ~75-85% test accuracy

# Phase 2: Species-level fine-tuning (final model)
species_datamodule = NeonCrownDataModule(..., taxonomic_level="species")
species_model = RGBClassifier(model_type="resnet50", num_classes=167)
trainer.fit(species_model, species_datamodule)
# Expected: ~65-75% test accuracy
```

## Command-Line Usage

```bash
# Train at genus level
python examples/train.py \
    --csv_path data/metadata/combined_dataset.csv \
    --hdf5_path data/combined_dataset.h5 \
    --modality rgb \
    --taxonomic_level genus \
    --model_type resnet50 \
    --batch_size 64 \
    --epochs 50

# Train at species level
python examples/train.py \
    --csv_path data/metadata/combined_dataset.csv \
    --hdf5_path data/combined_dataset.h5 \
    --modality rgb \
    --taxonomic_level species \
    --model_type resnet50 \
    --batch_size 64 \
    --epochs 50
```

## Model Considerations

### num_classes Parameter

**Important**: Make sure your model's `num_classes` matches your taxonomic level!

```python
# Species level
datamodule = NeonCrownDataModule(..., taxonomic_level="species")  # 167 classes
model = RGBClassifier(num_classes=167)  # ✓ Correct

# Genus level
datamodule = NeonCrownDataModule(..., taxonomic_level="genus")  # 60 classes
model = RGBClassifier(num_classes=60)  # ✓ Correct
```

The number of classes will vary slightly based on your filtering:
- Species level: 167 classes (default)
- Genus level: 60 classes (default), 59 if filtering Pinaceae

## Validation Warnings

When using `taxonomic_level="genus"`, the DataModule automatically validates genus extraction and warns about:

1. **Non-alphabetic genus names** (e.g., "Unknown", "2PLANT")
2. **Known family names** (e.g., "Pinaceae")
3. **Sample counts for edge cases**

Example warning:
```
UserWarning: Found family names treated as genera: {'Pinaceae': 26}. 
These represent unidentified species within that family. 
See docs/taxonomic_levels.md for more information.
```

**These are informational** - training will proceed normally. To exclude them, build an inclusion list with all other codes and pass it to `species_filter` (which keeps only species in the list).

## FAQ

**Q: Should I train at genus or species level?**
- Start with **genus** for faster iteration and architecture selection
- Move to **species** for final production models and fine-grained identification

**Q: Can I use pretrained weights from genus-level for species-level?**
- Yes! Transfer learning between taxonomic levels works well
- The backbone features transfer, just replace the classification head

**Q: What about Pinaceae?**
- It's a family name, not genus, but only 26 samples (0.05%)
- Keep it (recommended): Represents "unidentified conifer" class
- Exclude it: Build an inclusion list of all codes except `"PINACE"` and pass to `species_filter`

**Q: How do I know how many classes I have?**
```python
datamodule.setup()
print(f"Number of classes: {datamodule.full_dataset.num_classes}")
print(f"Class names: {datamodule.full_dataset.idx_to_label}")
```

**Q: Can I add more taxonomic levels (family, order)?**
- Yes! The same pattern extends to any taxonomic level
- Would need to modify genus extraction logic
- Contact maintainers if this is needed

## Performance Benchmarks

Expected accuracy ranges on NEON combined dataset (RGB only, ResNet50):

| Taxonomic Level | Classes | Baseline | With Pretrained | With Tuning |
|-----------------|---------|----------|----------------|-------------|
| **Genus** | 60 | 70-75% | 75-80% | 80-85% |
| **Species** | 167 | 50-55% | 65-70% | 70-75% |

*Note: Actual performance depends on data quality, hyperparameters, and training strategy*

## Additional Resources

- **Data inspection**: `python processing/misc/inspect_labels.py --csv_path path/to/labels.csv`
- **Training examples**: `examples/train.py`
- **Model architectures**: `docs/training.md`
- **Data processing**: `docs/processing.md`

## Citation

If you use genus-level classification in your research, please cite both the package and note the taxonomic level in your methods.
