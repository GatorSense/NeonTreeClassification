# NEON Tree Classification Dataset - Documentation

Welcome to the comprehensive documentation for the NEON Multi-Modal Tree Species Classification Dataset.

## Quick Links

- [README](../README.md) - Main README with dataset overview and quick start
- [Advanced Usage](advanced_usage.md) - Custom filtering, Lightning DataModule, and advanced features
- [Training Guide](training.md) - Model training examples and baseline results
- [Visualization Guide](visualization.md) - Data visualization tools and examples
- [Processing Pipeline](processing.md) - NEON data processing workflow

## Getting Started

1. **New Users**: Start with the [main README](../README.md) for installation and basic usage
2. **Training Models**: See the [Training Guide](training.md) for model training and baseline results
3. **Data Exploration**: Check out the [Visualization Guide](visualization.md) for exploring the dataset
4. **Advanced Features**: Read [Advanced Usage](advanced_usage.md) for custom configurations
5. **Data Processing**: For processing raw NEON data, see the [Processing Pipeline](processing.md)

## Documentation Structure

### [Advanced Usage](advanced_usage.md)
- Custom data filtering with Lightning DataModule
- Split methods (random, site-based, year-based)
- External test sets
- Advanced dataloader configuration
- Direct dataset usage
- Multi-GPU training
- Custom training loops

### [Training Guide](training.md)
- Quick training with examples script
- Baseline results and reproduction steps
- Custom model architectures
- Training best practices
- Multi-modal training
- Experiment tracking (Comet ML, W&B)
- Common issues and solutions

### [Visualization Guide](visualization.md)
- Overview of visualization tools
- RGB, HSI, and LiDAR visualization
- Interactive Jupyter notebook
- Custom visualizations
- Multi-modal comparisons
- Advanced spectral analysis

### [Processing Pipeline](processing.md)
- Complete data processing workflow
- NEON data product details
- Quality control procedures
- HDF5 dataset creation
- Configuration subset creation
- Processing best practices

## Support

For issues, questions, or contributions:
- GitHub Issues: [Report a bug or request a feature](https://github.com/Ritesh313/NeonTreeClassification/issues)
- Contributing: See [CONTRIBUTING.md](../CONTRIBUTING.md)

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{neon_tree_classification_2024,
  title={NEON Multi-Modal Tree Species Classification Dataset},
  author={[Author Names]},
  year={2024},
  publisher={GitHub},
  url={https://github.com/Ritesh313/NeonTreeClassification}
}
```

## License

See [LICENSE](../LICENSE) file for details.

## Acknowledgments

- National Ecological Observatory Network (NEON)
- Dataset statistics generated on 2025-08-28
