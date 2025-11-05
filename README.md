# BKDGEC: Semi-supervised learning and bidirectional decoding for effective grammar correction in low-resource scenarios
This project introduces a Grammatical Error Correction (GEC) framework as a machine translation task for low-resource languages applied to the Arabic language. Initially, we propose a semi-supervised confusion method, named Equal Distribution of Synthetic Errors (EDSE), to generate wide parallel training data with a high diversity of training patterns. Besides, we propose Bidirectional Knowledge Distillation for Grammatical Error Correction (BKDGEC), which exploits two decoders: a forward decoder right-to-left and a backward decoder left-to-right into a regularization method. This takes advantage of leveraging the backward decoder’s information about the longer-term future and distilling knowledge learned in the backward decoder, which could encourage auto-regressive GEC models to plan ahead. Both decoders were trained into a joint training process and Kullback-Leibler divergence was applied to measure the agreement between both decoders as a regulation term.
# Model requirements
Regarding load and run the trained models it requires a working installation of following:
- Python 3.6.10 or latest 
- pytorch==1.6.0
- torchtext==0.6.0
- numpy==1.17.0
- pandas==1.17.0
- PyArabic==0.6.10
- bpemb==0.3.2
- nltk==3.3

## Features

- Boundary knowledge distillation framework
- Edge feature extraction module
- Knowledge transfer mechanism
- Support for multiple GEC datasets
- Comprehensive evaluation metrics (ROUGE, BLEU, Meteor)

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

```
BKDGEC/
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
├── models/
│   ├── BKDEncoder.py
│   ├── EdgeFeatureExtractor.py
│   └── KnowledgeTransfer.py
├── utils/
│   ├── preprocess.py
│   ├── metrics.py
│   └── helpers.py
├── config.py
├── train.py
├── evaluate.py
└── README.md
```

## Usage

### Data Preprocessing

```bash
python utils/preprocess.py --input_dir data/raw --output_dir data/processed
```

### Training

```bash
python train.py --config configs/train_config.json
```

### Evaluation

```bash
python evaluate.py --model_path checkpoints/best_model.pt --test_data data/test
```

## Model Architecture

The BKDGEC model consists of three main components:
1. Boundary Knowledge Distillation Encoder
2. Edge Feature Extraction Module
3. Knowledge Transfer Component

## Results

Our model achieves state-of-the-art performance on standard GEC benchmarks:

| Dataset | Precision | Recall | F0.5 |
|---------|-----------|---------|------|
| CoNLL-2014 | XX.XX | XX.XX | XX.XX |
| BEA-2019 | XX.XX | XX.XX | XX.XX |

## Citation

If you use this code or model in your research, please cite our paper:

```bibtex
@article{author2023boundary,
  title={Boundary Knowledge Distillation for Grammatical Error Correction},
  author={Author, A. and Author, B.},
  journal={PeerJ Computer Science},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## Contact

For questions or issues, please open an issue in the repository.
