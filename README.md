# BKDGEC: Semi-supervised learning and bidirectional decoding for effective grammar correction in low-resource scenarios

## Overview
This project introduces a Grammatical Error Correction (GEC) framework as a machine translation task for low-resource languages, specifically applied to the Arabic language. The framework consists of two main components:

1. **EDSE (Equal Distribution of Synthetic Errors)**

     
<p align="center">
  <img src="/images%20and%20diagrams/data.png" alt="EDSE Architecture" width="70%">
  <br>
  <i>Architecture of the Equal Distribution of Synthetic Errors (EDSE) approach is made of two synthetic pipelines that have the same structure but generate different types of errors</i>
</p>

2. **BKDGEC (Bidirectional Knowledge Distillation for GEC)**
   - Utilizes two decoders:
     - Forward decoder (right-to-left)
     - Backward decoder (left-to-right)
   - Implements knowledge distillation through regularization
   - Uses Kullback-Leibler divergence to measure decoder agreement
   - Leverages backward decoder's information about longer-term future
   - Encourages auto-regressive GEC models to plan ahead
   - Joint training process for both decoders
  
<p align="center">
  <img src="/images%20and%20diagrams/model_gec.png" alt="BKDGEC Architecture" width="70%">
  <br>
  <i>The architecture of the BKDGEC model incorporates two decoders, labeled as Backward and Forward, represented by yellow boxes. These decoders consist of Self-Attention (SA), Cross-Attention
(CA), and a Feed-Forward Neural Network (FFN)</i>
</p>

## Requirements

### Python Environment
- Python 3.6.10 or later

### Core Dependencies
```
pytorch==1.6.0
torchtext==0.6.0
numpy==1.17.0
pandas==1.17.0
PyArabic==0.6.10
bpemb==0.3.2
nltk==3.3
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Project Structure
```
BKDGEC/
├── configs/            # Configuration files
├── data/              # Data directories
│   ├── raw/           # Raw input data
│   ├── processed/     # Processed data
│   └── synthetic/     # EDSE generated data
├── models/            # Model implementations
├── scripts/           # Utility scripts
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Usage

1. Data Generation (EDSE):
```bash
python scripts/edse_generate.py --input data/raw/input.txt --output data/synthetic/
```

2. Training:
```bash
python train_bkdgec.py --config configs/train_config.yml
```

3. Evaluation:
```bash
python evaluate.py --model_path checkpoints/best_model.pt --test_data data/test/
```

## Citation
If you use this code in your research, please cite our paper:

```bibtex
Mahmoud, Zeinab, Chunlin Li, Marco Zappatore, Aiman Solyman, Ali Alfatemi, Ashraf Osman Ibrahim, and Abdelzahir Abdelmaboud. "Semi-supervised learning and bidirectional decoding for effective grammar correction in low-resource scenarios." PeerJ Computer Science 9 (2023): e1639.
```

## Contact
For questions and issues, please open an issue in the repository.
