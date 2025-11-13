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
   
Employs two complementary decoders—one forward (right-to-left) and one backward (left-to-right)—that are trained jointly through a knowledge distillation mechanism with regularization. The model uses Kullback–Leibler divergence to measure agreement between the two decoding directions, allowing the backward decoder to provide information about longer-term future context while guiding the forward decoder to make more consistent predictions. This bidirectional setup mitigates exposure bias, encourages autoregressive GEC models to plan ahead, and enhances overall grammatical correction accuracy through a unified joint training process.
  
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
│
├── configs/ # Configuration files (EDSE + training configs)
│ └── configs.yml
│
├── data/ # Data directory 
│ ├── source_text.txt # Clean Arabic corpus 
│ ├── vocab.txt # Vocabulary 
│ └── train.csv # Dataset (src, trg)
│
├── images and diagrams/ # Visual figures 
│
├─ m2Scripts/
│ ├─ 2015_gold.m2 # QALB-2015 official M² annotations
│ └─ 2014_gold.m2 # QALB-2014 official M² annotations
│
├── scripts/ # Core project scripts
│ ├── bkdgec_model.py # BKDGEC model implementation
│ ├── generate_edse.py # EDSE data generation script
│ └── train_bkdgec.py # Model training script 
│
├── synthetic training data/ # The generated synthetic data
│
├── trained models/ # Saved checkpoints 
│
├── system outputs/ # Inference outputs
│
├── requirements.txt # Python dependencies
│
└── README.md # Project documentation
```

## Usage

1. Data Generation (EDSE):
```bash
python scripts/edse_generate.py --input data/source_text.txt --output data/
```

2. Training:
```bash
python scripts/train_bkdgec.py \
  --input data/train.csv \
  --config configs/configs.yml
```

3. Evaluation:
```bash
python m2Scripts/m2scorer.py \
  system_outputs/pred.txt \
  m2Scripts/QALB_test2014.m2
```

## Citation
If you use this code in your research, please cite our paper:

```bibtex
@article{mahmoud2023semi,
  title={Semi-supervised learning and bidirectional decoding for effective grammar correction in low-resource scenarios},
  author={Mahmoud, Zeinab and Li, Chunlin and Zappatore, Marco and Solyman, Aiman and Alfatemi, Ali and Ibrahim, Ashraf Osman and Abdelmaboud, Abdelzahir},
  journal={PeerJ Computer Science},
  volume={9},
  pages={e1639},
  year={2023},
  publisher={PeerJ Inc.}
}
```

## Contact
For questions and issues, please open an issue in the repository.
