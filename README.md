# BotDMM: Dual-channel Multi-Modal Learning for LLM-driven Bot Detection on Social Media

**Authors**: Jinglong Duan, Shiqing Wu, Weihua Li, Quan Bai, Minh Nguyen, Jianhua Jiang

**Link**: https://www.sciencedirect.com/science/article/pii/S1566253525008206

## Overview

This repository contains the implementation of BotDMM (Bot Detection with Multi-Modal learning), a novel framework for detecting LLM-driven bots on social media platforms. BotDMM introduces a dual-channel architecture that orthogonally decouples content and structure representations while modeling temporal dynamics across user interactions.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- scikit-learn
- tqdm

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric scikit-learn tqdm
```

## Datasets

### Twibot-20 (3-class classification)

Download: https://drive.google.com/drive/folders/1da2yI0oLGvEzG-TOJ8UohJSMX-9rnPYb?usp=drive_link

Classes: Human, Traditional Bot, LLM-driven Bot

### BotSim (Binary classification)

Download: https://drive.google.com/drive/folders/1da2yI0oLGvEzG-TOJ8UohJSMX-9rnPYb?usp=drive_link

Classes: Human, Bot

## Data Structure

```
Dataset/
├── descriptions/
│   ├── train_des_tensor.pt
│   ├── val_des_tensor.pt
│   └── test_des_tensor.pt
├── numerical/
│   ├── {train,val,test}_num_properties_tensor.pt
│   └── {train,val,test}_llm_features.pt
├── {train,val,test}_text_tensor{1-5}.pt
├── {train,val,test}_amr_tensor{1-5}.pt
├── {train,val,test}_edge_index_part{1-5}.pt
├── {train,val,test}_labels.pt          # Twibot-20
└── labels/                              # BotSim
    └── {train,val,test}.json
```

## Training

### Twibot-20

```bash
python scripts/train.py --data_dir Dataset/Twibot20 --dataset twibot20
```

### BotSim

```bash
python scripts/train.py --data_dir Dataset/BotSim --dataset botsim
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | Dataset/Twibot20 | Data directory |
| `--dataset` | twibot20 | Dataset: twibot20 or botsim |
| `--embedding_dim` | 128 | Embedding dimension |
| `--feature_dim` | 128 | Feature dimension |
| `--num_steps` | 5 | Number of temporal steps |
| `--dropout` | 0.3 | Dropout rate |
| `--lr` | 0.00005 | Learning rate |
| `--epochs` | 100 | Number of epochs |
| `--batch_size` | 32 | Batch size |
| `--alpha` | 0.5 | Orthogonal constraint coefficient |
| `--temperature` | 0.1 | Contrastive learning temperature |
| `--patience` | 20 | Early stopping patience |

## Citation

If you make use of this code or the BotDMM algorithm in your work, please cite the following paper:

```bibtex
@article{duan2025botdmm,
  title={BotDMM: Dual-channel Multi-Modal Learning for LLM-driven Bot Detection on Social Media},
  author={Duan, Jinglong and Wu, Shiqing and Li, Weihua and Bai, Quan and Nguyen, Minh and Jiang, Jianhua},
  journal={Information Fusion},
  pages={103758},
  year={2025},
  publisher={Elsevier}
}
```
