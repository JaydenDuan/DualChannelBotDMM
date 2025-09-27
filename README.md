BotDMM: Dual-channel Multi-Modal Learning for LLM-driven Bot Detection on Social Media

Authors: Jinglong Duan, Shiqing Wu, Weihua Li, Quan Bai, Minh Nguyen, Jianhua Jiang 
Article Url: https://www.sciencedirect.com/science/article/pii/S1566253525008206

Overview
This repository contains the implementation of BotDMM (Bot Detection with Multi-Modal learning), a novel framework for detecting LLM-driven bots on social media platforms. BotDMM introduces a dual-channel architecture that orthogonally decouples content and structure representations while modeling temporal dynamics across user interactions. Our approach is particularly effective for detecting sophisticated bots powered by large language models, which exhibit more human-like behavior patterns than traditional bots.

@article{duan2025botdmm,
  title={BotDMM: Dual-channel Multi-Modal Learning for LLM-driven Bot Detection on Social Media},
  author={Duan, Jinglong and Wu, Shiqing and Li, Weihua and Bai, Quan and Nguyen, Minh and Jiang, Jianhua},
  journal={Information Fusion},
  pages={103758},
  year={2025},
  publisher={Elsevier}
}

Requirements:
Recent versions of PyTorch, Transformers, PyTorch Geometric, and standard ML libraries are required. You can install all required packages using:

$ pip install -r requirements.txt

Basic Training:

python scripts/train.py \
    --config configs/config.yaml \
    --mode cv \
    --device cuda
    
--embedding_dim: Dimension of feature embeddings (default: 128)
--num_temporal_steps: Number of temporal snapshots (default: 5)
--alpha: Orthogonal constraint weight (default: 0.5)

