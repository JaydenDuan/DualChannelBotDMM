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

Basic Training follow command below:

python scripts/train.py \
    --cross_validation_dir dual_format_cv/cross_validation \
    --results_dir experiments/exp_001 \
    --embedding_dim 128 \
    --feature_dim 128 \
    --num_steps 5 \
    --dropout 0.3 \
    --lr 0.00005 \
    --weight_decay 0.0005 \
    --epochs 100 \
    --patience 20 \
    --clip_grad 1.0 \
    --lambda_feature_contrast 0.1 \
    --lambda_class_contrast 0.1 \
    --temperature 0.1 \
    --alpha 0.5 \
    --seed 42 \
    
--embedding_dim: Dimension of feature embeddings (default: 128)
--num_temporal_steps: Number of temporal snapshots (default: 5)
--alpha: Orthogonal constraint weight (default: 0.5)

