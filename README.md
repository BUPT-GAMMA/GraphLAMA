# GraphLAMA: Enabling Efficient Adaptation of Graph Language Models with Limited Annotations

This repository contains the official implementation and resources for GraphLAMA, an approach for efficiently adapting graph language models with limited annotations. The paper has been accepted to KDD 2025.

### 1. Environment Preparation
```shell
# Python
conda create -n graphlama python=3.8
conda activate graphlama

# PyTorch with CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# Optional: support for Vicuna base model via FastChat
pip3 install "fschat[model_worker,webui]"

# PyG and related packages (for torch 1.13.0 + cu117)
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# Project dependencies
pip install -r requirements.txt
```

### 2. Prepare Models and Data
- Base LLM weights (if applicable): Vicuna weights are available from the FastChat project: [link](https://github.com/lm-sys/FastChat#model-weights).
- Pre-training data: we follow the data format used in GraphGPT; please refer to their data preparation instructions at [GraphGPT](https://github.com/HKUDS/GraphGPT).
- Test-time tuning data: see `data/cora/reshape_cora.py` for an example pipeline.

### 3. Pre-training Stage
```shell
cd path/to/grapht3
sh ./scripts/tune_script/stage1.sh
sh ./scripts/tune_script/stage2.sh
```

### 4. Test-time Tuning Stage
```shell
cd path/to/grapht3
sh ./scripts/tune_script/SFTonGFM_train.sh
```

### 5. Evaluate
```shell
cd path/to/grapht3
sh ./scripts/tune_script/SFTonGFM_eval.sh
```

### 6. Paper
- Title: GraphLAMA: Enabling Efficient Adaptation of Graph Language Models with Limited Annotations
- Venue: KDD 2025
- Preprint: coming soon

### 7. Citation
If you find this repository useful, please cite our paper (BibTeX coming soon):
```
@inproceedings{graphlama_kdd2025,
  title     = {GraphLAMA: Enabling Efficient Adaptation of Graph Language Models with Limited Annotations},
  author    = {To be updated},
  booktitle = {Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2025}
}
```

### 8. Acknowledgements
This work builds upon open-source efforts, including the GraphGPT framework: https://github.com/HKUDS/GraphGPT. We thank the authors and community for their contributions.

### 9. License
This project is released under the license specified in `LICENSE`.