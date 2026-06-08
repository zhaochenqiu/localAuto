Markdown
# Local Autoregression with Finite-Support Random Variables for Image Generation

This repository contains the official PyTorch implementation of our paper **"Local Autoregression with Finite-Support Random Variables for Image Generation"** (ICPR 2026).

---

## Citation

If you find our code or paper useful for your research, please consider citing our work using the following BibTeX entry:

```bibtex
@inproceedings{ICPR_2026_zhao,
  author    = {Zhao, Chenqiu and Basu, Anup},
  title     = {Local Autoregression with Finite-Support Random Variables for Image Generation},
  booktitle = {International Conference on Pattern Recognition (ICPR)}, 
  year      = {2026}
}
```


## Pipeline & Usage
### Step 1: Dataset Preparation
Download, preprocess, and normalize the CIFAR-10 dataset to $[-1, 1]$. The processed data will be saved locally.
```
Bash 
python main.py --dataset 1
```


### Step 2: Train Autoencoder
Train the autoencoder model using the preprocessed dataset:
```
Bash
python main.py --autoenc
```


### Step 3: Extract Latent Representations
Generate or extract the latent representations/embeddings from the trained autoencoder:
```
Bash
python main.py --latent
```



### Step 3: Image Generation
Generate synthetic images for evaluation:
```
Bash
python main.py --generation
```

Runtime Notice: Generating a sufficient number of images to calculate the FID (Fréchet Inception Distance) score against the original CIFAR-10 dataset takes approximately 2 hours on a single NVIDIA RTX 4090 GPU.
