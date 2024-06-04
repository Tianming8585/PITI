# Utilizing PITI for Generating Autonomous UAV Images in Natural Environments.

- Competition Name: [Generative AI For Autonomous Uav Navigation In Natural Environments](https://tbrain.trendmicro.com.tw/Competitions/Details/34)
- Team Name: TEAM_5101
- Team Members: [@Tsao666](https://github.com/Tsao666), [me](https://github.com/Tianming8585)
- Final Competition Results:

  | Testing Dataset | FID Score | Rank |
  | :-------------- | :-------- | :--- |
  | Private         | 89.09644  | 4    |
  | Public          | 88.878136 | 6    |

- [Competition Report](./report.md)

- Key Technologies:

  1. [PITI](https://github.com/Tianming8585/PITI) By [me](https://github.com/Tianming8585)
  1. [DP_GAN](https://github.com/Tsao666/DP_GAN) By [@Tsao666](https://github.com/Tsao666)
  1. [Ensemble](<[https://github.com/Tsao666/DP_GAN](https://github.com/Tsao666/DP_GAN/blob/main/ensemble.md)>) By [@Tsao666](https://github.com/Tsao666) and [me](https://github.com/Tianming8585)

---

# Pretraining is All You Need for Image-to-Image Translation

> [Tengfei Wang](https://tengfei-wang.github.io/), [Ting Zhang](https://www.microsoft.com/en-us/research/people/tinzhan/), [Bo Zhang](https://bo-zhang.me/), [Hao Ouyang](https://ken-ouyang.github.io/), [Dong Chen](http://www.dongchen.pro/), [Qifeng Chen](https://cqf.io/), [Fang Wen](https://www.microsoft.com/en-us/research/people/fangwen/)  
> 2022

[paper](https://arxiv.org/abs/2205.12952) | [project website](https://tengfei-wang.github.io/PITI/index.html) | [video]() | [online demo](https://huggingface.co/spaces/tfwang/PITI-Synthesis)

## Introduction

We present a simple and universal framework that brings the power of the pretraining to various image-to-image translation tasks.

Diverse samples synthesized by our approach.  
<img src="figure/diverse.jpg" height="380px"/>

## Set up

### Installation

```
git clone https://github.com/PITI-Synthesis/PITI.git
cd PITI
```

### Environment

```
sudo apt-get update
sudo apt-get install openmpi-bin libopenmpi-dev -y
conda env create -f environment.yml
conda activate PITI
conda install -c conda-forge openmpi -y
pip install mpi4py==3.0.3 dlib==19.22.1
pip install gradio
```

### Pretrained Models

Please download pre-trained models for both `Base` model and `Upsample` model, and put them in `./ckpt`.
| Model | Task | Dataset
| :--- | :---------- | :----------
|[Base-64x64](https://hkustconnect-my.sharepoint.com/:u:/g/personal/tfwang_connect_ust_hk/EVslpwvzHJxFviyd3bw6KSEBWQ9B9Oqd5xUlemo4BNcHpQ?e=F5450q) | Mask-to-Image | Trained on COCO.
|[Upsample-64-256](https://hkustconnect-my.sharepoint.com/:u:/g/personal/tfwang_connect_ust_hk/ERPFM88nCR5Gna_i81cB_X4BgMyvkVE3uMX7R_w-LcSAEQ?e=EmL4fs) | Mask-to-Image | Trained on COCO.
|[Base-64x64](https://hkustconnect-my.sharepoint.com/:u:/g/personal/tfwang_connect_ust_hk/EQsQdJGrxaJDsDYFycIRTO4BNHdEOqZmO_QHSZVV23n5-g?e=I7FSlU) | Sketch-to-Image | Trained on COCO.
|[Upsample-64-256](https://hkustconnect-my.sharepoint.com/:u:/g/personal/tfwang_connect_ust_hk/Ec5DDBQkILpMm5lO0UeytzIBCteefJ_izY9izg7IEHAM8Q?e=6IL7Og)| Sketch-to-Image | Trained on COCO.

If you fail to access to these links, you may alternatively find our pretrained models [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tfwang_connect_ust_hk/Ej0KKEFuje5NnYwaR3wob7YBsca1mBoozuCwCrzc16ra_g?e=COucC2).

## Training

### Preparation

Download the following pretrained models into `./ckpt/`.

| Model                                                                                                                                                  | Task          | Dataset          |
| :----------------------------------------------------------------------------------------------------------------------------------------------------- | :------------ | :--------------- |
| [Base-64x64](https://hkustconnect-my.sharepoint.com/:u:/g/personal/tfwang_connect_ust_hk/EVslpwvzHJxFviyd3bw6KSEBWQ9B9Oqd5xUlemo4BNcHpQ?e=F5450q)      | Mask-to-Image | Trained on COCO. |
| [Upsample-64-256](https://hkustconnect-my.sharepoint.com/:u:/g/personal/tfwang_connect_ust_hk/ERPFM88nCR5Gna_i81cB_X4BgMyvkVE3uMX7R_w-LcSAEQ?e=EmL4fs) | Mask-to-Image | Trained on COCO. |

### Preprocess

Run the notebook `preprocess.ipynb` to preprocess training dataset.

### Start Training

Taking mask-to-image synthesis as an example: (sketch-to-image is the same)

#### Finetune the Base Model

Modify `mask_finetune_base.sh` and run:

```
bash mask_finetune_base.sh
```

### Inference

Run the following notebook `./generate-example.ipynb` to generate output images.

## Citation

If you find this work useful for your research, please cite:

```
@misc{
 title = {Utilizing PITI for Generating Autonomous UAV Images in Natural Environments.},
  author = {Zhe-Yu Guo},
  url={https://github.com/Tianming8585/PITI},
  year = {2024},
}
```

## Acknowledgement

Thanks for [PITI](https://github.com/PITI-Synthesis/PITI) for sharing their code and pretrained models.
