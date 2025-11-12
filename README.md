<h2 align="center">Sync4DGS: Dynamic 3D Scene Reconstruction from Any Unsynchronized Multi-View Videos</h2>
<p align="center">
  <a href="https://semyeong-yu.github.io/"><strong>Semyeong Yu</strong></a>
  ·
  <a href="https://www.viclab.kaist.ac.kr/"><strong>Munchurl Kim</strong></a>
  <br>
</p>

<div align='center'>
  <br><img src="img/thumbnail.png" width=70%>
  <!--<img src="img/eigentrajectory-model.svg" width=70%>-->
  <br>A novel view synthesis result of Sync4DGS.
</div>

<br>**Summary**: Sync4DGS with 3D trajectory-driven time alignment

<br>

## Contents

1. [Setup](#-Setup)
2. [Preprocess Datasets](#-Preprocess-Datasets)
3. [Training](#-Training)
4. [Evaluation](#-Evaluation)


## Setup

### Environment Setup

Clone the source code of this repo.
```shell
mkdir sync4dgs
cd sync4dgs
git clone --recursive https://github.com/KAIST-VICLab/Sync4DGS.git .
```

Installation through pip is recommended. First, set up your Python environment:
```shell
conda create -n sync4dgs python=3.9
conda activate sync4dgs
```

Make sure to install CUDA and PyTorch versions that match your CUDA environment. We've tested on RTX 4090 GPU with PyTorch  version 2.1.2.
Please refer https://pytorch.org/ for further information.

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Modify prefix of environment.yaml to your conda environment path.
Then the remaining packages can be installed with:

```shell
pip install --upgrade setuptools cython wheel
pip install -r requirements.txt
conda env update --file environment.yml
```

<!-- Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate Ex4DGS
``` -->

## Preprocess Datasets

For dataset preprocessing, we follow [STG](https://github.com/oppo-us-research/SpacetimeGaussians.git).

### Neural 3D Video Dataset
First, download the dataset from [here](https://github.com/facebookresearch/Neural_3D_Video). You will need colmap environment for preprocess.
To setup dataset preprocessing environment, run scrips:
```shell
./scripts/env_setup.sh
```

To preprocess dataset, run script:
```shell
./scripts/preprocess_all_n3v.sh <path to dataset>
```

### Technicolor dataset
Download the dataset from [here](https://www.interdigital.com/data_sets/light-field-dataset).
To setup dataset preprocessing environment, run scrips:

```shell
./scripts/preprocess_all_techni.sh <path to dataset>
```

Please refer [STG](https://github.com/oppo-us-research/SpacetimeGaussians.git) for further information.

## Training

Run command:
```shell
python train.py --config configs/<some config name>.json --model_path <some output folder>  --source_path <path to dataset>
```

## Evaluation

Run command:
```shell
python render.py --model_path <path to trained model>  --source_path <path to dataset> --skip_train --iteration <trained iter>
```