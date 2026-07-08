# MERIT-Multi-domain-Efficient-RAW-Image-Translation

## How To Use

### Setup The Environment

This project requires **Python 3.10.16** and **PyTorch 1.13.1+cu117**.

Install all dependencies using the provided conda environment file:

```bash
conda env create -f environment.yml
conda activate merit
```

### Dataset

Download our dataset MDRAW  [here](https://drive.google.com/file/d/1ToavWhv-Di8Muosmki59xByP0_-H5njT/view?usp=sharing)

### Preprocessing

Run the following scripts in order from the `preprocess/` directory:

```bash
python preprocess/0_paired_process.py
python preprocess/0_unpaired_process.py
python preprocess/1_paired_split.py
python preprocess/1_unpaired_split.py
python preprocess/2_build_noise_profile.py
```

### Training

```bash
python main.py --mode train
```

### Evaluation

```bash
python main.py --mode eval
```
