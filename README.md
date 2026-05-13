# DTEFusion

Created by Xiaocong Wu, Chongqing Key Laboratory of Green Design and Manufacturing of Intelligent Equipment, Chongqing University of Technology and Business, Chongqing, China. 
Contact: 2023311001@ctbu.edu.cn.

## Repository Structure

```text
DTEFusion-main/
├── config/
│   └── model2.yaml          # Main experiment configuration
├── models/
│   ├── models2.py           # DBTFuse1 model definition
│   ├── F_NSWT.py            # NSWT/DCT multi-spectral feature module
│   ├── TCMblock.py          # TCM/Mamba block
│   ├── fusionlayer.py       # Dense fusion layer
│   ├── filters.py           # Gradient and bilateral filters
│   └── loss.py              # Fusion loss
├── dataset.py               # Dataset parsing and preprocessing
├── config.py                # YAML config, model, dataset, optimizer loaders
├── main.py                  # Train, evolve, and test entry point
├── metric.py
├── metric_new.py            # Evaluation metrics used during validation
└── result/
    └── evolve.pth           # Existing checkpoint
```

## Environment

The code is intended for Python 3.10+ and PyTorch. A CUDA GPU is recommended because the model uses `mamba_ssm` and several tensor-heavy image operations.

Install the main dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy pillow matplotlib tqdm pyyaml scipy networkx
pip install pytorch-msssim pywavelets pytorch-wavelets einops thop
pip install mamba-ssm
```

`mamba-ssm` can be sensitive to CUDA, PyTorch, and compiler versions. If installation fails, install the version that matches your CUDA/PyTorch environment from the official `mamba-ssm` instructions.

## Dataset Format

Training uses three paired infrared-visible datasets by default:

- `MSRS`
- `Road`
- `LLVIP`

Each dataset directory must contain two subdirectories named `vi` and `ir`. Image filenames must match between the two folders.

Expected structure:

```text
traindata/
+-- MSRS/
|   +-- vi/
|   |   +-- 0001.png
|   |   +-- ...
|   +-- ir/
|       +-- 0001.png
|       +-- ...
+-- Road/
|   +-- vi/
|   +-- ir/
+-- LLVIP/
    +-- vi/
    +-- ir/
```

Testing supports `MSRS`, `M3FD`, `LLVIP`, and `Road`. For `MSRS`, `M3FD`, `LLVIP`, and `Road` test folders, the script expects:

```text
testdata/
+-- MSRS/
    +-- labels.txt
    +-- vi/
    +-- ir/
```

`labels.txt` should contain one image filename per line, for example:

```text
0001.png
0002.png
```


## Configuration

Edit `config/model2.yaml` before running:

```yaml
dataset:
  ratio: [0.8, 0.2]
  batch_size: 16
  shuffle: true
  url:
    LLVIP: /path/to/train_dataset
```

The loader uses the `MSRS` path if it exists; otherwise it uses the first path under `dataset.url`. Set `model.pretrained` when you want to initialize from an existing checkpoint.

There are still hard-coded output and test paths near the bottom of `main.py`. Update these paths for your machine before running:

```python
config_path = "/path/to/DTEFusion-main/config/model2.yaml"
save_loss_plot_path = "/path/to/output_dir"
config.model.pretrained = "/path/to/checkpoint.pth"
test(config, "/path/to/test_dataset", device)
```

## Training

Run normal training:

```bash
python main.py --train
```

The fixed loss weights used by this mode are defined in the `elif args.train:` block of `main.py`:

```python
fixed_a = 1.44825699
fixed_b = 8.55174301
fixed_c = 3.51766962
fixed_d = 6.48233038
```

Checkpoints are saved by `checkpoint()` in `main.py`. Adjust `folder_path` inside that function if you want outputs in a different location.

## Hyperparameter Search

Run PSO-based loss-weight search followed by final training:

```bash
python main.py --evolve
```

The search space is defined by `HYP_SPACE` in `main.py`. The current PSO settings are:

```python
pop_size = 50
n_iterations = 13
small_epochs = 2
```

These defaults can be expensive. For a quick smoke test, reduce `pop_size`, `n_iterations`, and `small_epochs`.

## Testing

Run inference with a pretrained checkpoint:

```bash
python main.py --test
```

Before testing, update the checkpoint and test dataset paths in the `elif args.test:` block of `main.py`.

The testing flow converts the fused luminance output back to RGB by combining it with the visible image chroma channels in YCbCr space. Results are written to a folder named after the test dataset and checkpoint.

## Citation
If you use this code in your research, please cite the following paper:

```bash
TY  - JOUR
T1  - DTEFusion: a dual-transformation network enhanced fusion method for infrared and visible images with improved mamba
AU  - Wu, Xiaocong
AU  - Feng, Xin
JO  - Infrared Physics & Technology
SP  - 106647
PY  - 2026
DA  - 2026/05/11/
SN  - 1350-4495
DO  - https://doi.org/10.1016/j.infrared.2026.106647
UR  - https://www.sciencedirect.com/science/article/pii/S1350449526002823
KW  - Infrared andvisibleimagefusion
KW  - Multiscalefeatureenhancement
KW  - Token convolutionalMamba
KW  - Adaptiveparametertuning
AB  - The primary goal of image fusion is to generate clearer and more visually perceptible results consistent with human vision. Most existing fusion methods primarily focus on optimizing specific objective metrics, which often results in insufficient representation of the inherent multiscale and multidirectional characteristics of human visual perception. To address this issue, a dual-transformation enhancement network (DTEFusion) is more suited to the human visual system. It is combined with an improved Mamba module for infrared and visible image fusion, mimicking the human visual mechanism by jointly modeling multiscale structural details and cross-modal feature interactions to achieve more perceptually consistent fusion results. Specifically, the proposed frequency channel attention nonsubsampled wavelet transform (F-NSWT)module combines frequency-channel attention with a nonsubsampled wavelet transform to enhance multiscale frequency-domain features in visible images; the FreMLP module strengthens target feature representations in infrared images; and the token-convolution mamba (TCM) module integrates the advantages of CNN and Mamba to effectively capture both local and global dependencies while maintaining computational efficiency. In addition, the cross-model skip-dense connection (C-SDC) fusion layer enables adaptive cross-modal feature interaction. Moreover, we employ a particle swarm optimization (PSO) algorithm to tune the loss function hyperparameters and adaptively optimize our objective metrics. Experimental results demonstrate that our method achieves superior overall performance compared with eleven state-of-the-art fusion algorithms. Our approach attains the highest MS-SSIM scores across four benchmark datasets, indicating closer alignment with human visual perception. The code for this paper will be released at https://github.com/bluemountain123 /DTEFusion.
ER  - 

}
```


