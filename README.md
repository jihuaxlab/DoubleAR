# Intelligent sensory of oil quality by adaptive residual attention networks and Raman spectroscopy

Code and data of our paper [Intelligent sensory of oil quality by adaptive residual attention networks and Raman spectroscopy](https://doi.org/10.1016/j.microc.2025.112680).

## Requirements

The code has been tested running under Python 3.7.4, with the following packages and their dependencies installed:
```
numpy==1.16.5
matplotlib==3.1.0
sklearn==0.21.3
seaborn==0.11.2
pytorch==1.7.1
```

## Usage

### Oil adulteration prediction

```
python main.py --c 3
```

### Oil storage time prediction

```
python main.py --c 4
```

## Options

We adopt an argument parser by package  `argparse` in Python, and the options for running code are defined as follow:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of representations')
parser.add_argument('--c', type=int, default=4,
                    help='Num of classes')
parser.add_argument('--d', type=int, default=700,
                    help='Num of spectra dimension')               

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
```

## Citation

```
@article{shi2025doublear,
    title = {Intelligent sensory of lard quality by adaptive residual attention networks and Raman spectroscopy},
    journal = {Microchem. J.},
    volume = {209},
    pages = {112680},
    year = {2025},
    author = {Zhuangwei Shi and Yunhao Su and Jianchen Zi and Shibiao Yang and Dongsheng Li and Yongkun Luo and Chenhui Wang and Hai Bi},
}
```
