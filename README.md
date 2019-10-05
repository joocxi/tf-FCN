# Fully Convolutional Networks for Liver Segmentation

In this project, we will experiment with training FCN models on a liver segmentation dataset provided by [IRCAD](https://www.ircad.fr/research/computer/)

## Set up
```bash
pip install virtualenv
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```
## Download data
```bash
sh download.sh
```

## Running
### Preprocessing
```bash
python run.py --mode preprocess
```
### Training
```bash

```

## Results

## TODOs
- [x] Preprocessing
- [ ] Building model
- [ ] Training pipeline
- [ ] Augmentation
- [ ] Visualization

**References**
1.  Long et al. , Fully Convolutional Networks for Semantic Segmentation. ([arxiv](https://arxiv.org/pdf/1411.4038.pdf))