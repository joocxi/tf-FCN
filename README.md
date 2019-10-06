# Fully Convolutional Networks for Liver Segmentation

In this project, we experiment with training FCN models on a liver segmentation dataset provided by [IRCAD](https://www.ircad.fr/research/computer/)

## Set up
```bash
pip install virtualenv
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```
## Download data
```bash
sh scripts/download.sh
```

## Running
### Preprocessing
```bash
python run.py --mode preprocess
```
We use TensorFlow iterator to iterate over the dataset. To check `image` and `mask` shape, run the command below
```bash
python run.py --mode iter
```
### Training
```bash
python run.py --mode train
```

## Results

## TODOs
- [x] Preprocessing
- [x] Building model
- [x] Training pipeline
- [ ] Augmentation
- [ ] Visualization
- [ ] TensorBoard

**References**
1.  Long et al. , Fully Convolutional Networks for Semantic Segmentation. ([arxiv](https://arxiv.org/pdf/1411.4038.pdf))