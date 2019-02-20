# SubSpectralNet-PyTorch

This repository contains the PyTorch Implementation of SubSpectralNets introduced in the following paper:

[SubSpectralNet - Using Sub-Spectrogram based Convolutional Neural Networks for Acoustic Scene Classification](https://arxiv.org/abs/1810.12642) (Accepted in ICASSP 2019) 

[Sai Samarth R Phaye](https://ssrp.github.io), [Emmanouil Benetos](http://www.eecs.qmul.ac.uk/~emmanouilb/), and [Ye Wang](https://www.smcnus.org/profile/ye-wang/).

[Click here for the presentation!](https://docs.google.com/presentation/d/1xyvpgGPkdrxgbBbEWvup5sPiajiWRdbQ7CZGd9nW0jY/)

We introduce a novel approach of using spectrograms in Convolutional Neural Networks in the context of acoustic scene classification. First, we show from the statistical analysis that some specific bands of mel-spectrograms carry discriminative information than other bands, which is specific to every soundscape. From the inferences taken by this, we propose SubSpectralNets in which we first design a new convolutional layer that splits the time-frequency features into sub-spectrograms, then merges the band-level features on a later stage for the global classification. The effectiveness of SubSpectralNet is demonstrated by a relative improvement of +14% accuracy over the DCASE 2018 baseline model. The detailed architecture of SubSpectralNet is shown below.

<p align = "center">
<img src="https://raw.githubusercontent.com/ssrp/SubSpectralNet/master/figures/SubSpectralNet.png" width="600">
</p>
                                                 
If you have any queries regarding the code, please contact us on the following email: phaye.samarth@gmail.com (Sai Samarth R Phaye)

## Usage

Following are the steps to to follow for using this implementation:

Before anything, it is expected that you download and extract the [DCASE 2018 ASC Development Dataset](https://zenodo.org/record/1228142). You should obtain a folder named "TUT-urban-acoustic-scenes-2018-development", which contains various folders like "audio", "evaluation_setup" etc. Once you get this folder, following are the steps to execute the code:

**Step 1. Clone the repository to local.**
Download the repository:
```
git clone https://github.com/ssrp/SubSpectralNet-PyTorch.git SubSpectralNets
cd SubSpectralNets/code
```

**Step 2. Install the prerequisites.**
```
pip install -r requirements.txt
```

**Step 3. Train a SubSpectralNet**  
Train with default settings:
```
python main.py --root-dir <location_of_dataset>/TUT-urban-acoustic-scenes-2018-development/
```
For more settings, the code is well-commented and it's easy to change the parameters looking at the comments. 

## Results
The results of the network are shared in Section 4 of the paper. Following is the accuracy curve obtained after training the model on 40 mel-bin magnitude spectrograms, 20 sub-spectrogram size and 10 mel-bin hop-size (72.18%, average-best accuracy in three runs).

<p align = "center">
<img src="https://raw.githubusercontent.com/ssrp/SubSpectralNet/master/figures/AccPlot.png" width="480">
</p>
