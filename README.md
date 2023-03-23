# Passive Acoustic Monitoring (PAM) Inferences

<div align="center">
<img class="img-fluid" src="assets/frog_shouting,_Henri_Rousseau_painting.png" alt="img-verification" width="250" height="250">
</div>

Repository to solve the species identification problem over passive acoustic monitoring files using machine learning models.


## Usage Instructions:


1. Install [Conda](http://conda.io/)

2. Clone this repository

```bash
git clone https://github.com/soundclim/pam_inferences/
```

3. Create environment and install requirements

```bash
cd anuraset
conda create -n pam_inferences_env python=3.8 -y
conda activate pam_inferences_env
#conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```
4. Upload your data in the `data` folder. The structure of this data folder is agnostic to the project. In practice we will search for any .wav file, save its path, extract the metadata, and apply the machine learning model. One example of a data folder structure in a real-world project is the next one:

<div align="center">
<img class="img-fluid" src="assets/orleans_data_structure.png" alt="img-verification" width="250" height="350">
</div>


5. Select the parameters as sliding window, window size, and the trained model using .yaml file. Run the inferences 

```bash
python inferences.py --config configs/exp_resnet18.yaml
```

6. Visualize results in `resutls_analysis.ipynb` notebook



## Inference pipeline



## License (TODO)

...


## Citing this work (TODO)

..-