This is the repository for Streamlit’s FaceGAN demo, built on Shaobo Guan’s TL-GAN project for image synthesis of faces with controllable parameters like gender, age, baldness, skin tone, and smile. 

## 1. Introduction

Progressively Growing Generative Adversarial Networks (PG-GANs) are a cutting edge method for creating random synthetic images. Examples of amazingly realistic faces generated from a PG-GAN trained on celebrities can be seen in NVidia's [PG-GAN demo](https://github.com/tkarras/progressive_growing_of_gans). For each random vector in the generative network's latent space, it outputs a different face. Shaobo Guan's TL-GAN project uses automated face categorizers to identify directions in the latent space that correspond to desired modifications of the generated image, making it older, balder, more female, or dozens of other possible transformations. Streamlit gives us a rapidfire way to explore TL-GAN in an interactive web app. Using `@st-cache` with the `hash_funcs` variable, we can make sure that our TensorFlow session and trained GAN model are preserved each time the app re-runs, making for snappy, responsive execution even when we're hacking away at the code. 


## 2. Instructions for running locally

This demo requires a CUDA-compatible GPU and Python 3.6 (TensorFlow is not yet compatible with Python 3.7 or 3.8). 

### 2.1  Install the code, dependencies, and models

1. Clone this repository: `git clone https://github.com/streamlit/streamlit_tl_gan_demo.git`
2. Create a new Python 3.6 virtual environment using venv, virtualenv, pyenv, or conda
3. From the project’s root directory, install dependencies: `pip install -r requirements.txt`
4. Download Nvidia's [pre-trained pg-GAN model](https://streamlit.io/assets/asset_model.tar.gz) and Shao Bo's [pre-fitted feature model](https://streamlit.io/assets/asset_results.tar.gz)
5. Decompress in the project's root directory: `tar xvzf asset_model.tar.gz; tar xzvf asset_results.tar.gz`

### 2.2 Run the demo using Streamlit
1. From the project's root directory, type `streamlit run src/tl_gan/streamlit_tl_demo.py`

2. Your default web browser should pop up with an interactive app that will let you play with the model.

3. Have fun! Notice that faces you've seen before generate faster, thanks to Streamlit's handy built-in cache. 
