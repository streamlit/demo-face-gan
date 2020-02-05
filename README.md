# Streamlit Demo: The Controllable GAN Face Generator
This project highlights Streamlit's new `hash_func` feature with an app that calls on TensorFlow to generate photorealistic faces, using Nvidia's [Progressive Growing of GANs](https://research.nvidia.com/publication/2017-10_Progressive-Growing-of) and Shaobo Guan's [Transparent Latent-space GAN](https://blog.insightdatascience.com/generating-custom-photo-realistic-faces-using-ai-d170b1b59255) method for tuning the output face's characteristics. For more information, check out the [tutorial on Towards Data Science](https://towardsdatascience.com/building-machine-learning-apps-with-streamlit-667cef3ff509). 

The Streamlit app is [implemented in only 150 lines of Python](https://github.com/streamlit/demo-face-gan/blob/master/app.py) and demonstrates the wide new range of objects that can be used safely and efficiently in Streamlit apps with `hash_func`. 

![In-use Animation](https://raw.githubusercontent.com/streamlist/demo-face-gan/master/demo.gif "In-use Animation")

## How to run this demo
The demo requires a CUDA-compatible GPU and Python 3.6 (TensorFlow is not yet compatible with later versions). We suggest creating a new virtual Python 3.6 environment, then running:

```
pip install --upgrade streamlit==0.53 tensorflow==1.13.1 tensorflow-gpu==1.13.1
streamlit run https://raw.githubusercontent.com/streamlit/demo-face-gan/master/app.py
```

### Questions? Comments?

Please ask in the [Streamlit community](https://discuss.streamlit.io).
