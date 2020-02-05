import numpy as np
import os
import pickle
import streamlit as st
import sys
import tensorflow as tf
import urllib

sys.path.append('tl_gan')
sys.path.append('pg_gan')
import feature_axis
import tfutil

def main():
    st.title("Streamlit Face-GAN Demo")
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)
    tl_gan_model, feature_names = load_tl_gan_model()
    features = get_random_features(feature_names)
    session, pg_gan_model = load_pg_gan_model()

    st.sidebar.title('Features')
    features['Young'] = st.sidebar.slider('Young', 0, 100, 50, 5)
    features['Male'] = st.sidebar.slider('Male', 0, 100, 50, 5)
    features['Smiling'] = st.sidebar.slider('Smiling', 0, 100, 50, 5)

    image_out = generate_image(session, pg_gan_model, tl_gan_model,
            features, feature_names)

    st.image(image_out, width=400)

def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

@st.cache(allow_output_mutation=True)
def load_pg_gan_model():
    """
    Create the tensorflow session.
    """
    config = tf.ConfigProto(allow_soft_placement=True)
    session = tf.Session(config=config)

    with session.as_default():
        with open(MODEL_FILE, 'rb') as f:
            G, D, Gs = pickle.load(f)
    return session, Gs

@st.cache
def load_tl_gan_model():
    """
    Load the linear model (matrix) which maps the feature space
    to the GAN's latent space.
    """
    with open(FEATURE_DIRECTION_FILE, 'rb') as f:
        feature_direction_name = pickle.load(f)

    feature_direction = feature_direction_name['direction']
    feature_names = feature_direction_name['name']
    num_feature = feature_direction.shape[1]
    feature_lock_status = np.zeros(num_feature).astype('bool')
    feature_direction_disentangled = \
        feature_axis.disentangle_feature_axis_by_idx(
            feature_direction,
            idx_base=np.flatnonzero(feature_lock_status))
    return feature_direction_disentangled, feature_names

@st.cache(allow_output_mutation=True)
def get_random_features(feature_names):
    """
    Return a random dictionary from feature names to feature
    values within the range [40,60] (out of [0,100]).
    """
    features = dict((name, 40+np.random.randint(0,21)) for name in feature_names)
    return features

@st.cache(hash_funcs={tf.Session : id, tfutil.Network : id}, show_spinner=False)
def generate_image(session, pg_gan_model, tl_gan_model, features, feature_names):
    """
    Converts a feature vector into an image.
    """
    # Create rescaled feature vector
    feature_values = np.array([features[name] for name in feature_names])
    feature_values = (feature_values - 50) / 250
    # Multiply by Shaobo's matrix to get the latent variables
    latents = np.dot(tl_gan_model, feature_values)
    latents = latents.reshape(1, -1)
    dummies = np.zeros([1] + pg_gan_model.input_shapes[1][1:])
    # Feed the latent vector to the GAN in TensorFlow
    with session.as_default():
        images = pg_gan_model.run(latents, dummies)
    # Rescale and reorient the GAN's output to make an image
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0),
                              0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
    return images[0]

FEATURE_DIRECTION_FILE = "feature_direction_2018102_044444.pkl"
MODEL_FILE = "karras2018iclr-celebahq-1024x1024.pkl"
EXTERNAL_DEPENDENCIES = {
    "feature_direction_2018102_044444.pkl" : {
        "url": "https://www.dropbox.com/sh/y1ryg8iq1erfcsr/AADZVwMYXdX88cyBDkx85WdHa/asset_results/pg_gan_celeba_feature_direction_40/feature_direction_20181002_044444.pkl?dl=1",
        "size": 164742
    },
    "karras2018iclr-celebahq-1024x1024.pkl": {
#        "url": "https://drive.google.com/open?id=188K19ucknC6wg1R6jbuPEhTq9zoufOx4",
        "url": "https://streamlit-demo-data.s3-us-west-2.amazonaws.com/facegan/karras2018iclr-celebahq-1024x1024.pkl",
        "size": 277043647
    }
}

if __name__ == "__main__":
    main()
