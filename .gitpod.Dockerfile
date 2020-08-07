FROM gitpod/workspace-full

RUN pyenv install 3.7.7 && \
    pyenv global 3.7.7 && \
    pip install --upgrade pip