FROM tensorflow/tensorflow:latest-py3-jupyter

RUN pip install tensorflow==2.0.0
RUN pip install matplotlib

RUN echo "if [ "\$PS1" ]; then PS1='\\h:\\w\\$ '; fi" > /etc/bash.bashrc
