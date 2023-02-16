FROM python:3.9

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get -y install gcc g++ libgl1 libglib2.0-0

RUN pip install shap matplotlib numpy opencv-python scikit-image scikit-learn scipy tqdm nlpaug nltk transformers pytest \
    pytest-lazy-fixture pytest-mock tensorflow tf-explain torch torchvision zennit captum pandas datasets keras_nlp lime

WORKDIR Quantus

CMD ["/bin/sh"]