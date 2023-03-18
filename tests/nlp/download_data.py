from transformers import TFAutoModelForSequenceClassification, AutoModelForSequenceClassification
from datasets import load_dataset


def main():
    load_dataset("sst2")
    TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    AutoModelForSequenceClassification.from_pretrained("gchhablani/fnet-base-finetuned-sst2")


if __name__ == "__main__":
    main()
