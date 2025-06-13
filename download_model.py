# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# # Sentiment model
# sent_model = "distilbert-base-uncased-finetuned-sst-2-english"
# AutoModelForSequenceClassification.from_pretrained(sent_model).save_pretrained("./models/sentiment")
# AutoTokenizer.from_pretrained(sent_model).save_pretrained("./models/sentiment")

# # Toxicity model
# tox_model = "unitary/toxic-bert"
# AutoModelForSequenceClassification.from_pretrained(tox_model).save_pretrained("./models/toxic")
# AutoTokenizer.from_pretrained(tox_model).save_pretrained("./models/toxic")


# from transformers import pipeline
# pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# pipeline("text-classification", model="unitary/toxic-bert")


from transformers import pipeline

def preload_all_models():
    print("Downloading sentiment-analysis model...")
    pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    print("Downloading emotion detection model...")
    pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion")

    print("Downloading toxicity detection model...")
    pipeline("text-classification", model="unitary/toxic-bert")

    print("Downloading spam detection model...")
    pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

    print("Downloading topic classification model...")
    pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    print("Downloading summarization model...")
    pipeline("summarization", model="google/pegasus-xsum")  # Use smaller summarizer

if __name__ == "__main__":
    preload_all_models()
