# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# # Sentiment model
# sent_model = "distilbert-base-uncased-finetuned-sst-2-english"
# AutoModelForSequenceClassification.from_pretrained(sent_model).save_pretrained("./models/sentiment")
# AutoTokenizer.from_pretrained(sent_model).save_pretrained("./models/sentiment")

# # Toxicity model
# tox_model = "unitary/toxic-bert"
# AutoModelForSequenceClassification.from_pretrained(tox_model).save_pretrained("./models/toxic")
# AutoTokenizer.from_pretrained(tox_model).save_pretrained("./models/toxic")


from transformers import pipeline
pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
pipeline("text-classification", model="unitary/toxic-bert")
