from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)

@app.route("/")
def main_page():
    return render_template(
        "index.html"
    )


############################# Integrate the model ##################################
import tensorflow as tf
import zipfile
import os
import trafilatura
import http
import requests
import bert_model

# extract weights
if not os.path.exists("C:\\Users\\Donyewakefield\\Desktop\\FMakers_WebApp\\distilbert_v1.zip"):
  zipped_model = requests.get("https://docs.google.com/uc?export=download&id=1E_ZC-Zmke2HrxeZu24yMTgqn-gQ3cI2K&confirm=t&uuid=68b827df-d38d-4f34-a034-385b88b67a08")

  model_name = "distilbert_v1"
  with open(f"{model_name}.zip", "wb") as zfp:
      zfp.write(zipped_model.content)

  with zipfile.ZipFile("C:\\Users\\Donyewakefield\\Desktop\\FMakers_WebApp\\distilbert_v1.zip") as zf:
      os.mkdir("distilbert_v1")
      zf.extractall("distilbert_v1")


# Get started on preprocessing 
# from transformers import TFBertTokenizer, TFBertForSequenceClassification, TFDistilBertForSequenceClassification
# hf_bert_model_path = "distilbert-base-uncased" 
# tokenizer = TFBertTokenizer.from_pretrained(hf_bert_model_path)  # no TFDistilBertTokenizer, so just use regular bert
# classifier = TFDistilBertForSequenceClassification.from_pretrained(hf_bert_model_path, num_labels=3)

# class BertTokenizerClassifierCombined(tf.keras.Model):
#   def __init__(self, tokenizer, classifier, output_layer_override=None, *args, **kwargs):
#     super().__init__(*args, **kwargs)
#     self.tokenizer = tokenizer
#     self.classifier = classifier
#     if output_layer_override is not None:
#       self.classifier.classifier = output_layer_override  # replace output with custom
  
#   def call(self, data):
#     tokenized = self.tokenizer(data)
#     del tokenized["token_type_ids"]  # for DistilBERT
#     print(tokenized)
#     out = self.classifier(**tokenized)
#     print(out)
#     return out

# model = BertTokenizerClassifierCombined(tokenizer, classifier)
# model.load_weights("distilbert_v1/weights/distilbert_v1.ckpt")
# model(["Hello World!"])
# model.summary()

# def run_model(url):
#   text = trafilatura.extract(requests.get(url).text)
#   logits = model.predict([text]).logits
#   print(tf.nn.softmax(logits[0]))  # [democrat score, republican score, center score]
  
# tf.keras.utils.set_random_seed(100)

# get the BERT model
bert = bert_model.get_model()

# This function will run the mode on given user data
def run_predict(form_data):
    text =  form_data["ENTER_TEXT"]
    url = form_data["ENTER_URL"]
    print(form_data)
    if url != "":
        url = f"http://{url}" if not url.startswith("http") else url
        text = trafilatura.extract(requests.get(url).text)  # todo: deal with invalid urls & url errors
    return bert.predict_prod(text)
        
# take in data and run model
@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /results is accessed directly."
    if request.method == 'POST':
        form_data = request.form
        result = run_predict(form_data)

      # Have output be geared toward a winner in the prediction
        if result[0] > result[1] and result[0] > result[2]:
          winner = result[0]
          winner_name = "This is a left bias article!"
        if result[1] > result[0] and result[1] > result[2]:
          winner = result[1]
          winner_name = "This is a right bias article!"
        if result[2] > result[0] and result[2] > result[1]:
          winner = result[2]
          winner_name = "This is a center article!"

      # Return the predicted results to the user
        return render_template('submit.html', result = result, winner = winner_name)


