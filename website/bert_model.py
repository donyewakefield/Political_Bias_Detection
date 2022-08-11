from transformers import TFDistilBertForSequenceClassification, TFBertTokenizer
import tensorflow as tf
import requests
import os
import zipfile

class BertTokenizerClassifierCombined(tf.keras.Model):
    def __init__(self, tokenizer, classifier, output_layer_override=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.classifier = classifier
        if output_layer_override is not None:  # legacy code - prob useless but just in case...
            self.classifier.classifier = output_layer_override  # replace output with custom

    def call(self, data):
        tokenized = self.tokenizer(data)
        del tokenized["token_type_ids"]  # for DistilBERT
        # print(tokenized)
        out = self.classifier(**tokenized)
        return out

    def predict_prod(self, data):
        if isinstance(data, str):
            data = [data]  # model expects an iterable of strings
        pred = tf.nn.softmax(self.predict(data).logits)[0]
        print(list(pred.numpy()))
        return list(pred.numpy())  # do softmax here since its not included in the model
        # this is because the model was trained with loss with logits, so the optimizer expected un-softmaxed outputs

def get_model():
    tf.keras.utils.set_random_seed(100)
    model_name = "distilbert_v1"

    hf_bert_model_path = "distilbert-base-uncased"  # https://huggingface.co/bert-base-uncased  Note that there are other BERT models available.
    tokenizer = TFBertTokenizer.from_pretrained(hf_bert_model_path)  # no TFDistilBertTokenizer, so just use regular bert
    classifier = TFDistilBertForSequenceClassification.from_pretrained(hf_bert_model_path, num_labels=3)
    model = BertTokenizerClassifierCombined(tokenizer, classifier)

    if not os.path.exists(model_name):

        zipped_model = requests.get("https://docs.google.com/uc?export=download&id=1E_ZC-Zmke2HrxeZu24yMTgqn-gQ3cI2K&confirm=t&uuid=68b827df-d38d-4f34-a034-385b88b67a08")
        with open(f"{model_name}.zip", "wb") as zfp:
            zfp.write(zipped_model.content)

        with zipfile.ZipFile(f"{model_name}.zip") as zf:
            os.mkdir(model_name)
            zf.extractall(model_name)

    model.load_weights(f"{model_name}/weights/{model_name}.ckpt")
    # print(f"TESTING LOADED MODEL: {model.predict_prod('Test!')}")  # sanity check
    return model