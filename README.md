# PolitiParser

By the Nerdy Neurons (SureStart MIT FutureMakers 2022 Team 13)



# Our Mission:

In an increasingly polarized world, we aim to encourage interaction between various viewpoints and ideologies.

# Here's how we do it:

## The Model:

We fine-tuned [HuggingFace's DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert), a knowledge-distilled version of [Google's original BERT implementation](https://arxiv.org/abs/1810.04805), with a 3-class classifier with softmax activations for classification. It was trained for 50 epochs of 100 steps each (totalling 6 passes through our 2700-sample dataset) using the `Adam` optimizer (with a learning rate of `5e-5`) 

Our dataset was derived from [Quantifying News Media Bias through Crowdsourcing and Machine Learning Dataset ](https://deepblue.lib.umich.edu/data/concern/data_sets/8w32r569d?locale=en#read_me_display) - we scraped the provided URLs for the articles' raw text, narrowed down the original 25 possible classes to 3, and balanced the dataset to include ~900 of each class (~2700 total samples). 

See `dl_model/PoliticsNLPBERT_Final.ipynb` for the full process and `dl_model/PoliticsNLPBERT_Webscraping.ipynb` for our dataset methodology.

The dataset is available [here](https://docs.google.com/uc?export=download&id=1H-IMIUDSM7Y-jPjo8skbcxNBBGovbiLL), and the model weights [here](https://docs.google.com/uc?export=download&id=1-0PA2XTdJhZxDvqSf5xj1TaxcQpM-U2z&confirm=t&uuid=68b827df-d38d-4f34-a034-385b88b67a08). (Note that this is only the model's weights - the model must first be defined and initialized as seen in the notebook).

## The Website:

An online portal to our model, which allows users to submit text and article URLs to have bias detected in. Built with Flask, JS, HTML, and CSS.

After cloning the repository and installing dependencies, it can be run (for development/testing - we do hope to get a webserver and domain someday) with:

```
cd website
python app.py
```
![250968030-999076b9-8a70-461d-8571-a10bb2e9e3b2](https://github.com/donyewakefield/Political_Bias_Detection/assets/71467135/f120d29f-3210-43ce-b8bc-43f1b725d79e)


## The Future:

We plan on implementing a browser extension which will improve the usability of our product. Furthermore, we plan on implementing an objectivity score so that users can have an understanding on if the media they are consuming is objective or subjective
In the long term, we hope to automate updates to our dataset by implementing a crowdsourcing model. In addition, we aim to implement a recommendation system that suggests relevant content on the other side of the political spectrum, so they gain a more holistic understanding of the topic at hand.

# Our Team:

![The Nerdy Neurons](https://docs.google.com/uc?export=download&id=1min006a_qcEcw7PrJApwK6vYJti24E-i)

## Thanks to our mentors:

## Shwetha Tinnium Raju

<img src="https://docs.google.com/uc?export=download&id=18JOi8veL4OLi7cThmUSigKbZjtbebL7-" alt="Shwetha Tinnium Raju" width="300"/>

## and Michalis Koumpanakis,

<img src="https://docs.google.com/uc?export=download&id=1SAquhvcEcCUYSIERaDD1c-6_wSZgZvOR" alt="Michalis Koumpanakis" width="300">

## as well as the SureStart team for making this all possible.
