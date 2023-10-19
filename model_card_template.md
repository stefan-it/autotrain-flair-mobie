---
language: de
license: mit
tags:
- flair
- token-classification
- sequence-tagger-model
base_model: {{ base_model }}
widget:
- text: {{ widget_text }}
---

# Fine-tuned Flair Model on German MobIE Dataset with AutoTrain

This Flair model was fine-tuned on the
[German MobIE](https://aclanthology.org/2021.konvens-1.22/)
NER Dataset using {{ base_model_short }} as backbone LM and the 🚀 [AutoTrain](https://github.com/huggingface/autotrain-advanced)
library.

## Dataset

The [German MobIE](https://github.com/DFKI-NLP/MobIE) dataset is a German-language dataset, which is human-annotated
with 20 coarse- and fine-grained entity types and entity linking information for geographically linkable entities. The
dataset consists of 3,232 social media texts and traffic reports with 91K tokens, and contains 20.5K annotated
entities, 13.1K of which are linked to a knowledge base.

The following named entities are annotated:

* `location-stop`
* `trigger`
* `organization-company`
* `location-city`
* `location`
* `event-cause`
* `location-street`
* `time`
* `date`
* `number`
* `duration`
* `organization`
* `person`
* `set`
* `distance`
* `disaster-type`
* `money`
* `org-position`
* `percent`

## Fine-Tuning

The latest [Flair version](https://github.com/flairNLP/flair/tree/42ea3f6854eba04387c38045f160c18bdaac07dc) is used for
fine-tuning. Additionally, the model is trained with the
[FLERT (Schweter and Akbik (2020)](https://arxiv.org/abs/2011.06993) approach, because the MobIE dataset thankfully
comes with document boundary information marker.

A hyper-parameter search over the following parameters with 5 different seeds per configuration is performed:

* Batch Sizes: {{ batch_sizes }}
* Learning Rates: {{ learning_rates }}

All models are trained with the awesome [AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced) from
Hugging Face. More details can be found in this [repository](https://github.com/stefan-it/autotrain-flair-mobie).

## Results

A hyper-parameter search with 5 different seeds per configuration is performed and micro F1-score on development set
is reported:

{{ results }}

The result in bold shows the performance of this model.

Additionally, the Flair [training log](training.log) and [TensorBoard logs](tensorboard) are also uploaded to the model
hub.
