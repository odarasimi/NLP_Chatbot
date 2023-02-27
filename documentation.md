# Subject 
This project is the implementation of an NLP chatbot answering questions about science

## Implementing an NLP based chatbot using extractive QA system

The purpose of this project is to create an NLP-based chatbot that uses an extractive QA system to provide answers to questions. The chatbot is designed to handle statements in multiple languages, determine the intent of the statement and provide a response based on the intent.

## Language Detection and Translation

When a statement is entered into the chatbot, the language is first detected. If the statement is in a language other than English, the statement is translated into English before being processed. This allows the chatbot to operate more efficiently and accurately, as it is trained to understand and respond in English. The translation is performed using the `googletrans` library.

## Statement Intent Determination

After the language detection and translation, the statement intent is determined. The chatbot can recognize and respond to greetings, goodbyes, and questions. For greetings and goodbyes, the chatbot provides a random response based on a set of pre-defined rules. For questions, the statement is passed on to the extractive QA system.

## Extractive QA System

The extractive QA system is trained using a model that is optimized to automatically identify and extract answers from a provided database. In this case, the model is a pretrained `deepset/tinyroberta-squad2` model, which was fine-tuned and trained on a science-based dataset called "Sciq". The model takes in a 'question-context' input and is labeled by an answer. The context serves as the document from which the answer is to be automatically extracted.

The "Sciq" dataset is combined with the quartz and openbookqa science based datasets (found on Huggingface hub) by merging the provided contexts into a large single context. Due to the large size of the context, the NLP framework `Haystack` is used to do the question/answering at scale. The Haystack framework retrieves the document and reads the data based on the model that was trained earlier. Once the system provides an answer, the response is translated back to the input language if required using the `googletrans` library.

* Due to the large size of the model, it was trained and uploaded to the Huggingface hub from where it will be called. However, the training script can be found in the folder.
* The corpus used for our language context is also provided in the folder, but the code to generate the data is present in the folder.

## Algorithm Stages

1. Enter statement to chatbot
2. Determine language of statement and translate to English if necessary
3. Determine intent of statement (greeting, goodbye, or question)
4. If greeting or goodbye, provide random pre-defined response
5. If question, pass the statement to the extractive QA system
6. Retrieve document and read data based on the model trained on "Sciq", "Openbookqa", "quartz" datasets
7. Provide answer and translate back to the input language if required

### Required Installations
* googletrans==3.1.0a0
* nltk
* datasets
* transformers
* spacy
* tensorflow
* farm-haystack

### Files

* chat - Folder containing all custom files except for documentation.md
* nlp - Subfolder containing the English language model for the spaCy library, used to determine phrase similarity
* sci_wiki.txt - Document used as a large corpus for our QA context, the chatbot performance can be improved by increasing the size of this reference document
* train_model.py - Script used to train the model used for inference
* documentation.md - Documentation
* predictor.py - Script for intent inference and answer prediction
* utils.py - Provides general utility functions

