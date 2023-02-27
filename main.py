import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time

import chat.predictor as predictor
from chat.predictor import chatbot, pipeline
from chat.utils import detect_language, translate_text


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    try:
        from haystack.document_stores import InMemoryDocumentStore
        import os
        from haystack.pipelines.standard_pipelines import TextIndexingPipeline
        from haystack.nodes import BM25Retriever, FARMReader
        from haystack.pipelines import ExtractiveQAPipeline
        import random
        import nltk
        from nltk.tokenize import word_tokenize
        import googletrans
    except ImportError as e:
        warnings.warn(f"Import error: {str(e)}")


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:

    # Return Processing Pipeline
    file_name = "sci_wiki.txt"
    cwd = (os.path.dirname(os.path.abspath(__file__)))
    doc_dir = os.path.join(cwd, file_name)
    pipe = pipeline(doc_dir)
    
    output = []
    for text in request.text:
        # TODO Add code here

        # Detect language, and translate if not in English
        language = detect_language(text)
        if language == "en":
            checked_text = text
        else:
            checked_text = translate_text(text, lang="en")

        # Determine input intent
        classification = predictor.classify_sentence(checked_text)
        intent = predictor.response_type(classification)
        # If input is a goodbye or question, respond with fixed random phrases, 
        # else predict answer
        if intent != "question":
            answer = intent
        else:
            try:
                answer = chatbot(pipe, text) 
            except:
                answer = "Sorry, I can only aspire to be ChatGPT, I do not have an answer to this question."
        
        # If original questioning language was in English respond as is
        # Else translate response back to original language
        if language == "en":
            response = answer
        else:
            response = translate_text(answer, lang=language)
        output.append(response)

    return SimpleText(dict(text=output))
