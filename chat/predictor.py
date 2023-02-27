'''The code below contains implementation for intent classification and answer prediction.'''

from haystack.document_stores import InMemoryDocumentStore
import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
import random
import nltk
from nltk.tokenize import word_tokenize

#-------------------------------------------------------------------------
# Intent classification

def classify_sentence(sentence):
    nltk.download('punkt')
    # Define greeting, goodbye, and question keywords
    greeting_keywords = ["hi", "hello", "hey", "greetings"]
    goodbye_keywords = ["bye", "goodbye", "farewell", "ciao"]
    question_keywords = ["what", "when", "where", "which", "who", "whom", "whose", "why", "how", "?"]
    # Tokenize the sentence
    words = word_tokenize(sentence.lower())

    # Check if the sentence is a greeting
    if any(word in greeting_keywords for word in words):
        return "greeting"

    # Check if the sentence is a goodbye
    elif any(word in goodbye_keywords for word in words):
        return "goodbye"

    # Check if the sentence is a question
    elif words[-1] == '?' or any(word in question_keywords for word in words[:-1]):
        return "question"

    # If none of the above, return None
    else:
        return None
    
#------------------------------------------------------------------------

def response_type(classification):
    # list of fixed responses
    greeting_response = ["Hi! How can I help you today?", "Hello!", "Hi!", "Hallo!", "Hallo! I know science, expecting your questions!", "Greetings! What question can I answer today?"]
    goodbye_response = ["Bye, have a great time!", "Thank you, bye!", "Cheers!", "Later, bye!", "Ciao!"]

    if classification == 'greeting':
        response = random.choice(greeting_response)
    elif classification == "goodbye":
        response = random.choice(goodbye_response)
    elif classification == "question":
        response = "question"
    else:
        response = "Could you please rephrase your question?"
    return response

#-----------------------------------------------------------------------

# # filename = "sci_wiki.txt"
# # cwd = (os.path.dirname(os.path.abspath(__file__)))
# # doc_dir = os.path.join(cwd, filename)

def pipeline(dir):
    document_store = InMemoryDocumentStore(use_bm25=True)

    files_to_index = [dir]
    indexing_pipeline = TextIndexingPipeline(document_store)
    indexing_pipeline.run_batch(file_paths=files_to_index)

    retriever = BM25Retriever(document_store=document_store)

    reader = FARMReader(model_name_or_path="anuoluwa/scincequest_model", use_gpu=False)

    pipe = ExtractiveQAPipeline(reader, retriever)

    return pipe

def chatbot(pipeline, query):
    prediction = pipeline.run(
        query=query,
        params={
            "Retriever": {"top_k": 10},
            "Reader": {"top_k": 5}
        }
    )

    checklist = []
    for i in range (len(prediction['answers'])):
        checklist.append(prediction['answers'][i].answer)
    resp = max(checklist, key=len)

    return resp



