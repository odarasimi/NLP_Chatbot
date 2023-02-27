''' Majorly generic utility functions.'''
from googletrans import Translator
from datasets import load_dataset, DatasetDict, Dataset
import spacy
import os

def detect_language(text):
    '''Identify the input language.
    Args:
    - text (str): Input text to detect language

    Returns:
    - str: Language code detected for the input text.
    '''
    translator = Translator()
    translation = translator.detect(text)
    return str(translation.lang)

def translate_text(word, lang="en"):
    '''Translate user input to selected language.'''
    translator = Translator()
    if lang == "en":
        translation = translator.translate(word, dest=lang)
    else:
        try:
            translation = translator.translate(word, dest=lang)
        except:
            translation = translator.translate(word, dest="en")
    return str(translation.text)

def count_words(sentence):
    length = len(str(sentence).split())
    return length

def find_index(sentence, letters):
    '''Find index of letters in sentence, used to identify the answer_start
    in our dataset.

    Args:
    - sentence (str): Input sentence to search.
    - letters (str): Letters to be searched in the sentence.

    Returns:
    - int: The index of the first occurrence of the letters in the sentence. If the letters are not found, it returns -1.
    '''
    sentence = sentence.lower()
    letters = letters.lower()
    idx = sentence.find(letters)
    return idx

def find_most_similar_phrase(sentence, target_phrase):
    '''Find most similar phrase in circumstances where answer is not in context.

    Args:
    - sentence (str): The sentence to search for the most similar phrase.
    - target_phrase (str): The target phrase to be compared against.

    Returns:
    - str: The most similar phrase to the target phrase that is found in the input sentence or None
    '''
    # Load English language model for the spaCy library
    spacy_nlp = spacy.load('/chat/nlp')

    sentence_doc = spacy_nlp(sentence)
    target_doc = spacy_nlp(target_phrase)
    # Initializes two variables to keep track of the maximum similarity score and the corresponding phrase
    max_similarity = -1
    most_similar_phrase = None
    # Loops through each noun chunk in the sentence
    for span in sentence_doc.noun_chunks:
        similarity = span.similarity(target_doc)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_phrase = span.text
    return most_similar_phrase

def rename_columns(dataset, list_of_columns):
    '''Rename columns for Dataset object.
    Args:
    - dataset (DatasetDict): The dataset to rename the columns in.
    - list_of_columns (List[str]): List of column names to be renamed.

    Returns:
    - None: The function modifies the input dataset in-place and does not return anything.
    '''
    for name in list_of_columns:
        try:
            main = name.split('.')[0]
            sub = name.split('.')[1]
            dataset[main] = dataset[main].rename_column(name, sub)
        except ValueError:
            pass

def create_corpus(loaded_dataset, context="support", file_name=None):
    '''Extract contexts from the dataset and merge into a large corpus.
    Args:
    - loaded_dataset (Dataset): the dataset to extract the contexts from.
    - context (str): the name of the column in the dataset containing the contexts.
    - file_name (str, optional): if provided, the resulting corpus will be written to a file with this name.

    Returns:
    - corpus (str): the merged corpus of all contexts.
    '''
    contexts = []
    for example in loaded_dataset:
        for value in loaded_dataset[example]:
            contexts.append(value[context])

    # Combine contexts into one big corpus
    corpus = "\n".join(contexts)

    if file_name:
        # Write corpus to text file
        with open(f"{file_name}.txt", "w", encoding="utf-8") as f:
            f.write(corpus)
    else:
        return corpus

# Remember to replace the 3 train,test,validation updates made below with this function.
def update_sample(dataset):
    '''Update the DatasetDict object by adding an answer_start and formatting the answer if not found in context,
    the DatasetDict object is immutable, so we create a new list of its contents.

    Args:
    - dataset (DatasetDict): the dataset to update.

    Returns:
    - new_examples (list): 
    '''
    new_examples = [] # Create an empty list to hold the new examples
    for example in dataset:
        # Extract the 'support' and 'correct_answer' fields from the example
        support = example['support']
        correct_answer = example['correct_answer']
        example_copy = example.copy()
        if len(support) == 0:
            # If 'support' is empty, add some random words and the 'correct_answer' to it
            support = 'random words ' + correct_answer
            example_copy['support'] = support
        # Find the starting index of the 'correct_answer' in the 'support'
        answer_start = find_index(support,correct_answer)
        # If the 'correct_answer' is found in the 'support', update the 'answer_start' field of the example
        if answer_start != -1:
            example_copy['answer_start'] = answer_start
        # If the 'correct_answer' is not found in the 'support', find the most similar phrase to it and update the fields accordingly
        else:
            try:
                # Find the most similar phrase to the 'correct_answer' in the 'support'
                similar = find_most_similar_phrase(support,correct_answer)          
                answer_start = find_index(support, similar)
                example_copy['correct_answer'] = similar
                example_copy['answer_start'] = answer_start
            except:
                # If an exception occurs while finding the most similar phrase, set the 'answer_start' field to -1
                example_copy['answer_start'] = -1
        # Append the updated example to the list of new examples
        new_examples.append(example_copy)
    # Return the list of new examples
    return new_examples

def find_file(filename, search_path):
    """
    Finds a file with the given name in the specified search path and its subdirectories.
    Returns the full path of the file if it exists, or None if it doesn't.
    """
    for root, dir, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None
