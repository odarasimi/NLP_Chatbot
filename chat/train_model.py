''' This code below was used to train the model used for our Extractive QA, 
finetuned with a Science Question and answering dataset. 
The model was trained and uploaded to the Huggingface hub to deal with size and 
can be found at https://huggingface.co/anuoluwa/scincequest_model.'''

from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoModelForQuestionAnswering, TrainingArguments, AdamW, AutoTokenizer, Trainer
import tensorflow as tf
from tensorflow import keras
from transformers import DefaultDataCollator
from chat.utils import rename_columns, update_sample, create_corpus

# Initialize base model and tokenizer
model_base = "deepset/tinyroberta-squad2" #"distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_base)
max_length = tokenizer.model_max_length

# Load the training datasets from Huggingface
sciq = load_dataset("sciq")
openbookqa = load_dataset("openbookqa", "additional")
quartz = load_dataset("quartz")

# Create document called sci_wiki.txt which is a large corpus to be used for context during inference
# The contexts for each dataset will be merged to create the document
create_corpus(sciq, context="support", file_name="sciq")
create_corpus(openbookqa, context="fact1", file_name="sci_wiki")
create_corpus(quartz, context="para", file_name="sci_wiki")

# Update the DatasetDict object by adding an answer_start and formatting the answer if not found in context,
# the DatasetDict object is immutable, so we create a new list of its contents

new_train_examples = update_sample(sciq['train'])
new_validation_examples = update_sample(sciq['validation'])
new_test_examples = update_sample(sciq['test'])

# When the DatasetDict object is formatted, it gets new column names, 
# we collapse it and obtain the  list below, which we pass into our renaming function.
data_columns = ['train.answer_start', 'validation.answer_start', 'train.correct_answer', 
    'validation.correct_answer', 'train.distractor1', 'validation.distractor1', 'train.distractor2',
    'validation.distractor2', 'train.distractor3', 'validation.distractor3', 'train.question', 'validation.question', 
    'train.support', 'validation.support', 'test.answer_start', 'test.correct_answer', 
    'test.distractor1', 'test.distractor2', 'test.distractor3', 'test.question', 
    'test.support'
]

# Creating three new instances of the DatasetDict class, new_train_dataset, new_validation_dataset, and new_test_dataset, respectively.
new_train_dataset = Dataset.from_dict({'train': new_train_examples})
new_validation_dataset = Dataset.from_dict({'validation': new_validation_examples})
new_test_dataset = Dataset.from_dict({'test': new_test_examples})

# Update the 'train', 'validation', 'test' keys of the sciq DatasetDict with the new Dataset object
# The column names will be formatted and will need to be flattened
sciq.update({'train': new_train_dataset})
sciq.update({'validation': new_validation_dataset})
sciq.update({'test': new_test_dataset})

# Collapses the column names in the dataset
sciq = sciq.flatten() 

rename_columns(sciq, data_columns) # Rename column names of DatasetDict object

# Preprocess inputs and return tokens for the "correct answer" within the context
# because the model needs a way to locate the start and end position of the answer within the context.
def preprocess_inputs(data):

    tokenized = tokenizer(
        data["question"],
        data["support"],
        max_length=max_length,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
    )

    # The layer expects these arguments for answer positions
    start_positions = []
    end_positions = []
    
    for i, (answer, offset) in enumerate(zip(data["correct_answer"], tokenized["offset_mapping"])):
        start_char = int(data["answer_start"][i])
        if start_char != -1:
            end_char = start_char + len([answer][0])
        else:
            start_char = -1
            end_char = -1
        sequence_ids = tokenized.sequence_ids(i)

        # Mark the start and end of the context
        idx_mark = 0
        while sequence_ids[idx_mark] != 1:
            idx_mark += 1
        context_start = idx_mark
        while sequence_ids[idx_mark] == 1:
            idx_mark += 1
        context_end = idx_mark - 1

        # If the answer is not within context, it is labelled (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(tokenized.char_to_token(i, start_char, sequence_index=1))
            end_positions.append(tokenized.char_to_token(i, end_char - 1, sequence_index=1))

    tokenized["start_positions"], tokenized["end_positions"] = start_positions, end_positions

    # The TFRobertaForQuestionAnswering layer may not support the offset mapping key, so we remove it
    offset_mapping_removed = tokenized.pop("offset_mapping")

    return tokenized

# Map preprocessing function to all samples in the dataset
tokenized_sciq = sciq.map(preprocess_inputs, batched=True, remove_columns=sciq["train"].column_names)

# Load the whole dataset as a dict of pytorch arrays
training = tokenized_sciq["train"].with_format("torch") 
validation = tokenized_sciq["validation"].with_format("torch")

# Load pretrained model
model = AutoModelForQuestionAnswering.from_pretrained(model_base) 

model_name = "scincequest_model" # Name of our model which we will train

# Supply training arguments
args = TrainingArguments(
    model_name,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

# Instantiate a simple data collator
data_collator = DefaultDataCollator()

# Provide arguments for Trainer
trainer = Trainer(
    model,
    args,
    train_dataset=training,
    eval_dataset=validation,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train model
trainer.train()