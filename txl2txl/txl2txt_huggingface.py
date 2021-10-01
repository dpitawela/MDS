# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from transformers import EncoderDecoderModel, TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import datasets
import os


# %%
download = False # to identify whether to download data, tokenizer, model

dsName = 'multi_x_science_sum'
tokenizerName = 'transfo-xl-wt103'
modelName = 'transfo-xl-wt103'

small_ds = True
small_ds_size = {'train':8, 'test':8, 'validation':8}

batch_size=4


# %%
# loading the dataset
if download:
    ds = datasets.load_dataset(dsName)
    ds.save_to_disk('data')

ds = datasets.load_from_disk('data')

if small_ds:
    train_data, test_data, validation_data = ds['train'].select(range(small_ds_size['train'])), ds['test'].select(range(small_ds_size['test'])), ds['validation'].select(range(small_ds_size['validation']))
else:
    train_data, test_data, validation_data = ds['train'],  ds['test'],  ds['validation']

del ds # to save memory
# train_ds[0]['ref_abstract']['abstract'][1]


# %%
# loading the tokenizer
if download:
    tokenizer = TransfoXLTokenizer.from_pretrained(tokenizerName, force_download=True)
    tokenizer.save_pretrained('Tokenizer')
else:
    tokenizer = TransfoXLTokenizer.from_pretrained('Tokenizer', local_files_only=True)

tokenizer.add_special_tokens({'pad_token': '<pad>', 'cls_token':'<cls>'})


# %%
tokenizer.special_tokens_map


# %%
def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  
  abstracts = batch['abstract'] # abstract sections of the current batch
  # list of ref_abstract dictionaries are taken and their sub lists of 'abstract' are merged into single list item
  ref_abstracts = [' '.join(ref_ab['abstract']) for ref_ab in batch['ref_abstract']]

  # each abstract is merged with it's referenced abstracts
  inputText = [abstract + ' ' + ref_abstract for abstract, ref_abstract in zip(abstracts, ref_abstracts)]
  # 'related_work' is the target
  outputText = batch["related_work"]

  inputs = tokenizer(inputText, return_attention_mask=True, add_special_tokens=True, padding="max_length", max_length=500, truncation=True)
  outputs = tokenizer(outputText, return_attention_mask=True, add_special_tokens=True, padding="max_length", max_length=500, truncation=True)

  
  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask

  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask

  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  # batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch


# %%
# encoding training data
cols_to_remove = ["aid", "mid", "abstract", "related_work", "ref_abstract"]
new_cols = ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"]
# new_cols = ["input_ids", "attention_mask", "decoder_input_ids", "labels"]

train_data_processed = train_data.map(
    process_data_to_model_inputs,
    load_from_cache_file=False, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=cols_to_remove
)
train_data_processed.set_format(type="torch", columns=new_cols)

# encoding testing data
test_data_processed = test_data.map(
    process_data_to_model_inputs,
    load_from_cache_file=False,
    batched=True, 
    batch_size=batch_size, 
    remove_columns=cols_to_remove
)
test_data_processed.set_format(type="torch", columns=new_cols)

# encoding validation data 
validation_data_processed = validation_data.map(
    process_data_to_model_inputs,
    load_from_cache_file=False,
    batched=True, 
    batch_size=batch_size, 
    remove_columns=cols_to_remove
)
validation_data_processed.set_format(type="torch", columns=new_cols)


# %%
# loading the model
if download:
    txl2txl = EncoderDecoderModel.from_encoder_decoder_pretrained(modelName, modelName, force_download=True)
    txl2txl.save_pretrained("txl2txl")
else:
    # txl2txl = EncoderDecoderModel.from_encoder_decoder_pretrained('encoder', 'decoder')
    # txl2txl = EncoderDecoderModel.from_encoder_decoder_pretrained(modelName, modelName)
    txl2txl = EncoderDecoderModel.from_pretrained("txl2txl")
    txl2txl.train()


# %%
# txl2txl.config.decoder_start_token_id = tokenizer.cls_token_id
# txl2txl.config.eos_token_id = tokenizer.sep_token_id
txl2txl.config.pad_token_id = tokenizer.pad_token_id
txl2txl.config.vocab_size = txl2txl.config.encoder.vocab_size
txl2txl.config.decoder.pad_token_id = tokenizer.pad_token


# %%
# txl2txl.config.max_length = 142
# txl2txl.config.length_penalty = 2.0

txl2txl.config.no_repeat_ngram_size = 3
txl2txl.config.early_stopping = True
txl2txl.config.length_penalty = 2.0
txl2txl.config.num_beams = 4


# %%
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # fp16=True, 
    output_dir="output",
    # logging_steps=2,
    # save_steps=10,
    # eval_steps=4,
    num_train_epochs=1,
    # logging_steps=1000,
    # save_steps=500,
    # eval_steps=7500,
    # warmup_steps=2000,
    # save_total_limit=3,
)


# %%
rouge = datasets.load_metric("rouge")
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# %%
trainer = Seq2SeqTrainer(
    model=txl2txl,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data_processed,
    eval_dataset=validation_data_processed
)
trainer.train()


# %%
def generate_summary(batch):
    # cut off at BERT max length 512
    abstracts = batch['abstract']
    ref_abstracts = [' '.join(ref_ab['abstract']) for ref_ab in batch['ref_abstract']]
    inputText = [abstract + ' ' + ref_abstract for abstract, ref_abstract in zip(abstracts, ref_abstracts)]
    inputs = tokenizer(inputText, return_attention_mask=True)

    input_ids = inputs.input_ids
    # attention_mask = inputs.attention_mask
    outputs = txl2txl.generate(input_ids) # attention_mask = attention_mask

    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["pred_summary"] = output_str
    return batch


# %%
results = test_data.map(generate_summary, batch_size=batch_size, batched=True)
results


# %%
rouge.compute(predictions=results["pred_summary"], references=results["related_work"], rouge_types=["rouge2"])["rouge2"].mid


# %%
# def printResults(index):
#     print("Abstract\n", results[index]['abstract'])
#     print("\nRef Abstracts\n", results[0]['ref_abstract']['abstract'])
#     print("\nGenerated Related Work\n", results[0]['pred_summary'])

# printResults(0)


# %%
# to show configurations
txl2txl


