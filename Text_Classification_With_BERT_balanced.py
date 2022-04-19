# Databricks notebook source
# MAGIC %sh
# MAGIC pip install transformers==2.11.0 --user

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
pd.options.mode.chained_assignment = None

# COMMAND ----------

data = pd.read_csv("/dbfs/FileStore/amazon_reviews_us_Mobile_Electronics_v1_00.csv", low_memory=False)
data.head()

# COMMAND ----------

df = data[['review_body','star_rating']]
df.dropna(inplace=True)
df['star_rating'] = df['star_rating'].astype(int)

# COMMAND ----------

### Over sampling to balance Dataset

# COMMAND ----------

df['star_rating'].value_counts()

# COMMAND ----------

possible_labels = df.star_rating.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
label_dict

# COMMAND ----------

df['label'] = df.star_rating.replace(label_dict)

# COMMAND ----------

df.head()

# COMMAND ----------

df.label.value_counts()

# COMMAND ----------

# separate data by labels
onestar = df[df.label==0]
twostar = df[df.label==1]
threestar = df[df.label==2]
fourstar = df[df.label==3]
fivestar = df[df.label==4]

# upsample minority labels
fivestar_upsampled = resample(fivestar, replace=True, n_samples=len(onestar), random_state=42)
twostar_upsampled = resample(twostar, replace=True, n_samples=len(onestar), random_state=42)
threestar_upsampled = resample(threestar, replace=True, n_samples=len(onestar), random_state=42)
fourstar_upsampled = resample(fourstar, replace=True, n_samples=len(onestar), random_state=42)

# combine sets back together
data_sampled = pd.concat([onestar, twostar_upsampled, threestar_upsampled, fourstar_upsampled, fivestar_upsampled])
# check data is now balanced
data_sampled.label.value_counts()

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(data_sampled.index.values, 
                                                  data_sampled.label.values, 
                                                  test_size=0.15, 
                                                  random_state=42, 
                                                  stratify=data_sampled.label.values)

# COMMAND ----------

pd.options.mode.chained_assignment = None
data_sampled['data_type'] = ['not_set']*data_sampled.shape[0]

data_sampled.loc[X_train, 'data_type'] = 'train'
data_sampled.loc[X_val, 'data_type'] = 'val'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleaning

# COMMAND ----------

data_sampled.groupby(['review_body', 'label', 'data_type']).count()

# COMMAND ----------

import copy
df = copy.deepcopy(data_sampled)

# COMMAND ----------

import re

def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', x)
    return x

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

# COMMAND ----------

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re
contractions, contractions_re = _get_contractions(contraction_dict)
def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)
# Usage
replace_contractions("this's a text with contraction")

# COMMAND ----------

# lower the text
df["review_body"] = df["review_body"].apply(lambda x: x.lower())

# Clean the text
df["review_body"] = df["review_body"].apply(lambda x: clean_text(x))

# Clean numbers
df["review_body"] = df["review_body"].apply(lambda x: clean_numbers(x))

# Clean Contractions
df["review_body"] = df["review_body"].apply(lambda x: replace_contractions(x))

# COMMAND ----------

df.head()

# COMMAND ----------

def length_review(x):
  return len(x.split())

df['length_text'] = df['review_body'].apply(length_review) # count number of tokens

# COMMAND ----------

df['length_text'].max(),df['length_text'].min(), df['length_text'].mean()

# COMMAND ----------

df['length_text'].describe()

# COMMAND ----------

df['length_text'].hist(bins=100)

# COMMAND ----------

df['length_text'].plot.kde()

# COMMAND ----------

df.drop("length_text",inplace = True, axis=1)

# COMMAND ----------

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# COMMAND ----------

encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].review_body.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
  truncation=True,
    pad_to_max_length=True, 
    max_length=128, 
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].review_body.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
  truncation=True,
    pad_to_max_length=True, 
    max_length=128, 
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

# COMMAND ----------

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

# COMMAND ----------

len(dataset_train), len(dataset_val)

# COMMAND ----------

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

# COMMAND ----------

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 3

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)

# COMMAND ----------

from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)

# COMMAND ----------

epochs = 5

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

# COMMAND ----------

from sklearn.metrics import f1_score

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

# COMMAND ----------

import random
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# COMMAND ----------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)

# COMMAND ----------

def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

# COMMAND ----------

dbutils.fs.mkdirs("/mnt/nlp/data_volume")

# COMMAND ----------

import datetime
# perform evaluation CHURNED_BEFORE_365_DAYS
experiment_name="NLP_text_classification"
user_name= "juan.romero@sage.com"
experiment_folder = 'experiments'
experimentName = f"/Users/{user_name}/{experiment_folder}/{experiment_name}" 
mlflow.set_experiment(experimentName)

time1=datetime.datetime.now().strftime(format="%Y%m%d_%H%M")
with mlflow.start_run(run_name='bert_uncased_model'):
  run_id = mlflow.active_run().info.run_id
  print(f"run_id : {run_id}")


  for epoch in tqdm(range(1, epochs+1)):

      model.train()

      loss_train_total = 0

      progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
      for batch in progress_bar:

          model.zero_grad()

          batch = tuple(b.to(device) for b in batch)

          inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                   }       

          outputs = model(**inputs)

          loss = outputs[0]
          loss_train_total += loss.item()
          loss.backward()

          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

          optimizer.step()
          scheduler.step()

          progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})


      torch.save(model.state_dict(), f'/dbfs/mnt/nlp/data_volume/finetuned_BERT_epoch_{epoch}.model')#nlp/data_volume/test
      tqdm.write(f'\nEpoch {epoch}')

      loss_train_avg = loss_train_total/len(dataloader_train)            
      tqdm.write(f'Training loss: {loss_train_avg}')

      val_loss, predictions, true_vals = evaluate(dataloader_validation)
      val_f1 = f1_score_func(predictions, true_vals)
      
      mlflow.log_metric('Training_loss', loss_train_avg)
      mlflow.log_metric('Validation_loss', val_loss)
      mlflow.log_metric('F1_Score', val_f1)
      
      tqdm.write(f'Validation loss: {val_loss}')
      tqdm.write(f'F1 Score (Weighted): {val_f1}')

# COMMAND ----------

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

# COMMAND ----------

model.load_state_dict(torch.load('/dbfs/mnt/nlp/data_volume/finetuned_BERT_epoch_1.model', map_location=torch.device('cpu')))

# COMMAND ----------

_, predictions, true_vals = evaluate(dataloader_validation)

# COMMAND ----------

accuracy_per_class(predictions, true_vals)
