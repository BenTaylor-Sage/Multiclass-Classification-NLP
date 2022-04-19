# Databricks notebook source
# MAGIC %md
# MAGIC # Multi-label Text Classification with BERT and PyTorch Lightning

# COMMAND ----------

# MAGIC %md
# MAGIC > TL;DR Learn how to prepare a dataset with toxic comments for multi-label text classification (tagging). We'll fine-tune BERT using PyTorch Lightning and evaluate the model.
# MAGIC 
# MAGIC Multi-label text classification (or tagging text) is one of the most common tasks you'll encounter when doing NLP. Modern Transformer-based models (like BERT) make use of pre-training on vast amounts of text data that makes fine-tuning faster, use fewer resources and more accurate on small(er) datasets.
# MAGIC 
# MAGIC In this tutorial, you'll learn how to:
# MAGIC 
# MAGIC - Load, balance and split text data into sets
# MAGIC - Tokenize text (with BERT tokenizer) and create PyTorch dataset
# MAGIC - Fine-tune BERT model with PyTorch Lightning
# MAGIC - Find out about warmup steps and use a learning rate scheduler
# MAGIC - Use area under the ROC and binary cross-entropy to evaluate the model during training
# MAGIC - How to make predictions using the fine-tuned BERT model
# MAGIC - Evaluate the performance of the model for each class (possible comment tag)
# MAGIC 
# MAGIC Will our model be any good for toxic text detection?
# MAGIC 
# MAGIC - [Read the tutorial](https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/)
# MAGIC - [Run the notebook in your browser (Google Colab)](https://colab.research.google.com/drive/14Ea4lIzsn5EFvPpYKtWStXEByT9qmbkj?usp=sharing)
# MAGIC - [Read the *Getting Things Done with Pytorch* book](https://github.com/curiousily/Getting-Things-Done-with-Pytorch)

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

!pip install pytorch-lightning==1.2.8 --quiet
!pip install transformers==4.5.1 --quiet

# COMMAND ----------

import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

%matplotlib inline
%config InlineBackend.figure_format='retina'

RANDOM_SEED = 42

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

pl.seed_everything(RANDOM_SEED)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data
# MAGIC 
# MAGIC Our dataset contains potentially offensive (toxic) comments and comes from the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). Let's start by download the data (from Google Drive):

# COMMAND ----------

!gdown --id 1VuQ-U7TtggShMeuRSA_hzC8qGDl2LRkr

# COMMAND ----------

# MAGIC %md
# MAGIC Let's load and look at the data:

# COMMAND ----------

df = pd.read_csv("toxic_comments.csv")
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC We have text (comment) and six different toxic labels. Note that we have clean content, too. 
# MAGIC 
# MAGIC Let's split the data:

# COMMAND ----------

train_df, val_df = train_test_split(df, test_size=0.05)
train_df.shape, val_df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing
# MAGIC 
# MAGIC Let's look at the distribution of the labels:

# COMMAND ----------

LABEL_COLUMNS = df.columns.tolist()[2:]
df[LABEL_COLUMNS].sum().sort_values().plot(kind="barh");

# COMMAND ----------

# MAGIC %md
# MAGIC We have a severe case of imbalance. But that is not the full picture. What about the toxic vs clean comments?

# COMMAND ----------

train_toxic = train_df[train_df[LABEL_COLUMNS].sum(axis=1) > 0]
train_clean = train_df[train_df[LABEL_COLUMNS].sum(axis=1) == 0]

pd.DataFrame(dict(
  toxic=[len(train_toxic)], 
  clean=[len(train_clean)]
)).plot(kind='barh');

# COMMAND ----------

# MAGIC %md
# MAGIC Again, we have a severe imbalance in favor of the clean comments. To combat this, we'll sample 15,000 examples from the clean comments and create a new training set:

# COMMAND ----------

train_df = pd.concat([
  train_toxic,
  train_clean.sample(15_000)
])

train_df.shape, val_df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tokenization
# MAGIC 
# MAGIC We need to convert the raw text into a list of tokens. For that, we'll use the built-in BertTokenizer:

# COMMAND ----------

BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's try it out on a sample comment:

# COMMAND ----------

sample_row = df.iloc[16]
sample_comment = sample_row.comment_text
sample_labels = sample_row[LABEL_COLUMNS]

print(sample_comment)
print()
print(sample_labels.to_dict())

# COMMAND ----------

encoding = tokenizer.encode_plus(
  sample_comment,
  add_special_tokens=True,
  max_length=512,
  return_token_type_ids=False,
  padding="max_length",
  return_attention_mask=True,
  return_tensors='pt',
)

encoding.keys()

# COMMAND ----------

encoding["input_ids"].shape, encoding["attention_mask"].shape

# COMMAND ----------

# MAGIC %md
# MAGIC The result of the encoding is a dictionary with token ids `input_ids` and an attention mask `attention_mask` (which tokens should be used by the model 1 - use or 0 - don't use).
# MAGIC 
# MAGIC Let's look at their contents:

# COMMAND ----------

encoding["input_ids"].squeeze()[:20]

# COMMAND ----------

encoding["attention_mask"].squeeze()[:20]

# COMMAND ----------

# MAGIC %md
# MAGIC You can also inverse the tokenization and get back (kinda) the words from the token ids:

# COMMAND ----------

print(tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze())[:20])

# COMMAND ----------

# MAGIC %md
# MAGIC We need to specify the maximum number of tokens when encoding (512 is the maximum we can do). Let's check the number of tokens per comment:

# COMMAND ----------

token_counts = []

for _, row in train_df.iterrows():
  token_count = len(tokenizer.encode(
    row["comment_text"], 
    max_length=512, 
    truncation=True
  ))
  token_counts.append(token_count)

# COMMAND ----------

sns.histplot(token_counts)
plt.xlim([0, 512]);

# COMMAND ----------

# MAGIC %md
# MAGIC Most of the comments contain less than 300 tokens or more than 512. So, we'll stick with the limit of 512.

# COMMAND ----------

MAX_TOKEN_COUNT = 512

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataset
# MAGIC 
# MAGIC We'll wrap the tokenization process in a PyTorch Dataset, along with converting the labels to tensors:

# COMMAND ----------

class ToxicCommentsDataset(Dataset):

  def __init__(
    self, 
    data: pd.DataFrame, 
    tokenizer: BertTokenizer, 
    max_token_len: int = 128
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    comment_text = data_row.comment_text
    labels = data_row[LABEL_COLUMNS]

    encoding = self.tokenizer.encode_plus(
      comment_text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return dict(
      comment_text=comment_text,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.FloatTensor(labels)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Let's have a look at a sample item from the dataset:

# COMMAND ----------

train_dataset = ToxicCommentsDataset(
  train_df,
  tokenizer,
  max_token_len=MAX_TOKEN_COUNT
)

sample_item = train_dataset[0]
sample_item.keys()

# COMMAND ----------

sample_item["comment_text"]

# COMMAND ----------

sample_item["labels"]

# COMMAND ----------

sample_item["input_ids"].shape

# COMMAND ----------

# MAGIC %md
# MAGIC Let's load the BERT model and pass a sample of batch data through:

# COMMAND ----------

bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)

# COMMAND ----------

sample_batch = next(iter(DataLoader(train_dataset, batch_size=8, num_workers=2)))
sample_batch["input_ids"].shape, sample_batch["attention_mask"].shape

# COMMAND ----------

output = bert_model(sample_batch["input_ids"], sample_batch["attention_mask"])

# COMMAND ----------

output.last_hidden_state.shape, output.pooler_output.shape

# COMMAND ----------

# MAGIC %md
# MAGIC The `768` dimension comes from the BERT hidden size:

# COMMAND ----------

bert_model.config.hidden_size

# COMMAND ----------

# MAGIC %md
# MAGIC The larger version of BERT has more attention heads and a larger hidden size.
# MAGIC 
# MAGIC We'll wrap our custom dataset into a [LightningDataModule](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html):

# COMMAND ----------

class ToxicCommentDataModule(pl.LightningDataModule):

  def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len

  def setup(self, stage=None):
    self.train_dataset = ToxicCommentsDataset(
      self.train_df,
      self.tokenizer,
      self.max_token_len
    )

    self.test_dataset = ToxicCommentsDataset(
      self.test_df,
      self.tokenizer,
      self.max_token_len
    )

  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=2
    )

  def val_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=2
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=2
    )

# COMMAND ----------

# MAGIC %md
# MAGIC `ToxicCommentDataModule` encapsulates all data loading logic and returns the necessary data loaders. Let's create an instance of our data module:

# COMMAND ----------

N_EPOCHS = 10
BATCH_SIZE = 12

data_module = ToxicCommentDataModule(
  train_df,
  val_df,
  tokenizer,
  batch_size=BATCH_SIZE,
  max_token_len=MAX_TOKEN_COUNT
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model
# MAGIC 
# MAGIC Our model will use a pre-trained [BertModel](https://huggingface.co/transformers/model_doc/bert.html#bertmodel) and a linear layer to convert the BERT representation to a classification task. We'll pack everything in a [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html):

# COMMAND ----------

class ToxicCommentTagger(pl.LightningModule):

  def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
    super().__init__()
    self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = nn.BCELoss()

  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.pooler_output)
    output = torch.sigmoid(output)    
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output

  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def training_epoch_end(self, outputs):
    
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)

    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)

    for i, name in enumerate(LABEL_COLUMNS):
      class_roc_auc = auroc(predictions[:, i], labels[:, i])
      self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)


  def configure_optimizers(self):

    optimizer = AdamW(self.parameters(), lr=2e-5)

    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )

    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Most of the implementation is just a boilerplate. Two points of interest are the way we configure the optimizers and calculating the area under ROC. We'll dive a bit deeper into those next.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimizer scheduler
# MAGIC 
# MAGIC The job of a scheduler is to change the learning rate of the optimizer during training. This might lead to better performance of our model. We'll use the [get_linear_schedule_with_warmup](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup).
# MAGIC 
# MAGIC Let's have a look at a simple example to make things clearer:

# COMMAND ----------

dummy_model = nn.Linear(2, 1)

optimizer = AdamW(params=dummy_model.parameters(), lr=0.001)

warmup_steps = 20
total_training_steps = 100

scheduler = get_linear_schedule_with_warmup(
  optimizer, 
  num_warmup_steps=warmup_steps,
  num_training_steps=total_training_steps
)

learning_rate_history = []

for step in range(total_training_steps):
  optimizer.step()
  scheduler.step()
  learning_rate_history.append(optimizer.param_groups[0]['lr'])

# COMMAND ----------

plt.plot(learning_rate_history, label="learning rate")
plt.axvline(x=warmup_steps, color="red", linestyle=(0, (5, 10)), label="warmup end")
plt.legend()
plt.xlabel("Step")
plt.ylabel("Learning rate")
plt.tight_layout();

# COMMAND ----------

# MAGIC %md
# MAGIC We simulate 100 training steps and tell the scheduler to warm up for the first 20. The learning rate grows to the initial fixed value of 0.001 during the warm-up and then goes down (linearly) to 0.
# MAGIC 
# MAGIC To use the scheduler, we need to calculate the number of training and warm-up steps. The number of training steps per epoch is equal to `number of training examples / batch size`. The number of total training steps is `training steps per epoch * number of epochs`:

# COMMAND ----------

steps_per_epoch=len(train_df) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS

# COMMAND ----------

# MAGIC %md
# MAGIC We'll use a fifth of the training steps for a warm-up:

# COMMAND ----------

warmup_steps = total_training_steps // 5
warmup_steps, total_training_steps

# COMMAND ----------

# MAGIC %md
# MAGIC We can now create an instance of our model:

# COMMAND ----------

model = ToxicCommentTagger(
  n_classes=len(LABEL_COLUMNS),
  n_warmup_steps=warmup_steps,
  n_training_steps=total_training_steps 
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation
# MAGIC 
# MAGIC Multi-label classification boils down to doing binary classification for each label/tag.
# MAGIC 
# MAGIC We'll use Binary Cross Entropy to measure the error for each label. PyTorch has [BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html), which we're going to combine with a sigmoid function (as we did in the model implementation). Let's look at an example:

# COMMAND ----------

criterion = nn.BCELoss()

prediction = torch.FloatTensor(
  [10.95873564, 1.07321467, 1.58524066, 0.03839076, 15.72987556, 1.09513213]
)
labels = torch.FloatTensor(
  [1., 0., 0., 0., 1., 0.]
) 

# COMMAND ----------

torch.sigmoid(prediction)

# COMMAND ----------

criterion(torch.sigmoid(prediction), labels)

# COMMAND ----------

# MAGIC %md
# MAGIC We can use the same approach to calculate the loss of the predictions:

# COMMAND ----------

_, predictions = model(sample_batch["input_ids"], sample_batch["attention_mask"])
predictions

# COMMAND ----------

criterion(predictions, sample_batch["labels"])

# COMMAND ----------

# MAGIC %md
# MAGIC #### ROC Curve
# MAGIC 
# MAGIC Another metric we're going to use is the area under the Receiver operating characteristic (ROC) for each tag. ROC is created by plotting the True Positive Rate (TPR) vs False Positive Rate (FPR):
# MAGIC 
# MAGIC $$
# MAGIC \text{TPR} = \frac{\text{TP}}{\text{TP} \text{+} \text{FN}}
# MAGIC $$
# MAGIC 
# MAGIC $$
# MAGIC \text{FPR} = \frac{\text{FP}}{\text{FP} \text{+} \text{TN}}
# MAGIC $$

# COMMAND ----------

from sklearn import metrics

fpr = [0.        , 0.        , 0.        , 0.02857143, 0.02857143,
       0.11428571, 0.11428571, 0.2       , 0.4       , 1.        ]

tpr = [0.        , 0.01265823, 0.67202532, 0.76202532, 0.91468354,
       0.97468354, 0.98734177, 0.98734177, 1.        , 1.        ]

_, ax = plt.subplots()
ax.plot(fpr, tpr, label="ROC")
ax.plot([0.05, 0.95], [0.05, 0.95], transform=ax.transAxes, label="Random classifier", color="red")
ax.legend(loc=4)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Example ROC curve")
plt.show();

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training

# COMMAND ----------

!rm -rf lightning_logs/
!rm -rf checkpoints/
