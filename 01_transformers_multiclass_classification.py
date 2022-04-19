# Databricks notebook source
# MAGIC %md
# MAGIC # Fine Tuning Transformer for MultiClass Text Classification

# COMMAND ----------

# MAGIC %md
# MAGIC ### Introduction
# MAGIC 
# MAGIC In this tutorial we will be fine tuning a transformer model for the **Multiclass text classification** problem. 
# MAGIC This is one of the most common business problems where a given piece of text/sentence/document needs to be classified into one of the categories out of the given list.
# MAGIC 
# MAGIC #### Flow of the notebook
# MAGIC 
# MAGIC The notebook will be divided into seperate sections to provide a organized walk through for the process used. This process can be modified for individual use cases. The sections are:
# MAGIC 
# MAGIC 1. [Importing Python Libraries and preparing the environment](#section01)
# MAGIC 2. [Importing and Pre-Processing the domain data](#section02)
# MAGIC 3. [Preparing the Dataset and Dataloader](#section03)
# MAGIC 4. [Creating the Neural Network for Fine Tuning](#section04)
# MAGIC 5. [Fine Tuning the Model](#section05)
# MAGIC 6. [Validating the Model Performance](#section06)
# MAGIC 7. [Saving the model and artifacts for Inference in Future](#section07)
# MAGIC 
# MAGIC #### Technical Details
# MAGIC 
# MAGIC This script leverages on multiple tools designed by other teams. Details of the tools used below. Please ensure that these elements are present in your setup to successfully implement this script.
# MAGIC 
# MAGIC  - Data: 
# MAGIC 	 - We are using the News aggregator dataset available at by [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/News+Aggregator)
# MAGIC 	 - We are referring only to the first csv file from the data dump: `newsCorpora.csv`
# MAGIC 	 - There are `422937` rows of data.  Where each row has the following data-point: 
# MAGIC 		 - ID Numeric ID  
# MAGIC 		 - TITLE News title  
# MAGIC 		 - URL Url  
# MAGIC 		 - PUBLISHER Publisher name  
# MAGIC 		 - CATEGORY News category (b = business, t = science and technology, e = entertainment, m = health)  
# MAGIC 		 - STORY Alphanumeric ID of the cluster that includes news about the same story  
# MAGIC 		 - HOSTNAME Url hostname  
# MAGIC 		 - TIMESTAMP Approximate time the news was published, as the number of milliseconds since the epoch 00:00:00 GMT, January 1, 1970
# MAGIC 
# MAGIC 
# MAGIC  - Language Model Used:
# MAGIC 	 - DistilBERT this is a smaller transformer model as compared to BERT or Roberta. It is created by process of distillation applied to Bert. 
# MAGIC 	 - [Blog-Post](https://medium.com/huggingface/distilbert-8cf3380435b5)
# MAGIC 	 - [Research Paper](https://arxiv.org/abs/1910.01108)
# MAGIC      - [Documentation for python](https://huggingface.co/transformers/model_doc/distilbert.html)
# MAGIC 
# MAGIC 
# MAGIC  - Hardware Requirements:
# MAGIC 	 - Python 3.6 and above
# MAGIC 	 - Pytorch, Transformers and All the stock Python ML Libraries
# MAGIC 	 - GPU enabled setup 
# MAGIC 
# MAGIC 
# MAGIC  - Script Objective:
# MAGIC 	 - The objective of this script is to fine tune DistilBERT to be able to classify a news headline into the following categories:
# MAGIC 		 - Business
# MAGIC 		 - Technology
# MAGIC 		 - Health
# MAGIC 		 - Entertainment 

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section01'></a>
# MAGIC ### Importing Python Libraries and preparing the environment
# MAGIC 
# MAGIC At this step we will be importing the libraries and modules needed to run our script. Libraries are:
# MAGIC * Pandas
# MAGIC * Pytorch
# MAGIC * Pytorch Utils for Dataset and Dataloader
# MAGIC * Transformers
# MAGIC * DistilBERT Model and Tokenizer
# MAGIC 
# MAGIC Followed by that we will preapre the device for CUDA execeution. This configuration is needed if you want to leverage on onboard GPU. 

# COMMAND ----------

# from google.colab import drive
# drive.mount('/content/drive')

# COMMAND ----------

!pip install transformers

# COMMAND ----------

# Importing the libraries needed
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer

# COMMAND ----------

# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section02'></a>
# MAGIC ### Importing and Pre-Processing the domain data
# MAGIC 
# MAGIC We will be working with the data and preparing for fine tuning purposes. 
# MAGIC *Assuming that the `newCorpora.csv` is already downloaded in your `data` folder*
# MAGIC 
# MAGIC Import the file in a dataframe and give it the headers as per the documentation.
# MAGIC Cleaning the file to remove the unwanted columns and create an additional column for training.
# MAGIC The final Dataframe will be something like this:
# MAGIC 
# MAGIC |TITLE|CATEGORY|ENCODED_CAT|
# MAGIC |--|--|--|
# MAGIC |  title_1|Entertainment | 1 |
# MAGIC |  title_2|Entertainment | 1 |
# MAGIC |  title_3|Business| 2 |
# MAGIC |  title_4|Science| 3 |
# MAGIC |  title_5|Science| 3 |
# MAGIC |  title_6|Health| 4 |

# COMMAND ----------

df = pd.read_csv("/content/drive/MyDrive/colab/datasets/reviews.csv", low_memory=False)
df.head()

# COMMAND ----------

df.columns

# COMMAND ----------


import pandas as pd
pd.options.mode.chained_assignment = None
df = df[['review_body','star_rating']]

df.dropna(inplace=True)

df['star_rating'] = df['star_rating'].astype(int)

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
def remove_punctuation(x):
  new_string = re.sub(r'[^\w\s]', '', x)
  return new_string
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

df["review_body"] = df["review_body"].apply(lambda x: remove_punctuation(x))

# COMMAND ----------

# Import the csv into pandas dataframe and add the headers
#df = pd.read_csv('./data/newsCorpora.csv', sep='\t', names=['ID','TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
# df.head()
# # Removing unwanted columns and only leaving title of news and the category which will be the target

# df.head()

# # Converting the codes to appropriate categories using a dictionary
my_dict = {
    '1':'1',
    '2':'2',
    '3':'3',
    '4':'4',
    '5':'5',
}

def update_cat(x):
    return my_dict[str(int(x))]

df['CATEGORY'] = df['star_rating'].apply(lambda x: update_cat(x))

encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x]=len(encode_dict)
    return encode_dict[x]

df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))

# COMMAND ----------

df.head()

# COMMAND ----------

df.columns = ['TITLE', 'star_rating', 'CATEGORY', 'ENCODE_CAT']
df.head()

# COMMAND ----------

from sklearn.utils import resample

# separate data by labels
onestar = df[df.star_rating==1]
twostar = df[df.star_rating==2]
threestar = df[df.star_rating==3]
fourstar = df[df.star_rating==4]
fivestar = df[df.star_rating==5]

# upsample minority labels
onestar_upsampled = resample(onestar, replace=True, n_samples=len(fivestar), random_state=42)
twostar_upsampled = resample(twostar, replace=True, n_samples=len(fivestar), random_state=42)
threestar_upsampled = resample(threestar, replace=True, n_samples=len(fivestar), random_state=42)
fourstar_upsampled = resample(fourstar, replace=True, n_samples=len(fivestar), random_state=42)

# combine sets back together
data_sampled = pd.concat([onestar_upsampled, twostar_upsampled, threestar_upsampled, fourstar_upsampled, fivestar])
# check data is now balanced
data_sampled.star_rating.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section03'></a>
# MAGIC ### Preparing the Dataset and Dataloader
# MAGIC 
# MAGIC We will start with defining few key variables that will be used later during the training/fine tuning stage.
# MAGIC Followed by creation of Dataset class - This defines how the text is pre-processed before sending it to the neural network. We will also define the Dataloader that will feed  the data in batches to the neural network for suitable training and processing. 
# MAGIC Dataset and Dataloader are constructs of the PyTorch library for defining and controlling the data pre-processing and its passage to neural network. For further reading into Dataset and Dataloader read the [docs at PyTorch](https://pytorch.org/docs/stable/data.html)
# MAGIC 
# MAGIC #### *Triage* Dataset Class
# MAGIC - This class is defined to accept the Dataframe as input and generate tokenized output that is used by the DistilBERT model for training. 
# MAGIC - We are using the DistilBERT tokenizer to tokenize the data in the `TITLE` column of the dataframe. 
# MAGIC - The tokenizer uses the `encode_plus` method to perform tokenization and generate the necessary outputs, namely: `ids`, `attention_mask`
# MAGIC - To read further into the tokenizer, [refer to this document](https://huggingface.co/transformers/model_doc/distilbert.html#distilberttokenizer)
# MAGIC - `target` is the encoded category on the news headline. 
# MAGIC - The *Triage* class is used to create 2 datasets, for training and for validation.
# MAGIC - *Training Dataset* is used to fine tune the model: **80% of the original data**
# MAGIC - *Validation Dataset* is used to evaluate the performance of the model. The model has not seen this data during training. 
# MAGIC 
# MAGIC #### Dataloader
# MAGIC - Dataloader is used to for creating training and validation dataloader that load data to the neural network in a defined manner. This is needed because all the data from the dataset cannot be loaded to the memory at once, hence the amount of dataloaded to the memory and then passed to the neural network needs to be controlled.
# MAGIC - This control is achieved using the parameters such as `batch_size` and `max_len`.
# MAGIC - Training and Validation dataloaders are used in the training and validation part of the flow respectively

# COMMAND ----------

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

# COMMAND ----------

class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        title = str(self.data.TITLE[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len

# COMMAND ----------

# Creating the dataset and dataloader for the neural network

train_size = 0.8
train_dataset=df.sample(frac=train_size,random_state=200)
test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = Triage(train_dataset, tokenizer, MAX_LEN)
testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

# COMMAND ----------

train_dataset.head()

# COMMAND ----------

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section04'></a>
# MAGIC ### Creating the Neural Network for Fine Tuning
# MAGIC 
# MAGIC #### Neural Network
# MAGIC  - We will be creating a neural network with the `DistillBERTClass`. 
# MAGIC  - This network will have the DistilBERT Language model followed by a `dropout` and finally a `Linear` layer to obtain the final outputs. 
# MAGIC  - The data will be fed to the DistilBERT Language model as defined in the dataset. 
# MAGIC  - Final layer outputs is what will be compared to the `encoded category` to determine the accuracy of models prediction. 
# MAGIC  - We will initiate an instance of the network called `model`. This instance will be used for training and then to save the final trained model for future inference. 
# MAGIC  
# MAGIC #### Loss Function and Optimizer
# MAGIC  - `Loss Function` and `Optimizer` and defined in the next cell.
# MAGIC  - The `Loss Function` is used the calculate the difference in the output created by the model and the actual output. 
# MAGIC  - `Optimizer` is used to update the weights of the neural network to improve its performance.
# MAGIC  
# MAGIC #### Further Reading
# MAGIC - You can refer to my [Pytorch Tutorials](https://github.com/abhimishra91/pytorch-tutorials) to get an intuition of Loss Function and Optimizer.
# MAGIC - [Pytorch Documentation for Loss Function](https://pytorch.org/docs/stable/nn.html#loss-functions)
# MAGIC - [Pytorch Documentation for Optimizer](https://pytorch.org/docs/stable/optim.html)
# MAGIC - Refer to the links provided on the top of the notebook to read more about DistiBERT. 

# COMMAND ----------

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# COMMAND ----------

model = DistillBERTClass()
model.to(device)

# COMMAND ----------

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section05'></a>
# MAGIC ### Fine Tuning the Model
# MAGIC 
# MAGIC After all the effort of loading and preparing the data and datasets, creating the model and defining its loss and optimizer. This is probably the easier steps in the process. 
# MAGIC 
# MAGIC Here we define a training function that trains the model on the training dataset created above, specified number of times (EPOCH), An epoch defines how many times the complete data will be passed through the network. 
# MAGIC 
# MAGIC Following events happen in this function to fine tune the neural network:
# MAGIC - The dataloader passes data to the model based on the batch size. 
# MAGIC - Subsequent output from the model and the actual category are compared to calculate the loss. 
# MAGIC - Loss value is used to optimize the weights of the neurons in the network.
# MAGIC - After every 5000 steps the loss value is printed in the console.
# MAGIC 
# MAGIC As you can see just in 1 epoch by the final step the model was working with a miniscule loss of 0.0002485 i.e. the output is extremely close to the actual output.

# COMMAND ----------

# Function to calcuate the accuracy of the model

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

# COMMAND ----------

# Defining the training function on the 80% of the dataset for tuning the distilbert model

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if _%5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return 

# COMMAND ----------

for epoch in range(EPOCHS):
    train(epoch)

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section06'></a>
# MAGIC ### Validating the Model
# MAGIC 
# MAGIC During the validation stage we pass the unseen data(Testing Dataset) to the model. This step determines how good the model performs on the unseen data. 
# MAGIC 
# MAGIC This unseen data is the 20% of `newscorpora.csv` which was seperated during the Dataset creation stage. 
# MAGIC During the validation stage the weights of the model are not updated. Only the final output is compared to the actual value. This comparison is then used to calcuate the accuracy of the model. 
# MAGIC 
# MAGIC As you can see the model is predicting the correct category of a given headline to a 99.9% accuracy.

# COMMAND ----------

def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return epoch_accu


# COMMAND ----------

print('This is the validation section to print the accuracy and see how it performs')
print('Here we are leveraging on the dataloader crearted for the validation dataset, the approcah is using more of pytorch')

acc = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % acc)

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section07'></a>
# MAGIC ### Saving the Trained Model Artifacts for inference
# MAGIC 
# MAGIC This is the final step in the process of fine tuning the model. 
# MAGIC 
# MAGIC The model and its vocabulary are saved locally. These files are then used in the future to make inference on new inputs of news headlines.
# MAGIC 
# MAGIC Please remember that a trained neural network is only useful when used in actual inference after its training. 
# MAGIC 
# MAGIC In the lifecycle of an ML projects this is only half the job done. We will leave the inference of these models for some other day. 

# COMMAND ----------

# Saving the files for re-use

output_model_file = '/content/drive/MyDrive/Colab Notebooks/saves/pytorch_distilbert_news.bin'
output_vocab_file = '/content/drive/MyDrive/Colab Notebooks/saves/vocab_distilbert_news.bin'

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)

print('All files saved')
print('This tutorial is completed')
