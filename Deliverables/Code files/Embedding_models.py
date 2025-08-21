from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import json 
from types import MethodType
from sklearn.manifold import TSNE
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Optional
import math
import seaborn as sns

# In this file, we collect the code that was implemented for the circuit analysis of embedding models. In particular, we 
# create a class "EmbeddingCircuitAnalyzer" that takes in an encoder embedding model as input and collects all the analysis techniques 
# that one may wish to use to analyze the model. This makes it easier to make use of the methods that we implemented. 



### Some auxiliary functions 



def read_file(file): 
    """
    Reads a json file with questions and answers and returns lists with the questions, correct answers 
    (and potentially wrong answers). Each line in the json file should contain a dictionary with information about a question-answer pair; 
    the "questions" key gives the question, the "rationale" key gives the correct answer (or rationale), and the  "wrong i" key gives 
    the i-th wrong answer option. 
    """
    with open(file,"r") as infile: 
        infile.readline()
        lines = infile.readlines()

    #strip unwanted punctuation at the end of the dictionary at each line in the file 
    data = [json.loads(line.strip("\n").rstrip(',').rstrip("]")) for line in lines] 

    questions = [line["question"] for line in data]   #questions 
    answers = [line["rationale"] for line in data]     #correct answers

    #register wrong answer options in the wrong_answers dictionary (in the case when the file contains wrong answers) 
    wrong_answers = {}
    for i in range(len(data)):  
            for j in range(4): 
                if "wrong " + str(j+1) in data[i]:
                    wrong_answers[f"{j}.{i}"] = data[i]["wrong " + str(j+1)]

    return questions, answers, wrong_answers

def tSNEvisualization(embeddings, color_names, hover_names): 
    """
    Performs a t-SNE analysis with two dimensions and visualizes the results. 

    Arguments: 
        embeddings 
            a tensor or list with the embedding vectors that should be visualized 
        color_names 
            a list with the color names of all the embedding vectors
        hover_names 
            a list with the names of all embedding vectors when you hover over them
    """

    tsne = TSNE(n_components=2, random_state=30)
    tSNE_embeddings = tsne.fit_transform(embeddings.numpy())

    x = [x for x, y in tSNE_embeddings]
    y = [y for x, y in tSNE_embeddings]

    fig = px.scatter(
        tSNE_embeddings, x=0, y=1,
        width=700, height=700,
        color = color_names,
        hover_name=hover_names,
    )
    fig.update_layout(
        xaxis_title = "dim1", 
        yaxis_title = "dim2"   
    )
    
    fig.update_traces(
    hovertemplate=None,
    )
    
    fig.show()

def closest_neighbors(embedding_Q,embedding_A,k): 
    '''
    Returns the percentage of question-answer pairs whose questions contain their corresponding answers 
    among the k nearest neighbors from the dataset of questions and answers. 
    The distance between questions and answers are computed using the L2 norm (though one would get the same result 
    if one used cosine distance). 

    Arguments: 
        embedding_Q 
            embedding vectors of the questions. 
        embedding_A 
            embedding vectors of the answers
        k 
            the number of neighbors checked. 
    '''

    # counts the number of question-answer pairs that are within the k nearest neighbors
    answer_is_near = 0

    #concatenate questions and answers into one dataset
    embedding = torch.cat([embedding_Q,embedding_A],dim=0)  

    #go through the questions and checks if answer is within k nearest neighbors; if yes, add 1 to answer_is_near
    for i in range(len(embedding_Q)): 
        distances, indices = torch.topk(-torch.norm(embedding[i] - embedding,dim=1),k)
        if torch.isin(i+len(embedding_Q),indices): 
            answer_is_near += 1

    return answer_is_near/len(embedding_Q)



### Code for doing circuit analysis



## We define some functions and objects that are needed in the EmbeddingCircuitAnalyzer class. These functions and objects are implemented based on 
## the multi-qa-MiniLM-L6-cos-v1 model, and should function as an example of how the arguments to be used in the EmbeddingCircuitAnalyzer class can be instantiated. 


# initialize a tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", trust_remote_code = True, use_fast = False)

#initialize the model
model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", trust_remote_code = True)
model.eval()

# Use GPU if available; if not, use the CPU
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def mean_pooling(token_embeddings, attention_mask):
    '''
    Implements the mean pooling method to compute the final embedding vector from the final embedding vectors of the tokens. 
    Taken from the mean_pooling method in the Hugging Face documentation of the multi-qa-MiniLM-L6-cos-v1 model. 

    Arguments: 
        token_embeddings
            tensor which contains the last hidden state, ie. the final embeddings of all the tokens. Shape: (batch_size, seq_len, d_model)
        attention_mask
            used to ignore "fake" tokens that come from padding
    '''
                                    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
 


def embed(model, input_text,max_length = None):      
    """
    Performs a forward pass on the input text prompt using the model, including the final mean pooling and normalization. 
    The function returns the final embedding vector of the input text. Mostly taken from the encode method in the 
    Hugging Face documentation of the multi-qa-MiniLM-L6-cos-v1 model. 

    Arguments: 
        model
            the transformer model used
        input_text
            the input text prompt
    """

    if max_length is None: 
        input_tokens = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
    else: 
        input_tokens = tokenizer(input_text, padding = "max_length", max_length = max_length, truncation=True, return_tensors='pt')

    #Compute token embeddings
    with torch.no_grad(): 
        model_output = model(**input_tokens,output_attentions=True)
 
    # Perform pooling and normalize
    embedding = mean_pooling(model_output[0], input_tokens['attention_mask'])
    return F.normalize(embedding, p=2, dim=1)




## The EmbeddingCircuitAnalyzer class. This class can be used for any desired BERT embedding model from Hugging Face, not just the multi-qa-MiniLM-L6-cos-v1 model. 


class EmbeddingCircuitAnalyzer: 
    """ 
    Class for analyzing circuits in BERT embedding models from Hugging Face (in particular, those models that use BertSelfAttention or BertSdpaSelfAttention). 

    The class contains the following main analysis techniques: 
        -Ablation (modifying activations in a node)
        -Activation patching (patching activations from a clean prompt into a corrupted prompt)
        -Path patching (patching in new activations in a receiver node that has 
        been transmitted along a direct path due to a modification of the activations in a previous sender node)

    Arguments: 
        model 
            the desired instantiated encoder embedding model. 
        embed 
            a function that calculates the embedding vector of an input text. The embed function should have three parameters: model, input_text and max_length. 
            The function returns the final embedding vector of "input_text", after this text has been forward passed into "model". If max_length is not None, then 
            it specifies the maximum number of tokens that will be used from the input text in the forward pass. The embed function should use token padding. 

    """

    def __init__(self,model,embed): 
        self.model = model
        self.embed = embed

    

