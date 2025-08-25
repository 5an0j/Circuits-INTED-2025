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

def slice_or_patch(old_tensor, indices, new_values = None):
    '''
    Slices or patches a PyTorch tensor. The function takes a PyTorch tensor old_tensor as input and does either 
    of two things: 
        
        1: new_values = None. In this case, the function returns a smaller tensor with elements retrieved from old_tensor by slicing
        at the indices specified in the list "indices". 

        2: new_values != None. In this case, the function returns a new tensor that is equal to old_tensor, except that the elements 
        at the indices in the list "indices" are replaced by the elements in the tensor new_values.
    '''
    
    # if an element in "indices" is slice(None), we set the corresponding element equal to 
    # a tensor containing all indices in old_tensor along the desired dimension. If a certain element 
    # in "indices" contains a list of indices, we turn it into a PyTorch tensor. This allows us to implement
    # torch.meshgrid on the list. 
    for i in range(len(indices)): 
        if isinstance(indices[i], slice):
            indices[i] = torch.arange(old_tensor.shape[i])
        if isinstance(indices[i], list): 
            indices[i] = torch.tensor(indices[i])

    # torch.meshgrid() must be used so that we can properly slice the tensor old_tensor. It gives grids of indices 
    # with the same dimension as old_tensor
    index_grids = torch.meshgrid(*indices, indexing="ij")

    if new_values is None: 
        return old_tensor[index_grids]
    else: 
        old_tensor[index_grids] = new_values
        return old_tensor



### Code for doing circuit analysis



## We define some functions and objects that are needed in the EmbeddingCircuitAnalyzer class. These functions and objects are implemented based on 
## the multi-qa-MiniLM-L6-cos-v1 model, and should function as an example of how the arguments to be used in the EmbeddingCircuitAnalyzer class can be instantiated. 

def main(): 

    # initialize a tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", trust_remote_code = True, use_fast = False)

    #initialize the model
    global model
    model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", trust_remote_code = True,attn_implementation="eager")
    model.eval()

    # Use GPU if available; if not, use the CPU
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    global final_layer
    def final_layer(token_embeddings, attention_mask):
        '''
        Implements the final layer (after all the attention-MLP blocks) of the multi-qa-MiniLM-L6-cos-v1 model. The last layer is the mean pooling method used to 
        compute the final embedding vector from the final embedding vectors of the tokens. The function is taken from the mean_pooling method in the Hugging Face documentation of the multi-qa-MiniLM-L6-cos-v1 model. 

        Arguments: 
            token_embeddings
                tensor which contains the last hidden state, ie. the final embeddings of all the tokens. Shape: (batch_size, seq_len, d_model)
            attention_mask
                used to ignore "fake" tokens that come from padding
        '''
                                        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return F.normalize(torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9),p=2,dim=1)
    
    global embed
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
            max_length 
                if provided, it specifies the maximum number of tokens that will be used. 
        """

        if max_length is None: 
            input_tokens = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
        else: 
            input_tokens = tokenizer(input_text, padding = "max_length", max_length = max_length, truncation=True, return_tensors='pt')

        #Compute token embeddings
        with torch.no_grad(): 
            model_output = model(**input_tokens,output_attentions=True)
    
        # Perform pooling and normalize
        embedding = final_layer(model_output[0], input_tokens['attention_mask'])
        return embedding

    global forward_attention_output
    def forward_attention_output(layer_idx):
        '''
        Returns the forward method that computes the attention block activations in the layer with index layer_idx. 
        Can be found in the model architecture by printing model.encoder. 

        Arguments: 
            layer_idx 
                the index of the layer whose attention block activations are computed by the forward method
        ''' 
        return model.encoder.layer[layer_idx].attention.self

    global forward_MLP_output
    def forward_MLP_output(layer_idx):
        '''
        Returns the forward method that computes the MLP neuron activations in the layer with index layer_idx. 
        Can be found in the model architecture by printing model.encoder. 

        Arguments: 
            layer_idx 
                the index of the layer whose MLP neuron activations are computed by the forward method
        ''' 
        return model.encoder.layer[layer_idx].intermediate

    global final_attention_output
    def final_attention_output(layer_idx):
        '''
        Returns the forward method that computes the final vector of activations added to the residual stream 
        after the attention block at the layer at index layer_idx. That is, this function includes the computation of 
        any final layernorm, dense and dropout operations. 
        The required method can be be found in the model architecture by printing model.encoder. 

        Arguments: 
            layer_idx 
                the index of the layer whose vector of attention block activations added to the residual stream are computed by the forward method
        ''' 
        return model.encoder.layer[layer_idx].attention.output

    global final_MLP_output
    def final_MLP_output(layer_idx):
        '''
        Returns the forward method that computes the final vector of activations added to the residual stream 
        after the MLP at the layer at index layer_idx. That is, this function includes the computation of 
        any final layernorm, dense and dropout operations. 
        The required method can be be found in the model architecture by printing model.encoder. 

        Arguments: 
            layer_idx 
                the index of the layer whose vector of MLP activations added to the residual stream are computed by the forward method
        ''' 
        return model.encoder.layer[layer_idx].output




## The EmbeddingCircuitAnalyzer class. This class can be used for any desired BERT embedding model from Hugging Face, not just the multi-qa-MiniLM-L6-cos-v1 model. 


class EmbeddingCircuitAnalyzer: 
    """ 
    Class for analyzing circuits in BERT embedding models from Hugging Face (in particular, those models that use BertSelfAttention or BertSdpaSelfAttention). 

    The class contains the following main analysis techniques: 
        -Ablation (modifying activations in a node)
        -Activation patching (patching activations from a clean prompt into a corrupted prompt)
        -Path patching (patching in new activations in a receiver node that has 
        been transmitted along a direct path due to a modification of the activations in a previous sender node)
        -Logit lens for encoder embedding models (finding how the final embedding vector of a text prompt changes, as a function 
        of the index of the intermediate layer whose output is used to compute the embedding vector)

    Arguments: 
        model 
            the desired instantiated encoder embedding model. 
        final_layer 
            function that implements the final layer (after all the attention_MLP layers) of the embedding model, ie. the layer that computes 
            the final embedding vector of the input text from the last hidden state of all the token embeddings after the last attention-MLP layer. 
            The function returns the final, normalized embedding vector of the text. The function must have two arguments: 
                -token_embeddings
                    tensor which contains the last hidden state, ie. the final embeddings of all the tokens. Shape: (batch_size, seq_len, d_model)
                -attention_mask
                    attention mask tensor with zeros and ones used to ignore "fake" tokens that come from padding
        model_tokenizer
            the tokenizer for the particular embedding model used. 
        embed 
            a function that calculates the embedding vector of an input text. The embed function should have three parameters: model, input_text and max_length. 
            The function returns the final embedding vector of "input_text", after this text has been forward passed into "model". If max_length is not None, then 
            it specifies the maximum number of tokens that will be used from the input text in the forward pass. The embed function should use token padding. 
        forward_attention_output
            Function that returns the forward method that computes the attention block activations in the layer with a given index. If the model 
            implements a layernorm after each attention and MLP block, then this function should return the forward method that is called before the layernorm. 
            The appropriate forward method can be be found in the model architecture by printing model.encoder, once a model has been specified. 
            The function should take in an integer value as an argument, representing the layer index. 
        forward_MLP_output
            Function that returns the forward method that computes the MLP neuron activations in the layer with a given index. If the model 
            implements a layernorm after each attention and MLP block, then this function should return the forward method that is called before the layernorm. 
            The appropriate forward method can be be found in the model architecture by printing model.encoder, once a model has been specified. 
            The function should take in an integer value as an argument, representing the layer index. 
        final_attention_output
            Returns the forward method that computes the final vector of activations added to the residual stream 
            after the attention block in the layer at a given index. That is, this function includes the computation of 
            any final layernorm, dense and dropout operations. The required method can be be found in the model architecture by printing model.encoder. 
            The function should take in an integer value as an argument, representing the layer index. 
        final_MLP_output
            Returns the forward method that computes the final vector of activations added to the residual stream 
            after the MLP in the layer at a given index. That is, this function includes the computation of 
            any final layernorm, dense and dropout operations. The required method can be be found in the model architecture by printing model.encoder. 
            The function should take in an integer value as an argument, representing the layer index. 

    """

    def __init__(self,model,final_layer,model_tokenizer,embed,forward_attention_output,forward_MLP_output,final_attention_output,final_MLP_output): 
        self.model = model
        self.embed = embed
        self.final_layer = final_layer
        self.model_tokenizer = model_tokenizer
        self.forward_attention_output = forward_attention_output
        self.forward_MLP_output = forward_MLP_output
        self.final_attention_output = final_attention_output
        self.final_MLP_output = final_MLP_output

        self.num_layers = len(model.encoder.layer)
        self.d_model = model.config.hidden_size
        self.MLP_intermediate_size = model.config.intermediate_size
        self.num_heads = model.config.num_attention_heads
        self.d_head = self.d_model // self.num_heads

        self.original_forward = forward_attention_output(1).__class__.forward 
        self.handles = []

    def cache_and_patch_attn_head_activations(self,layer_idx,attn_head_output = None,head_idx = slice(None),tokens = slice(None), patch_head_activation = None,alpha = None):
        '''
        Caches and/or patches attention head activations. The function modifies the BertSelfAttention.forward() method by
        including a section where we cache the activations and put them in the attn_head_output dictionary, and/or where we modify 
        the attention head activations. Attention head activations can be modified in two ways - either by specifying a numerical factor alpha that the activations
        should be multiplied by, or by patching new activations into the old ones. In the latter case, the new activations are patched from the patch_head_activation
        tensor. 

        Arguments: 
            layer_idx
                the transformer layer at which we cache and/or patch. 
            attn_head_output
                dictionary which is filled with the cached activations. For a given layer layer_idx, the attention head activations 
                are set as the values mapped to by the key (layer_idx, "attn"). 
            head_idx
                list containing the indices of the attention heads whose activations should be either cached or patched. 
            tokens 
                list containing the indices of the tokens whose activations should be either cached or patched. 
            patched_head_activation 
                tensor containing the new activations that should be patched into the attention head activations. 
            alpha 
                float number that gives the factor that the attention head activations should be multiplied by. If this parameter 
                is specified, then it overwrites the patching.
        '''

        # Modified forward method. Most of the method is taken from BertSelfAttention.forward()
        def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[tuple[tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,cache_position: Optional[torch.Tensor] = None,) -> tuple[torch.Tensor]:
        
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores/math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = F.softmax(attention_scores,dim=-1)

            attention_probs = self.dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)

            #Here we cache activations if attn_head_output has been specified. 
            if attn_head_output is not None: 
                attn_head_output[(layer_idx, "attn")] = slice_or_patch(context_layer.clone(),[slice(None),head_idx,tokens,slice(None)])                                           

            # Here the modify attention head activations. If alpha is specified, we multiply old activations with this value; 
            # otherwise, if patch_head_activation has been specified, we patch its elements into the attention head activations. 
            if alpha is not None: 
                new_activations = slice_or_patch(context_layer.clone(),[slice(None),head_idx,tokens,slice(None)]) * alpha
                context_layer = slice_or_patch(context_layer.clone(),[slice(None),head_idx,tokens,slice(None)],new_activations)                                                         
            else: 
                if patch_head_activation is not None: 
                    context_layer = slice_or_patch(context_layer.clone(),[slice(None),head_idx,tokens,slice(None)],patch_head_activation)

            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            outputs = (context_layer, attention_probs)

            return outputs
    
        #Here, we substiute our new forward method. 
        self.forward_attention_output(layer_idx).forward = MethodType(forward, self.forward_attention_output(layer_idx))

    def cache_and_patch_MLP_activations(self,layer_idx, MLP_output = None, neuron_idx = slice(None), tokens=slice(None), patch_neuron_activation = None,alpha = None):
        '''
        caches and/or patches MLP neuron activations. We create a hook for the forward method of the MLP layer. In the hook function, we cache the activations and 
        put them in the MLP_output dictionary, and/or modify the MLP neuron activations. These activations can be modified in two ways - either by specifying a numerical factor alpha 
        that the activations should be multiplied by, or by patching new activations into the old ones. In the latter case, the new activations are patched from the patch_neuron_activation
        tensor. 

        Arguments: 
            layer_idx
                the transformer layer at which we cache and/or patch. 
            MLP_output
                dictionary which is filled with the cached activations. For a given layer layer_idx, the neuron activations 
                are set as the values mapped to by the key (layer_idx, "MLP"). 
            neuron_idx
                list containing the indices of the neurons whose activations should be either cached or patched. 
            tokens 
                list containing the indices of the tokens whose activations should be either cached or patched. 
            patched_neuron_activation 
                tensor containing the new activations that should be patched into the MLP neuron activations. 
            alpha 
                float number that gives the factor that the neuron activations should be multiplied by. If this parameter 
                is specified, then it overwrites the patching.
        '''

        # Here we define the hook function
        def hook(module, inp, out): 
            orig_output = out.clone()

            # Here we cache neuron activations if MLP_output has been specified. 
            if MLP_output is not None: 
                MLP_output[(layer_idx, "MLP")] = slice_or_patch(orig_output,[slice(None),tokens,neuron_idx])                                           

            # Here the modify MLP neuron activations. If alpha is specified, we multiply old activations with this value; 
            # otherwise, if patch_neuron_activation has been specified, we patch its elements into the neuron activations. 
            if alpha is not None: 
                new_activations = slice_or_patch(orig_output,[slice(None),tokens,neuron_idx]) * alpha
                orig_output = slice_or_patch(orig_output,[slice(None),tokens,neuron_idx],new_activations)                                                       
            else: 
                if patch_neuron_activation is not None: 
                    orig_output = slice_or_patch(orig_output,[slice(None),tokens,neuron_idx],patch_neuron_activation)
                                                                                    

            return orig_output
        
        # Handles to the hook functions are filled in the handles list. 
        self.handles.append(self.forward_MLP_output(layer_idx).register_forward_hook(hook))

    def cache_and_patch_activations(self,node_type, layer_idx, output = None, node_idx = slice(None), tokens = slice(None), patch_activation = None,alpha = None): 
        '''
        caches and/or patches activations. The node_type specifies the desired node at which we wish to cache and/or patch. It can either 
        take the value "MLP" (for MLP neuron activations) or "attn" (for attention head activations).

        Arguments: 
            node_type 
                the type of the node whose activations should be cached or patched. Can be either "MLP" or "attn". 
            layer_idx
                the transformer layer at which we cache and/or patch
            output
                dictionary which is filled with the cached activations. For a node at the given layer layer_idx and type node_type, 
                the neuron or attention head activations are set as the values mapped to by the key (layer_idx, node_type). 
            node_idx
                list containing the indices of the nodes (neurons or attention heads) whose activations should be either cached or patched. 
            tokens 
                list containing the indices of the tokens whose activations should be either cached or patched. 
            patched_activation 
                tensor containing the new activations that should be patched into the old activations. 
            alpha 
                float number that gives the factor that the old activations should be multiplied by. If this parameter 
                is specified, then it overwrites the patching.
        '''

        #turn node_idx and tokens into a list if a single scalar value has been specified
        if not isinstance(node_idx,(list,slice)):
            node_idx = [node_idx]
        if not isinstance(tokens,(list,slice)):
            tokens = [tokens]

        if node_type == "MLP": 
            self.cache_and_patch_MLP_activations(layer_idx = layer_idx,MLP_output = output, neuron_idx = node_idx, tokens = tokens, patch_neuron_activation = patch_activation,alpha = alpha)
        if node_type == "attn": 
            self.cache_and_patch_attn_head_activations(layer_idx = layer_idx,attn_head_output = output, head_idx = node_idx, tokens = tokens, patch_head_activation = patch_activation,alpha = alpha)

    def remove_handles_and_retrieve_original_forward(self): 
        '''
        removes all handles to MLP hook functions and retrieves all forward methods on attention blocks to their
        original forward methods. 
        '''

        # clear handles
        [handle.remove() for handle in self.handles]
        self.handles.clear()

        # retrieve original forward methods
        for i in range(self.num_layers): 
            self.forward_attention_output(i).forward = MethodType(self.original_forward, self.forward_attention_output(i))

    def cache_and_patch_on_many_nodes(self,nodes,new_activations = None,tokens = None,old_activations = None,alpha = None):    
        '''
        caches and/or patches activations of several nodes. The function calls cache_and_patch_activations for every node specified 
        in the "nodes" dictionary. 

        Arguments: 
            nodes  
                dictionary containing information about every node whose activations one wishes to cache or patch. 
                Every key-value pair is of the form (layer_idx, node_type): node_indices, where layer_idx is the layer index, 
                node_type is the type of the node ("MLP" or "attn") and node_indices is a list with the indices of the neurons or 
                attention heads whose activations will be patched or cached. 
            new_activations 
                dictionary with the new activations to be patched for every node in the "nodes" dictionary. The key-value pairs 
                are of the form (layer_idx, node_type): activations. For a node at a given layer layer_idx and node type node_type, the activations tensor 
                provides the new activations to be patched in. 
            tokens 
                list with the token positions at which each node in the "nodes" dictionary should have its activations either cached or patched. Thus, the order 
                of elements in "tokens" and "nodes" must be consistent. 
            old_activations 
                dictionary which is filled with the cached activations. For a given node at the layer layer_idx and with type node_type, the neuron or attention 
                head activations are set as the values mapped to by the key (layer_idx, node_type). 
            alpha 
                float number that gives the factor that the old activations should be multiplied by. If this parameter 
                is specified, then it overwrites the patching.

        '''

        # make an empty tokens and new_activations list if they were not specified - these will be filled up 
        # in the loop below to fit the length of "nodes". 
        if tokens is None: 
            tokens = []
        if new_activations is None: 
            new_activations = {}

        # In the loop below, we register the layer index, node indices and node types in lists 
        # layers, node_indices and node_types, respectively, for every node in "nodes". 
        layers = []
        node_indices = []
        node_types = []
        for location, node in nodes.items(): 
            layers.append(location[0])
            node_indices.append(node)
            node_types.append(location[1])

            # if "tokens" has too few elements, we add slice(None) to it (which means that activations of 
            # all tokens will be either cached or patched for the corresponding node). 
            if len(node_types) > len(tokens): 
                tokens.append(slice(None))

            # likewise, we add None to new_activations if it does not contain the key of the given node (meaning 
            # that no activations will be patched in for this node)
            if not((location[0],location[1]) in new_activations): 
                new_activations[(location[0],location[1])] = None

        #we cache and patch activations for all nodes in "nodes". 
        for i in range(len(layers)): 
            self.cache_and_patch_activations(node_type = node_types[i], layer_idx = layers[i], output = old_activations, node_idx = node_indices[i], tokens = tokens[i], patch_activation = new_activations[(layers[i],node_types[i])],alpha = alpha)

    def ablate(self,nodes, data, tokens = None, ablation_type = "zero", alpha = 0.0):
        '''
        performs an ablation. The ablation is done for all the nodes specified in the "nodes" dictionary. The function can
        perform a zero, mean or resampling ablation. The function returns the embedding of the text batches in "data" after a forward pass 
        with the ablated activations. 

        Arguments: 
            nodes
                dictionary containing information about every node whose activations one wishes to ablate. 
                Every key-value pair is of the form (layer_idx, node_type): node_indices, where layer_idx is the layer index, 
                node_type is the type of the node ("MLP" or "attn") and node_indices is a list with the indices of the neurons or 
                attention heads whose activations will be ablated.  
            data 
                list with batches; thus, each element in the list should be a text one wishes to do a forward pass and ablation on. 
            tokens 
                list with the token positions at which each node in the "nodes" dictionary should have its activations ablated. Thus, the order 
                of elements in "tokens" and "nodes" must be consistent. This list is only relevant for zero ablation. 
            ablation_type 
                string that specifies the type of ablation. It should either take the value "zero", "mean" or "resampling". 
            alpha 
                float number that gives the factor that the old activations should be multiplied by. This is only relevant for the case of zero ablation. 
                Thus, one can use this value to do a "soft" version of zero ablation, where the activations are multiplied by a given number 
                that is not necessarily zero. 
        '''

        # zero ablation
        if ablation_type == "zero": 
            self.cache_and_patch_on_many_nodes(nodes,tokens = tokens, alpha = alpha)

        # mean or resampling ablation
        if ablation_type == "mean" or ablation_type == "resampling": 

            # we first cache the old activations of the specified nodes
            dataset_activations = {}
            self.cache_and_patch_on_many_nodes(nodes, old_activations = dataset_activations)
            self.embed(self.model,data)
            self.remove_handles_and_retrieve_original_forward()

            if ablation_type == "mean": 
                mean_activations = {}

                # here we calculate the average activations over all batches and patch these averages into the activations
                # of the specified nodes 
                for node_location, activations in dataset_activations.items(): 
                    mean_activation = torch.mean(activations, dim = 0,keepdim = True)
                    mean_activations[node_location] = mean_activation.expand([len(data)] + [-1]*(mean_activation.dim()-1))
                self.cache_and_patch_on_many_nodes(nodes, new_activations = mean_activations)
            else: 

                # here we reshuffle the activations of the various batches and patch them into the activations
                # of the unshuffled batches 
                reshuffled_activations = {node: activation_values[torch.randperm(len(data))] for node,activation_values in dataset_activations.items()}
                self.cache_and_patch_on_many_nodes(nodes, new_activations = reshuffled_activations)

        #calculate the embedding with the ablated activations
        ablated_embedding = self.embed(self.model,data)
        self.remove_handles_and_retrieve_original_forward()

        return ablated_embedding
    
    def activation_patch(self,clean_prompt, corr_prompt,nodes,tokens = None): 
        '''
        performs an activation patching from a clean prompt to a corrupted prompt. The function patches in the activations of the clean run
        to the corrupted run at the nodes specified in the "nodes" dictionary and the tokens specified in the "tokens" list. It returns 
        both the L2 and cosine distance between the embeddings of the clean and corrupted prompts (with its activations patched). 

        Arguments: 
            orig_prompt 
                a string with the clean text prompt 
            corr_prompt 
                a string with the corrupted text prompt
            nodes 
                dictionary containing information about every node whose activations one wishes to patch. 
                Every key-value pair is of the form (layer_idx, node_type): node_indices, where layer_idx is the layer index, 
                node_type is the type of the node ("MLP" or "attn") and node_indices is a list with the indices of the neurons or 
                attention heads whose activations will be patched.  
            tokens 
                list with the token positions at which each node in the "nodes" dictionary should have its activations patched. Thus, the order 
                of elements in "tokens" and "nodes" must be consistent. 
        '''

        # in order to be able to compare the activations from the two prompts, the two prompts must have equal numbers of tokens. 
        # in general, we use padding with maximal length to do this. 
        max_tokens = max(len(self.model_tokenizer(clean_prompt)["input_ids"]),len(self.model_tokenizer(corr_prompt)["input_ids"]))

        # cache the activations from the clean run
        orig_activations = {}
        self.cache_and_patch_on_many_nodes(nodes,tokens = tokens, old_activations = orig_activations)
        orig_embeddings = self.embed(self.model,clean_prompt, max_tokens)
        self.remove_handles_and_retrieve_original_forward()

        # patch in the activations to the corrupted run 
        self.cache_and_patch_on_many_nodes(nodes,tokens = tokens, new_activations = orig_activations)
        corr_embeddings = self.embed(self.model,corr_prompt, max_tokens)
        self.remove_handles_and_retrieve_original_forward()

        return orig_embeddings, corr_embeddings, 1 - F.cosine_similarity(orig_embeddings, corr_embeddings), torch.norm(orig_embeddings - corr_embeddings, p=2)
    
    def path_patching(self,sender_node,clean_prompt, receiver_node = None, corr_prompt = None,new_sender_node_activation = None, alpha = None):     #sender_node, receiver_node. ([layer,"MLP/attn"]: node_indices)
        '''
        implements path patching, based on the algorithm by Callum McDougall (which is equivalent to the algorithm given in the 
        Interpretability in the Wild paper (Wang et al., 2022)). The function returns the final embedding vector of the clean prompt 
        before and after having performed the path patching. One can use the function to implement path patching in two ways: 
        
            1: perform a path patching with a clean and corrupt prompt. In this case, the activations in the sender node are cached from the
            corrupt prompt. 
            2: manually alter the activations in the sender node (without the use of a corrupt prompt). 
        
        Path patching works by systematically caching and patching the activations on the receiver and sender node, so
        that one finds how the output embedding of the clean prompt is affected by the contribution of the direct path from the sender to receiver 
        in the corrupted prompt (or the contribution of the direct path to alter the activations in the receiver node due to the manual change to the sender
        node activations). 

        The steps 1, 2 and 3 below refer to the steps of the path patching algorithm specified in the notebook by Callum McDougall. 

        Arguments: 
            sender_node 
                dictionary specifying the sender node. It consists of a single key-value pair of the form (layer_idx, node_type): node_indices, 
                where layer_idx is the layer index, node_type is the type of the node ("MLP" or "attn") and node_indices is a list with the indices of the neurons or 
                attention heads in the sender node.
            receiver_node 
                dictionary specifying the receiver node. It consists of a single key-value pair of the form (layer_idx, node_type): node_indices, 
                where layer_idx is the layer index, node_type is the type of the node ("MLP" or "attn") and node_indices is a list with the indices of the neurons or 
                attention heads in the receiver node. If receiver_node is None, then the receiver node is effectively set to equal the final mean pooling layer
                (ie. we do path patching from the sender node to the final output) 
            clean_prompt 
                a string with the clean text prompt 
            corr_prompt 
                a string with the corrupted text prompt
            new_sender_node_activation 
                dictionary containing key-value pairs with the new activations in the sender nodes (for case 2). 
                The key is the tuple (sender_node_layer_idx, sender_node_type) and the value is a tensor with the new activations. 
            alpha 
                float number that gives the factor that the old sender node activations should be multiplied by (for case 2). Thus, instead of using 
                new_sender_node_activation, one can use this number to alter the sender node activations. 
                
        '''

        if corr_prompt is not None: 
            max_tokens = max(len(self.model_tokenizer(clean_prompt)["input_ids"]),len(self.model_tokenizer(corr_prompt)["input_ids"]))
        else: 
            max_tokens = len(self.model_tokenizer(clean_prompt)["input_ids"])

        #  ------ STEP 1 -----

        #Cache clean attention head activations
        clean_attn_heads = {}
        attn_head_nodes = {(i,"attn"): list(range(self.num_heads)) for i in range(self.num_layers)}

        self.cache_and_patch_on_many_nodes(attn_head_nodes,old_activations = clean_attn_heads)

        orig_embedding = self.embed(self.model,clean_prompt, max_tokens)
        self.remove_handles_and_retrieve_original_forward()

        #Cache corrupt receiver node activations (if the corrupt prompt has been specified)
        if corr_prompt is not None: 
            corr_sender_node_act = {}

            self.cache_and_patch_on_many_nodes(sender_node, old_activations = corr_sender_node_act)                                                      #cache_and_patch_activations(node_type = sender_node_type, layer_idx = sender_location[0],output = corr_sender_node_act)

            self.embed(self.model, corr_prompt, max_tokens)
            self.remove_handles_and_retrieve_original_forward()
        
        # ------- STEP 2 ------

        if receiver_node is not None: 
            freezed_attn_head_nodes = dict(list(attn_head_nodes.items())[list(sender_node.keys())[0][0] + 1:list(receiver_node.keys())[0][0]])
        else: 
            freezed_attn_head_nodes = dict(list(attn_head_nodes.items())[list(sender_node.keys())[0][0] + 1:])

        #Patch clean attention head activations (ie. freeze these activations)
        self.cache_and_patch_on_many_nodes(freezed_attn_head_nodes,new_activations = clean_attn_heads)

        #Patch corrupt receiver node activations, or manually patch new sender node activations
        if corr_prompt is not None: 
            self.cache_and_patch_on_many_nodes(sender_node, new_activations = corr_sender_node_act)
        else: 
            self.cache_and_patch_on_many_nodes(sender_node, new_activations = new_sender_node_activation, alpha = alpha)

        # If the receiver node is not the final mean pooling layer, then do a forward pass on the clean prompt 
        # and cache the receiver node activations. If the receiver node is the final layer, then this 
        # part is not needed (since the output would be unchanged after patching cached final layer activations from a previous run)
        if receiver_node is not None: 
            receiver_node_act = {}
            self.cache_and_patch_on_many_nodes(receiver_node, old_activations = receiver_node_act)
            self.embed(self.model,clean_prompt, max_tokens)
            self.remove_handles_and_retrieve_original_forward()

            # -------- STEP 3 ---------

            # Patch the cached receiver node activations 
            self.cache_and_patch_on_many_nodes(receiver_node, new_activations = receiver_node_act)

        # Final forward pass on the clean prompt with the patched receiver node activations
        new_embedding = self.embed(self.model,clean_prompt, max_tokens)
        self.remove_handles_and_retrieve_original_forward()

        # orig_embedding: before path patching. new_embedding: after path patching. 
        return orig_embedding, new_embedding
    
    def embedding_distance(self,reference_text,text,title,label): 
        '''
        performs logit lens for encoder models. The function takes in two lists of text prompts ("reference_text" and "text") and plots the cosine embedding distance
        between the embedding of a reference prompt in the list "reference_text" and the corresponding intermediate embedding of the element in the list "text" as a function of 
        sublayer in the transformer model. The intermediate embeddings of a forward pass of a text prompt 
        are found by sending the output of a sublayer (attention layer or MLP layer) through the last mean pooling layer. 

        In order to compare text prompts in reference_text and text at the same positions in the two lists, 
        it is important that reference_text and text have the same number of elements. 

        Arguments: 
            reference_text 
                a list with reference text prompts. 
            text 
                a list with text prompts that we calculate intermediate embeddings of. 
            title 
                The title of the plots. 
            label 
                a list with the labels for each graph, corresponding to each pair of text prompts in "reference_text" and "text". 
        '''

        # We need the final output of the attention and MLP layers. In the cache_and_patch_activations function, we found *intermediate* 
        # attention head and MLP neuron activations. Thus, in order to cache the final activations in each layer, we instead 
        # create hooks for the final activations of both MLPs and attention layers. 
        intermediate_act_mlp = []
        intermediate_act_attn = []

        def hook_mlp(module,inp,out): 
            output = out.clone()
            intermediate_act_mlp.append(output)
        def hook_attn(module,inp,out):
            output = out.clone() 
            intermediate_act_attn.append(output)
        
        # The final output of MLP layer i is found from encoder.layer[i].output, whereas the final output of attention layer 
        # i is found from encoder.layer[i].attention.output. Both of these can be found by checking the model architecture with model.encoder
        [self.handles.append(self.final_MLP_output(i).register_forward_hook(hook_mlp)) for i in range(self.num_layers)]  
        [self.handles.append(self.final_attention_output(i).register_forward_hook(hook_attn)) for i in range(self.num_layers)]  
        self.embed(self.model,text) 
        self.remove_handles_and_retrieve_original_forward()

        # We send the activations through the last layer, consisting of a mean pooling and normalization.  
        intermediate_embeddings_mlp = torch.stack([self.final_layer(intermediate_act_mlp[i],self.model_tokenizer(text, padding = True, truncation=True, return_tensors='pt')["attention_mask"]) for i in range(self.num_layers)])
        intermediate_embeddings_attn = torch.stack([self.final_layer(intermediate_act_attn[i],self.model_tokenizer(text, padding = True, truncation=True, return_tensors='pt')["attention_mask"]) for i in range(self.num_layers)])

        reference_embedding = self.embed(self.model,reference_text) 

        # The cosine distance between the embedding of the i'th reference prompt in "reference_txt", and 
        # the intermediate embeddings of the i'th prompt in "text" are calculated and plotted as a function of sublayer number. 
        for i in range(len(text)): 

            # cosine distances for attention sublayers and MLP sublayers respectively                             
            distances_attn = 1 - F.cosine_similarity(reference_embedding[i],intermediate_embeddings_attn[:,i,:],dim=1) 
            distances_mlp = 1 - F.cosine_similarity(reference_embedding[i],intermediate_embeddings_mlp[:,i,:],dim=1)

            distances = [distance for distance_pair in zip(distances_attn,distances_mlp) for distance in distance_pair]                               

            plt.plot(list(range(2*self.num_layers)),distances,label=label[i])
            plt.xlabel("Sublayer #")
            plt.ylabel("Cosine distance")
            plt.title(title)
            plt.grid()
            plt.legend()
        plt.show()

    def heatmap(self,values, title,label,node_type="attn"): 
        '''
        plots a heatmap of values corresponding to each attention head or MLP node. Thus, one can use this function 
        to plot heatmaps of any desired values that one has computed for each attention head or MLP node in a transformer model.

        Arguments: 
            values 
                a two dimensional PyTorch tensor or Numpy array with values for every attention head or MLP node. The first index should 
                represent the layer index, and the second index should represent the attention head index or MLP node index at the given layer. 
            title 
                title of the heatmap. 
            label 
                label for the values on the color bar next to the heatmap.  
            node_type 
                the type of nodes that one wishes to visualize the values for. Must either take the value "attn" (for attention heads) 
                or "MLP" (for MLP nodes). 
        '''

        if node_type == "attn": 
            x_label = "Head"
            node_name = "H"
        if node_type == "MLP": 
            x_label = "MLP node"
            node_name = "N"

        # Plot a heatmap for values at each node (attention head or MLP node)
        fig, ax = plt.subplots(figsize = (8, 6))
        sns.heatmap(values, cmap='viridis', annot=True, fmt=".2g", annot_kws={"size": 8},
                xticklabels=[f"{node_name}{i}" for i in range(values.shape[1])],
                yticklabels=[f"L{i}" for i in range(values.shape[0])],
                cbar_kws={"label": label})
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Layer")
        fig.tight_layout()
        plt.show()


    def intervene_on_many_nodes(self,reference_prompt, prompt, title, node_type, corr_prompt = None, node_is_receiver = True, path_patching_node = None, 
                            intervention_type = "ablation", tokens = None, 
                            alpha = 0.0,num_MLP_nodes = 1): 
        '''
        performs an intervention on each node (attention head or MLP neurons) in a transformer model, computes the relative distance change due to the intervention
        and displays the values in a heatmap. The relative cosine distance change is calculated from the change in 
        distance between the embedding of the text "reference_prompt" and the embedding of the text "prompt" after an intervention on the activations of a forward pass
        on "prompt". 

        The MLP nodes are defined by chunking all neurons into a number of batches specified by the parameter "num_MLP_nodes". The interventions are performed on all 
        neurons within a given batch. 

        Arguments: 
            reference_prompt 
                the reference text that is used to compute the relative cosine distance change. 
            prompt 
                the text prompt whose activations we intervene on. 
            title 
                title of the heatmap
            node_type 
                the type of nodes that one wishes to intervene on. Must either take the value "attn" (for attention heads) 
                or "MLP" (for MLP nodes). 
            corr_prompt 
                the corrupt text prompt that should be used when performing a path patching. The activations from this corrupt 
                prompt are patched in to the activations of "prompt". 
            node_is_receiver 
                If True, a path patching intervention would be performed for every sender node (and the heatmap will show the corresponding 
                relative distance change). If False, the path patching intervention would be performed for every receiver node. 
            path_patching_node 
                the specified node, for use in the case when a path patching is performed. If node_is_receiver is True, this variable specifies 
                the receiver node. If node_is_receiver is False, it specifies the sender node. The node is specified in a dictionary with a single key-value pair, 
                where the key is a tuple with elements (node_layer_idx, node_type) (where node_type is either "attn" or "MLP"). The value is a list with the indices 
                of the attention heads or MLP neurons in the node. 
            intervention_type 
                specifies the type of intervention which is performed. Can take the value "ablation", "activation patching" or "path patching". 
            tokens 
                a list with the indices of the tokens on which the interventions will be performed. 
            alpha 
                in the case of either ablation or path patching, alpha is the numerical factor that either the old activations or 
                sender nodes activations are multiplied by, respectively. 
            num_MLP_neurons
                the number of MLP nodes in each layer, ie. the number of batches that the MLP neurons are split into. 
        '''

        #final embedding vectors of reference text and text prompt. 
        reference_embedding = self.embed(self.model,reference_prompt)
        orig_embedding = self.embed(self.model,prompt)

        # define an array "values" with values of relative distance change for every node, and a list "node_indices" 
        # with the indices of attention heads or MLP neurons for each node at a given layer. 
        if node_type == "attn": 
            values = np.zeros((self.num_layers,self.num_heads),float)
            indices_nodes = list(range(self.num_heads))
        if node_type == "MLP": 
            # Split the MLP neurons into chunks or batches, and specify each node in a list (according to the indices of the neurons within it)
            chunk_size = self.MLP_intermediate_size//num_MLP_nodes
            values = np.zeros((self.num_layers,num_MLP_nodes),float)
            indices_nodes = [list(range(chunk_size*i, chunk_size*(i+1))) for i in range(num_MLP_nodes)]

        # Here we find the relative cosine distance change after an intervention on every node. 
        # The distance changes are filled up in the "values" array.
        for i in range(self.num_layers): 
            for j in range(len(indices_nodes)): 
                
                node = {(i,node_type): indices_nodes[j]}

                # Check the type of intervention and perform it on the activations in "node"
                if intervention_type == "ablation": 
                    intervened_embedding = self.ablate(node,prompt,tokens = tokens, alpha = alpha)
                if intervention_type == "activation patching": 
                    _, intervened_embedding, _, _ = self.activation_patch(reference_prompt, prompt,node, tokens = tokens)
                if intervention_type == "path patching": 
                    if node_is_receiver: 
                        receiver_node = path_patching_node
                        sender_node = node
                    else: 
                        receiver_node = node
                        sender_node = path_patching_node 
                    _, intervened_embedding = self.path_patching(sender_node = sender_node,clean_prompt = prompt, corr_prompt = corr_prompt, receiver_node = receiver_node, alpha = alpha)

                # Compute the relative cosine distance changes and put the value in the array values
                distance_reference_intervened_prompt = 1 - F.cosine_similarity(reference_embedding[0], intervened_embedding[0],dim=0)
                distance_reference_orig_prompt = 1 - F.cosine_similarity(reference_embedding[0], orig_embedding[0],dim=0)
                values[i,j] = (distance_reference_orig_prompt - distance_reference_intervened_prompt)/distance_reference_orig_prompt

        # Visualize the relative distance changes in a heatmap
        self.heatmap(values,title=title,label="Relative cosine distance change",node_type = node_type)


    def visualize_activations(self,title, node_type, text = None,tokens = slice(None),act = None,num_MLP_nodes = 1): 
        '''
        visualizes the internal activations of a forward pass on text inputs. The activations of each attention head or MLP node are visualized in a heatmap. 
        The activations are calculated by computing the norm of the internal activation vectors for each node and token, and then averaged over the tokens 
        and batches specified in the "tokens" and "text" lists respectively. 

        One can either specify the input text(s) and compute the internal activations during a forward pass, or one can specify 
        the activations that one wishes to visualize. The latter option may be useful if one wishes to visualize activations after an intervention. 

        The MLP nodes are defined by chunking all neurons into a number of batches specified by the parameter "num_MLP_nodes". The normed activations are calculated among
        the neurons in each batch. 

        Arguments: 
            title 
                title of the heatmaps (see "title" in the "heatmap" function). 
            node_type 
                the type of nodes whose activations one wishes to visualize. Must either take the value "attn" (for attention heads) 
                or "MLP" (for MLP nodes). 
            text 
                string or list with input text prompt(s), whose activations we wish to visualize.
            tokens  
                list with indices of the tokens whose activations we wish to visualize. 
            act 
                dictionary with the activations that one wishes to visualize, for the second 
                option given above (if one doesn't want to specify text input(s) and do a forward pass on them). For each node, 
                the dictionary must contain a key-value pair. They key is a tuple (layer_idx,node_type) and the value is a tensor
                with the corresponding activations. 
            num_MLP_nodes 
                the number of MLP nodes in each layer, ie. the number of batches that the MLP neurons are split into. 
        '''
        
        # Cache attention head and MLP neuron activations, in the case when text prompt(s) have been specified
        if act == None:
            act = {}
            if node_type == "attn": 
                [self.cache_and_patch_attn_head_activations(i,attn_head_output = act,tokens = tokens) for i in range(self.num_layers)]
            if node_type == "MLP": 
                [self.cache_and_patch_MLP_activations(i, MLP_output = act, tokens=tokens) for i in range(self.num_layers)]
        if text is not None: 
            self.embed(self.model,text)
        self.remove_handles_and_retrieve_original_forward()
        
        # Stack together the activation values from the act dictionary. Also, compute the norm of activation vectors, 
        # and average over tokens and batches
        if node_type == "attn": 
            act = torch.stack(list(act.values()))
            avg_act = torch.norm(act, p = 2,dim=-1).mean(dim=[1,3])
        if node_type == "MLP":
            act = torch.stack([torch.stack(torch.chunk(el,num_MLP_nodes,dim=-1)) for el in act.values()])
            avg_act = torch.norm(act, p = 2,dim=-1).mean(dim=[2,3])

        # Visualize the results in a heatmap
        self.heatmap(avg_act, title = title,label="Normed activations",node_type = node_type)

if __name__ == "__main__": 
    main()
    

    

