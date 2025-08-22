# Circuits-INTED-2025

We implement and apply simple, established methods in Transformer Mechanistic Interpretability (ablation, activation patching and path patching) to localize and describe model features and circuits. Using these, we locate an attention head which seems to respond strongly to color-words. (CE1)

We also explore visualisation libraries like BertViz (encoders) and CricuitsVis (decoders), and interpretability libraries like Captum and TransformerLens. Using input attribution from Captum, we seek to i) find model misconceptions on physics questions, and ii) compare the embedding space structure of different embedding models, though the results are disappointing. (CE2)

TransformerLens seems a promising library for further study…. (CE3) TBD

A complete overview of the project can be found in our report, including theoretical background and general discussion of Mechanistic Interpretability with reference to central literature. The subprojects are described in detail in their respective computational essays. 

## Project Structure

### `Deliverables/`
Main directory, containing the project deliverables
    
    - **`Computational essays/`**
        Contains the following computational essays, divinded into three parts for readability and environment compatibility.
        - `Overview` : A brief overview of the essays to come (perhaps with the overview section from the report? otherwise, might cut and add the one-line-descriptions here instead.)
        - `CE1_Encoder` : 
        - `CE2_Attribution` : 
        - `CE3_Decoder` : 
        
        - `Environments/`
            Contains conda environments for running each computational essay. (For the most part, one could use any environment with standard packages for ML (with Torch) plus Captum, BertViz, TransformerLens and CircuitsVis, though there are some annoying version dependencies in CE2 which the provided environment solves.)
    - **`Report/`**
        Contains the report in pdf-format.
        - `report_name` :

### `Supplementary/`
    This folder contains non-deliverables. While we removed messy test files, there might be some interesting extra stuff?


## Overview / takeaways

Future work / overview - either here or in 