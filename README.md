# Circuits-INTED-2025

We implement and apply simple, established methods in Transformer Mechanistic Interpretability (ablation, activation patching and path patching) to localize and describe model features and circuits. Using these, we locate an attention head which seems to respond strongly to color-words (CE1). We also explore visualisation libraries like BertViz (encoders) and CricuitsVis (decoders), and interpretability libraries like Captum and TransformerLens. Using input attribution from Captum, we seek to i) find model misconceptions on physics questions, and ii) compare the embedding space structure of different embedding models, though the results are disappointing (CE2). TransformerLens seems a promising library for further study…. (CE3) TBD

A complete overview of the project can be found in our report, including theoretical background and general discussion of Mechanistic Interpretability with reference to central literature. The subprojects are described in detail in their respective computational essays. 

## Project Structure

### `Deliverables/`
Main directory, containing the project deliverables
    
**`Computational essays/`**

Contains the following computational essays, divided into three parts for readability and environment compatibility.
    
- `CE1_embedding_models` : A thorough run-through of our work on BERT encoder embedding models, showing central methods such as ablation, activation patching and path patching. It concludes with the discovery of a possible color-detection head.
    - Datasets FCI.json and test.tok.json are used in CE1 and explained there.
- `CE2_Attribution` : Using input attribution from Captum, we seek to i) find model misconceptions on physics questions, and ii) compare the embedding space structure of different embedding models, though the results are disappointing. We also briefly present encoder visualisation with BertViz.
- `CE3_Decoder` : We demonstrate finding induction heads in small-scale models, activation patching for GPT-style models and visualizing prediction developments with logit lens. 

- `Environments/`
    Contains conda environments for running each computational essay. (For the most part, one could use any environment with standard packages for ML (with Torch) plus Captum, BertViz, TransformerLens and CircuitsVis, though there are some annoying version dependencies in CE2 which the provided environment solves.)

- `Code files\` 
    Contains a .py file with a class implementation of the interventions from CE1, as well as a documentation notebook demo-ing its use. Together with the class doc-strings, this is hopefully sufficent documentation to permit external use.


**`Report/`**

Contains the report in pdf- and word-format.
- `Circuits.pdf` : Note that this report is *very* long. **We suggest reading the abstract, then skipping to the overview (Section 2), before reading the first few paragraphs under each to each subsection in Applications (First paragraphs of section 4.1 and 4.2 and 4.3)**. This should give enough context that the reader will be able to navigate to the subsections of interest in the theory and applications section. The introduction and discussion are written to be general and should not require that the reader is familiar with the theory section nor details from the applications section.
- `Circuits.docx` : Report in editable format.
- `figures/`
Contains figures if relevant.


### `Supplementary/`
This folder contains non-deliverables. While we removed messy test files, there might be some interesting extra stuff?


## Overview / takeaways

Future work / overview - either here or in 