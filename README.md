# GLP1 Project

## Overview
Social media data has great potential to uncover novel ASEs unreported by manufacturers and explore patterns of co-occurring ASEs. 
This exploration is vital due to the individual nuances often overlooked in small-scale clinical trials.
We collected 11,185 $\mathbb{X}$ posts, 489,529 Reddit posts, and 13,491 PubMed publications related to GLP-1 receptor agonists. 
By analyzing textual content using a Named Entity Recognition (NER) technique, we identified ASEs on social media that were not reported by manufacturers.
Next, we constructed an ASE-ASE network, clustered ASEs with similar effects into groups, and estimated the frequency of unknown ASEs by training a Graph Convolutional Network model.
Our data analytics approach uncovered 21 potential ASEs on social media, such as irritability and numbness, beyond the knowledge of existing pharmacovigilance data.
Furthermore, our novel computational method successfully distinguished between frequent and infrequent ASEs (F1-score 0.79; AUC 0.82).
Our knowledge-discovery approach can be applied to any drug discussed on social media to identify novel ASEs and estimate their frequencies.
}

# Repository Contents
## Papaer LateX Documents

`/LaTex/sn-article.tex` - main manuscript.

`/LaTex/sb-bibluigraphy.bib` - refecences of the manuscript.

## Running the Code

`.env.example` - variable and credentials needed to run the code.

`ASE_ASE_Network.R` - code for building the ASE-ASE network.

`GNN` - train a CGN model to classify edges in the ASE-ASE network as frequent or not.

`Reddit` - collect and analyze Reddit data.

`X.py` - collect and analyze X (formerly Twitter) data.

`pubmed` - collect and analyze PubMed data.

`sentiment_analysis` - analyze sentiment in Reddit and X.

`plots.py` - create plots of long-tailed ASE distribution and barplot of ASE frequency category/

`requirements.txt` - required python packages to run the code.

## Miscellaneous
Please send any questions you might have about the code and/or the algorithm to...



## Citing our work
If you find this code useful for your research, please consider citing us:
```
@article{GLP1RA,
  title     = {Social Media Analytics for Knowledge Discovery Beyond Standard Pharmacovigilance: Uncovering Side Effects of GLP-1 Receptor Agonists},
  author    = {},
  journal   = {},
  volume    = {},
  number    = {},
  pages     = {from page– to page},
  year      = {2024}
}
