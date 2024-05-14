# Paired Transformed Autoencoders

[Learning Bidirectional Action-Language Translation with Limited Supervision and Testing with Incongruent Input](https://www.tandfonline.com/doi/full/10.1080/08839514.2023.2179167)

Last updated: 14 May 2024.

Copyright (c) 2023, Ozan Özdemir <<ozan.oezdemir@uni-hamburg.de>>

## Requirements
- Python 3
- Pytorch
- NumPy
- Tensorboard

## Implementation
Paired Transformed Autoencoders (PTAE) - Pytorch Implementation

## Training Example
```
$ cd src
$ python main_ptae.py
```
- main_ptae.py: trains the PTAE model
- ptae.py: defines the PTAE architecture
- channel_separated_cae: defines the channel separated CAE
- crossmodal_transformer: defines the Crossmodal Transformer
- standard_cae: defines the standard CAE
- config.py: training and network configurations
- data_util.py: for reading the data
- inference.py: inference time implementation for PTAE
- inference_conflict.py: inference time implementation of conflict experiments for PTAE

## Citation

**PTAE**
```bibtex
@Article{OKWLHBW23, 
 	 author =  {Özdemir, Ozan and Kerzel, Matthias and Weber, Cornelius and Lee, Jae Hee and Hafez, Burhan and Bruns, Patrick and Wermter, Stefan},  
 	 title = {Learning Bidirectional Action-Language Translation with Limited Supervision and Testing with Incongruent Input}, 
 	 booktitle = {},
 	 journal = {Applied Artificial Intelligence},
 	 number = {1},
 	 volume = {37},
 	 pages = {},
 	 year = {2023},
 	 month = {Feb},
 	 publisher = {},
 	 doi = {10.1080/08839514.2023.2179167}, 
 }
```