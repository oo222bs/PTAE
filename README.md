# Paired Transformed Autoencoders

Paired Transformed Autoencoders

Last updated: 26 May 2023.

This code has been adapted from Copyright (c) 2018, Tatsuro Yamada <<yamadat@idr.ias.sci.waseda.ac.jp>>

Original repository: https://github.com/ogata-lab/PRAE/

Copyright (c) 2023, Ozan Özdemir <<ozan.oezdemir@uni-hamburg.de>>

## Requirements
- Python 3
- Pytorch
- NumPy
- Tensorboard

## Implementation
Paired Transformed Autoencoders - Pytorch Implementation

## Example
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
- onehotencoder.py: one hot encodes the descriptions
- inference.py: inference time implementation for PTAE
