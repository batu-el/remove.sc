Link to Paper: https://drive.google.com/file/d/1kUKYPgOuO7L0O2zEAoE1sgTRpFLr8Cp4/view?usp=sharing

# References and Summary
### Notebooks Folder
Most of the code for our experiments can be found under "notebooks" folder <br>
The notebooks are divided into 3 parts: Data, Training, and Analysis notebooks <br>
**Data Notebooks:**  used to inspect the datasets <br>
**Training Notebooks:** used to train our models <br>
**Analysis Notebooks:** used for our experiments <br>
The analysis notebooks contain the code for our experiments <br>

########################
### Reprshift Folder ###
########################
# reprshift contains the code for some of our models and is based on Subpopbench:
# https://github.com/YyzHarry/SubpopBench
# Our implementations of the Loss function based methods and evaluation pipeline,
# are also based on the implementations from Subpopbench:  
# https://github.com/YyzHarry/SubpopBench

################################
### transformerlens_2 Folder ###
################################
# Transformer Lens is the library we use to inspect the model internals.
# We made modifications to the code to adapt it for our purpose.
# The modifications are minor and only serve the purpose of converting the HookedEncoder from MLM to Classification. Specifically, we ran test to understand the differences and changed the stucture of the MLM head.
# This modified version of the library is in transformerlens_2 folder. 
# The original repository can be found here: https://github.com/TransformerLensOrg/TransformerLens
# we also use this library's functionality to run activation patching experiments. 

#############
### Other ###
#############
# We use the code from https://github.com/jayroxis/CKA-similarity/tree/main for our CKA experiments
