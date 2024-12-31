# Understanding and Improving Representation Learning in the Presence of Shortcuts

[![Paper](https://img.shields.io/badge/Paper-007ACC?style=for-the-badge&labelColor=007ACC)](https://drive.google.com/file/d/1kUKYPgOuO7L0O2zEAoE1sgTRpFLr8Cp4/view?usp=sharing)
[![Drive Folder](https://img.shields.io/badge/Drive_Folder-007ACC?style=for-the-badge&labelColor=007ACC)](https://drive.google.com/drive/folders/1T_b7RTp3zTHDRM1Z27c7m__Smff6FWpV?usp=sharing)

## Abstract 
Shortcuts are decision rules that exploit spurious correlations between the input attributes and labels that hold for the majority of the training examples. Consequently, they lead to worse performance on minority data groups where the spurious correlations do not hold. Models fine-tuned with Empirical Risk Minimization (ERM) have been observed to struggle with predictions on out-of-distribution (OOD) test sets, where the proportion of data groups differs from the one seen by the model during training, due to the model's reliance on shortcuts. In response, previous research has proposed various modifications to the loss function to re-weight the contribution of training examples; however, the effect of these modifications on the models' internal representations and decision-making processes is not well understood. In this thesis, we develop Competing Rules Hypothesis (CRH), which describes the model's decision-making process as a competition between an intended rule and a shortcut rule, as a framework to understand how models implement simple shortcuts. Building on CRH, we propose Representation Shift (reprshift), which surgically modifies a single layer inside the network to systematically shift the representations of examples with shortcuts, as an interpretability-based approach to shortcut mitigation. 

Our experiments are divided into two parts. In the first part, we inspect the internal representations learned by models fine-tuned with ERM and four existing loss function-based shortcut mitigation methods. Using natural language inference (MultiNLI) and toxicity detection (CivilComments) datasets, we (1) compare the representations learned by different loss functions using Centered Kernel Alignment (CKA), (2) probe the representations for information about the shortcut attributes, and (3) investigate how the classifier layers use the information from the earlier layer representations to make predictions. In the second part, we conduct causal intervention experiments to understand how an ERM-trained model implements a shortcut rule and find suggestive evidence for CRH. Finally, we demonstrate that reprshift can be used to substantially improve worst-group performance on MultiNLI.

## Background
![Alt text](assets/background/distribution-shifts.png)
![Alt text](assets/background/loss-function-based-approaches.png)
![Alt text](assets/background/new-perspective.png)

## Empirical Evidence for Competing Rules Hypothesis
![Alt text](assets/crh/hypothesis.png)
### A. Constructive Interference
![Alt text](assets/crh/constructive-int.png)
### B. Destructive Interference
![Alt text](assets/crh/destructive-int.png)
### C. Narrow Channels
![Alt text](assets/crh/narrow-channels.png)

## Representation Shift


## Inspecting Representations

## Training Dynamics

## Organization
### Notebooks Folder
Most of the code for our experiments can be found under "notebooks" folder <br>
The notebooks are divided into 3 parts: Data, Training, and Analysis notebooks <br>
**Data Notebooks:**  used to inspect the datasets <br>
**Training Notebooks:** used to train our models <br>
**Analysis Notebooks:** used for our experiments <br>
The analysis notebooks contain the code for our experiments <br>
We ran all our experiments using the GPUs accessed via Google Colab.

### Reprshift Folder
reprshift contains the code for some of our models and is based on Subpopbench: https://github.com/YyzHarry/SubpopBench <br>
Our implementations of the Loss function based methods and evaluation pipeline, are also based on the implementations from Subpopbench:  https://github.com/YyzHarry/SubpopBench

### transformerlens_2 Folder
Transformer Lens is the library we use to inspect the model internals. We made modifications to the code to adapt it for our purpose. The modifications are minor and only serve the purpose of converting the HookedEncoder from MLM to Classification. Specifically, we ran test to understand the differences and changed the stucture of the MLM head. This modified version of the library is in transformerlens_2 folder.  <br>
The original repository can be found here: https://github.com/TransformerLensOrg/TransformerLens
We also use this library's functionality to run activation patching experiments. 

### Other
We use the code from https://github.com/jayroxis/CKA-similarity/tree/main for our CKA experiments
