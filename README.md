# TransLoRA-GPT
## About
This repository contains the code and resources of the following paper:

Transferable generative pre-trained transformer using low-rank adaptation (TransLoRA-GPT): a framework for predicting battery aging trajectories from charging fragments

## Overview of the framework
This paper proposes an LLM-based framework, named the transferable generative pre-trained transformer using low-rank adaptation (TransLoRA-GPT). The integration of feature engineering reduces the model's dependency on complete charge-discharge data, alleviating the issue of data scarcity. The architectural modifications and low-rank adaptation (LoRA) technique provide the model with a feature-to-SOH mapping capability. Lastly, the deployment process ensures the model's transferability.
The proposed methodology involves three stages: First, a base model, TransLoRA-GPT (base), is developed on a source domain. Second, to leverage the model's transfer learning capabilities, the base model undergoes a lightweight adaptation process using only 10% of the data from the target domain, resulting in the TransLoRA-GPT (transferred) model. Finally, the trained model is evaluated by predicting the SOH degradation trajectory of a test battery over the subsequent 100 cycles using only features from its first 20 cycles.

<p align="center">
<img width="535" height="896" alt="image" src="https://github.com/user-attachments/assets/f4a62341-d7cf-499c-bc8f-19b618a9e37c" />
</p>

## Key Files and Directory Structure:


## License
TransLoRA-GPT is licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0.
