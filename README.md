# IG-RL : Inductive Graph Reinforcement Learning for Massive-Scale Traffic Signal Control
This repository contains code provided alongside the paper : Inductive Graph Reinforcement Learning for Massive-Scale Traffic Signal Control
https://ieeexplore.ieee.org/document/9405489

https://arxiv.org/pdf/2003.05738.pdf

Also check out the new model-based inductive model (MuJAM : Joint Action Modeling with MuZero) which builds upon IG-RL: 
https://arxiv.org/pdf/2208.00659.pdf

'requirements.txt' lists required dependencies
'code_structure.png' illustrates the structure of the code (for the training of IG-RL). Please note that it was designed with a '2-gpus setting' in mind, as illustrated.

This repository contains pre-trained models and parameters as well as config files ('.ini files') required to re-run(/train/evaluate) experiments included in the IG-RL paper. 

To create your own experiment(s), you can proceed as follows : 

1) Create a folder/arborescence under the config folder (e.g. config/binary/New_Exp/)
2) Under this new folder, create one or multiple .ini files with your own settings/hyperparameters. Multiple examples of such .ini files can be found. For instance, config/binary/GCN/Q_L/Train/BINARY_GEN_GCN_IQL.ini

- Note : There are many settings/hyperparameters in complete .ini config files. settings.ini enumerates the most important/interesting settings (with their descriptions). 
- You can identify the settings you're interested in using settings.ini and look for this setting in a complete .ini file to modify it accordingly.

3) Create one last .ini file at the same location, pointing toward all the .ini files you intend to include in a given run. Multiple examples of such .ini files can be found. For instance, config/binary/GCN/Train/Train_all.ini
4) Execute : $python main.py --config-dir={path to the last .ini file mentioned in the previous step}

'current_schedule.pkl' is automatically generated to keep a list of .ini files/experiments which are to be executed during the current run

Results are stored alongside the .ini files (following the arborescence you defined in step 1)

Results typically include tensorboard logs :
  - For experiments involving training, these logs include training metrics (losses, reward, etc.)
  - For experiments involving evaluation, these logs include aggregated performance metrics (queue_lengths, delays, CO2 emissions, etc.)
For experiments involving evaluations, .pkl files include detailed per-trip information (delay, duration, etc.)


Note : We are currently working on a new (improved) version of the code for IG-RL. A link will be added to this repository as soon as it becomes available. 
