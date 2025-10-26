# A-Lightweight-Learning-Model-for-Green-OFDM-Signal-Classification
The simulation code of "A Lightweight Learning Model for Green OFDM  Signal Classification"
## Usage

- **main.py** is the primary entry point of the project.  
- You can specify different training methods by importing and calling corresponding modules in **main.py**.  
  For example, you can execute simulations with different approaches such as `train_ml_ls.py` by declaring and running it within `main.py`.
- Curve GOSD corresponds to `train_xgb.py` (the DFT setting can also be modified in this code).
- Curve DNN corresponds to `train_DNNt.py`.
- Curve ML w/ Perfect CSI corresponds to `train_ml_ideal.py`.
- Curve ML w/ LS CSI corresponds to `train_ml_ls.py`.
## Utilities

- **utils.py** includes a collection of helper functions.  
  Some of them may be redundant, and users can freely remove or modify unnecessary parts based on their preference.
