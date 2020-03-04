# integrated_gradients_regression
# Written Assignment 2
## Using Integrated Gradients to Explain Neural Network predictions
### Dependencies
All of the work was done in the conda environment `assignment2_env.yml`
The Boston housing dataset was loaded using scikit-learn, the Concrete dataset is in the folder `concrete/yeh-concret-data/Concrete_Data_Yeh.csv`
The `proj2_base.py` file contains the data loader functions that also preprocess the data so that each feature has 0 mean and unit variance 
### Training the network 
To train the networks there are separate folders and files for each network:
* boston shallow: `boston/shallow/bost_shallow_test.py`
* boston deep: `boston/deep/bost_deep_test.py`
* concrete shallow: `concrete/shallow/conc_shallow_test.py`
* concrete deep: `concrete/deep/conc_deep_test.py`
To create the learning curve plots the scripts ending `analysis.py` in each folder can be used. 
The folders also contain the binary files with the rms errors used for the plots
### Attributions and analysis
The file called `attribute.py` contains the script to calculate integrated gradietns
Do different statistical analysis on them and create the plots in the paper
