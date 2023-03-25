# Machine_Learning_Project
Project for RPI course Machine Learning 6962

This project aims to learn the nominal behavior of a metal additive manufacturing process through the use of known nominal data and an autoencoder.

Main file is Full_Spec_ML_raw,py
Saved data is located in "count_50_raw_spec.mat"
This mat file contains several cell arrays, which each contain spectrograms of a metal additive manufacturing process, taken from a reduced dimensionality in-situ camera signal.
h_cell contains nominal training data
u_cell contains 22 layers of anomalous data, which are unused at the moment
h_nom contains nominal validation data
h_test contains 3 layers of validation data
u_test contains 3 layers of anomalous data, which are used for validation
meta_cell contains relevant metadata for the above cell arrays, organized as listed above, with two entries per layer - the experiment it belongs to (1,2 or 3) and the corresponding layer index. This is only for image checking if necessary.
