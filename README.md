<p>
	<img src="https://avatars0.githubusercontent.com/u/64927580?s=200&v=4" width="10%" align="left">
</p>



# diff_predictor
Diff_predictor is a package to work in tandem with diff_classifier (https://github.com/Nance-Lab/diff_classifier). 

### Description:
Diff_predictor was created using the coding language, Python, and can be downloaded from the  GitHub repository (https://github.com/dash2927/diff_predictor). This package contains tools for prediction and analysis of multiple particle tracking data of nanoparticles taken from biological tissue imaging.

### Analysis
Data analysis was performed male Sprague-Dawley (SD) rat pups at varying ages, depending on the specific study. These analysis notebooks can be found in the diff_predictor/notebooks folder of the package. Individual slices were plated on 30-mm cell culture inserts in non-treated 6-well plates. Prior to plating, 6-well plates were filled with 1 mL SCM. Slices were incubated in sterile conditions at 37°C and 5% CO2. Predictive analysis tested for pup age and for brain region.

All MPT studies were performed within 24 h of slice preparation. Slices were imaged in a temperature-controlled incubation chamber maintained at 37°C, 5% CO2, and 80% humidity. 30 minutes (min) prior to video acquisition, 40nm polystyrene nanoparticles conjugated with poly(ethylene-glycol) (PS-PEG) were diluted in 1x phosphate-buffered saline (PBS) to a concentration of ~0.014%. Nanoparticles were injected into each slice using a 10 µL glass syringe (model 701, cemented needle, 26-gauge, Hamilton Company, Reno, NV). Videos were collected at 33 frames-per-second and 100x magnification for 651 frames via fluorescent microscopy using a cMOS camera (Hamamatsu Photonics, Bridgewater, NJ) mounted on a confocal microscope. Nanoparticle trajectories and trajectory mean square displacements (MSDs) were calculated via diff_classifier (https://github.com/Nance-Lab/diff_classifier), a Python package developed within Nance Lab.

Diff_predictor uses data taken from Amazon Simple Storage Service (S3) located on Amazon Web Service (AWS). Trajectory data are downloaded as a dataframe using the AWS interface package, boto3, and MSDs and geometric features are calculated using diff_classifier. Using diff_predictor, all required data in an experiment are appended and a tag representing the prediction target is applied to each tracking datapoint. For age-prediction, the target is the age of rat pup from which the data are derived. For region-prediction, the target is the region of the rat pup brain that the data are taken. After the tag is applied, the dataframe is cleaned of all values that were not able to be calculated correctly. The dataframe is then balanced using undersampling so that all target categories have an equal amount of datapoints. 

To prevent data leakage between testing and training sets when creating ensemble decision tree models, the datapoints are binned such that all datapoints that were used in the same statistical feature calculation are in the same bin. The datapoints are then split into training and testing sets based on a desired split. For example, a 0.7 training/testing split would have 70% of all data in the training set and 30% of data in the testing set. For binned feature data, datasets are split in a way that no data within bins are separated. Hyperparameters are then set and cross-validation is applied to the desired model for the training set. A random grid-search is applied at this time to choose which parameters perform the best. At the end of cross-validation, the model with the best performance and parameters will then be selected. This selected model will then be used with the test set that was derived earlier. For ensemble decision trees, SHAP is then used to analyze features.

### References
Curtis, C., A. Rokem, and E. Nance, diff_classifier: Parallelization of multi-particle tracking video analyses. Journal of open source software, 2019. 4(36): p. 989.
Shapley, L.S., A value for n-person games. Contributions to the Theory of Games, 1953. 2(28): p. 307-317.

