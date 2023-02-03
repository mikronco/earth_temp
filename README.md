# AI for Climate Change 

ML models can play an important role in tackling several problems connected with the effects of climate change. This reporisory contains some simple examples and applications of ML to toy datasets in the fields of weather hazards, forecasting, land monitoring and temperature projections. It can be used as an introductory and self-sufficient course on the most popular techniques for data analysis and statistical learning, ranging from linear regression to convolutional neural networks, with a focus on weather, climate and satellite data. 

![CC](imgs/cc_img.jpg)

Only basic knowledge of statistics is assumed. Some previous experience with coding in python would be useful but is not mandatory. At the end of the course the student will have hands-on practice with the main python libraries for ML and DL and an overview of relevant problems in the climate domain which can be addressed with ML. 

## Description

The course is organized in eight notebooks, each of them covers a different topic and task: 

* Forecasting global temperatures and carbon dioxide emissions on a yearly scale. 
* Detecting burned areas using meteorological variables. 
* Daily predictions of rain occurrence. 
* Detecting burned areas using meteorological variables. 
* Classifying storms. 
* Daily predictions of temperatures from time series of weather data. 
* Detecting burned areas using meteorological variables. 
* Forecasting the ENSO index. 
* Forecasting sea ice extension. 
* Detecting vegetation cover with satellites. 



## Dependencies

The following packages are needed to run the notebooks: 

* python 3.6
* numpy
* pandas
* matplotlib
* scipy
* scikit-learn
* notebook
* nb_conda_kernels
* xarray
* dask
* netCDF4
* bottleneck
* cartopy
* seaborn
* lightgbm
* tensorflow=2.4
* keras

It is recommended to create the environment by executing the command: conda env create -f config.yml 
