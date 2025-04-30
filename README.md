# Multi-task Learning Enhanced CNN-LSTM Framework for Weather Spatiotemporal Prediction

## Project Overview
This repository contains the implementation of a multi-task learning enhanced CNN-LSTM framework for spatiotemporal weather prediction, along with occlusion-based explainable analysis. The project was completed as part of ME5311 at the National University of Singapore.

## Key Features
* **Multi-task Learning Architecture**: Simultaneous prediction of atmospheric pressure and temperature fields using a shared CNN-LSTM encoding structure with task-specific decoding components.
* **Spatiotemporal Feature Extraction**: CNN layers extract spatial patterns while LSTM captures temporal dependencies in meteorological data.
* **Occlusion-based Explainability**: Analysis of the importance of individual elements in the weather forecast generation process using a model-independent approach.
* **Improved Generalization**: Demonstrated better generalization performance and training efficiency compared to single-task models, particularly for pressure prediction.

## Dataset
The model is trained on sea level pressure and 2-meter temperature field data. The prediction task involves forecasting the next day's weather based on seven consecutive days of meteorological data.

## Results
The experiments show that:
* The MTL-enhanced framework achieves comparable accuracy to separate CNN-LSTM models
* Significantly improved generalization performance for pressure prediction (11.1% vs 25% generalization error)
* Enhanced training efficiency through shared representation learning
* Better consideration of global and temporal dependencies in the prediction process
./Explainable_result_examples/occlusion_temp_central_y25_x40.png

## Implementation
The code is implemented in Python using deep learning frameworks. All experiments were conducted on an NVIDIA GeForce RTX 4060 Laptop GPU with 8GB memory.

## Limitations
* Since the data is provided by the course, it is uncertain whether it can be shared, so only the relevant code is provided
* Due to computing power limitations, further exploration of GNN-Transformer was not performed; at the same time, for the explanatory part, further analysis of different explanatory strategies such as SHAP was not performed, but the SHAP option is provided in the explanatory code
* Due to the requirements of the course assignment (limited to six pages of text and two pictures) and time constraints, only limited comparison and exploration were carried out

## Repository Structure
```
├── CNN-LSTM_MTL.py                        # Implementation of CNN-LSTM and MTL architectures
├── CNN-LSTM_MTL_explain.py                # Run explainability analysis
├── CNN-LSTM_MTL_model.pth                 # The original result applied in report, under seed(42)
├── ME5311_Project_Report_SunChang.pdf     # The project report
└── README.md                              # Project documentation


```
