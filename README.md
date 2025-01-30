# Species Outlier Detection Using Machine Learning
## File References

The repository includes the following Jupyter notebooks for different models:

1. **Unsupervised models**
    - [unsupervised_models.ipynb](unsupervised_models.ipynb)
    - Uses FinBIF occurrence data from [api.laji.fi](https://api.laji.fi) and several local raster data sets
    - Flags unlike observations as outliers without separate training data

2. **Random Forest (RF)**
    - [random_forest_with_background_samples.ipynb](random_forest_with_background_samples.ipynb)
    - Uses FinBIF occurrence data from [api.laji.fi](https://api.laji.fi) and several local raster data sets
    - Flags unlike observations as outliers from the testing data

3. **Supervised models for bird atlas data**
    - [multiple_models_YKJ_squares.ipynb](multiple_models_YKJ_squares.ipynb)
    - Uses bird atlas data, 10 km x 10 km YKJ squares and environmental data in one preprocessed file
    - Predicts probabilities for each bird for 10 km x 10 km squares
This repository provides implementations for outlier detection in species observation data from Finnish Biodiversity Information Facility (FinBIF) using species distribution modeling (SDM) and machine learning models. The workflow depends on the model, but usually includes data preparation, spatial sampling for background data, environmental enrichment, model training, evaluation, and visualization of results. The models integrate re-classified CORINE land cover data, raster-based environmental variables, and occurrence data to predict species probabilities and identify potential outliers.

## Models Used
The repository includes three different approaches:

1. **Unsupervised models**
    - The file [unsupervised_models.ipynb](unsupervised_models.ipynb)
    - Uses FinBIF occurrence data from api.laji.fi and several local raster data sets
    - Flags unlike observations as outliers without separate training data

2. **Random Forest (RF)**
    - The file [random_forest_with_background_samples.ipynb](random_forest_with_background_samples.ipynb)
    - Uses FinBIF occurrence data from api.laji.fi and several local raster data sets
    - Flags unlike observations as outliers from the testing data
   
3. **Supervised models for bird atlas data**
    - The file [multiple_models_YKJ_squares.ipynb](multiple_models_YKJ_squares.ipynb)
    - Uses bird atlas data, 10 km x 10 km YKJ squares and environmental data in one preprocessed file
    - Calculates a mean results from three different models: Random Forest, Histogram Gradient Boosting and Maximum Entropy.
    - Predicts probabilities for each bird for 10 km x 10 km squares

> **Note:** All models report some false-positive outliers, meaning they may classify valid observations as anomalies.

## Dataset Sources
Due to size constraints (>1 GB), raster datasets are not included in this repository. You can download them from official sources or send me an email:

- **CORINE Land Cover 2018 (25 ha resolution), reclassified**: [SYKE Open Data](https://www.syke.fi/fi-FI/Avoin_tieto/Paikkatietoaineistot/Ladattavat_paikkatietoaineistot)
- **Elevation Model (25m x 25m)**: [National Land Survey of Finland (NLS-FI)](https://paituli.csc.fi/download.html)
- **Monthly Mean Temperature (1961-2023, 10km x 10km resolution)**: [Finnish Meteorological Institute (FMI)](https://paituli.csc.fi/download.html)
- **Monthly Precipitation (1961-2023, 10km x 10km resolution)**: [FMI Open Data](https://paituli.csc.fi/download.html)
- **Forest Biomass Data (m3/ha)**: [Natural Resources Institute Finland (LUKE)](https://kartta.luke.fi/opendata/valinta.html)
- **Coastline Length Calculation (YKJ 10km x 10km grids)**: Calculated from [SYKE Ranta10 dataset](https://www.syke.fi/fi-FI/Avoin_tieto/Paikkatietoaineistot/Ladattavat_paikkatietoaineistot)
- **YKJ 10 km x 10 km squares**: 

All data sets have been preprocessed. Read more: [ML methods for outlier detection](https://github.com/AlpoTurunen/FinBIF_Outlier_Detection/blob/main/ML_methods_for_outlier_detection.pdf)

## Installation

### Dependencies
Install the required dependencies using:
```
pip install -r requirements.txt
```

### Environmental Variables
Store the following credentials in a `.env` file for API access:
```
VIRVA_ACCESS_TOKEN=Access token for sensitive data queries from api.laji.fi. Not open for everyone.
ACCESS_TOKEN=Access token for querying open data from api.laji.fi
ACCESS_EMAIL=Email for sensitive data queries from api.laji.fi
```

## Usage

Usage depends completely on the method used. 

For [random_forest_with_background_samples.ipynb](random_forest_with_background_samples.ipynb) and [unsupervised_models.ipynb](unsupervised_models.ipynb) you can just specify the taxon_if parameter after creating the .env file and run the model. [multiple_models_YKJ_squares.ipynb](multiple_models_YKJ_squares.ipynb) is more complicates as it needs a preprocessed GeoPackage file. You can ask more details or read the file [ML methods for outlier detection](https://github.com/AlpoTurunen/FinBIF_Outlier_Detection/blob/main/ML_methods_for_outlier_detection.pdf).


## Model Outputs
1. **Unsupervised models:** Each observation receives a probability score `[0,1]`, where lower values indicate higher likelihood of being an outlier.
2. **Random Forest:** Each observation in the testing dataset receives a probability score `[0,1]`, where lower values indicate higher likelihood of being an outlier.
3. **Supervised models for bird atlas data:** Each YKJ grid square receives a probability core `[0,1]` where lower values indicate lower breeding likelihood. 


3. **Visualization & Storage:**
    - All results can be interpolated to the continuous raster using [scripts/interpolate_results.py](scripts/interpolate_results.py) script.
    - Results can be also visualized on a map without interpolating.
    - Data can be exported as a **GeoPackage (.gpkg)** file for GIS applications.

![Example Map Visualization](https://github.com/user-attachments/assets/56602a50-2917-43b9-a3c9-bcd12409db51)

## Method Selection Guide


| **Method**                        | **Unsupervised Models** | **Random Forest for All Data** | **Supervised Models for Bird Atlas Data** |
|-----------------------------------|------------------------|--------------------------------|------------------------------------------|
| **Data**                          | All laji.fi observations | All laji.fi observations | Breeding probability indices from the Bird Atlas (YKJ grid) |
| **Purpose**                       | Assigns a probability [0–1] to all species observations to identify outliers based on selected variables. | Assigns a probability [0–1] to each test dataset observation to identify outliers based on selected variables. | Assigns a breeding probability to each species in each YKJ grid cell. |
| **Advantages**                    | No need for separate training or absence data. <br> Easy to use for selected variables. | Well-supported by research. <br> Allows reliability assessment using statistical metrics. | Uses high-quality Bird Atlas data. <br> Produces clear results. <br> Allows reliability assessment using statistical metrics. |
| **Challenges**                     | Difficult to assess model performance without comparison data. <br> Sensitive to parameter choices. <br> Accuracy depends on observation location precision. <br> Differences between models. | Sensitive to parameter choices. <br> Requires generation of absence data, as real absence data is available only for a few species (e.g., butterflies). | Variation within 10 km x 10 km grid cells may be greater than between them. <br> Requires extensive data preprocessing. <br> Predicts breeding probability rather than direct observation reliability. <br> Differences between models. |

