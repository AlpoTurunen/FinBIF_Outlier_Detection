# Species Outlier Detection Using Random Forest

This repository provides two Python implementations for outlier detection from species observation data using species distribution modeling (SDM) and machine learning models (Random Forest & three unsupervised models). The workflow includes data preparation, spatial sampling for background data, environmental enrichment, model training, evaluation, and visualization of results. The model integrates CORINE data, raster-based environmental variables, and occurrence data to predict species probabilities.

## How to use

### Dependencies

Install the dependencies using:
```
pip install -r requirements.txt
```

### Enviromental variables
And store the following credentials in a .env file for API access:
```
VIRVA_ACCESS_TOKEN: Access token for querying sensitive occurrence data from api.laji.fi.
ACCESS_TOKEN: Access token for querying open data from api.laji.fi.
ACCESS_EMAIL: Email for sensitive data queries from api.laji.fi.
```

### Usage

1. Modify the ```taxon_id``` parameter for your target species or taxon
2. Select features you want to use. All features are not always suitable for all species. For example, ```tree_vol``` can be suitable for flying squirrels but not for fishes. Edit the row: 

```
feature_columns = ['x', 'y', 'Urban', 'Park', 'Rural', 'Forest', 'Open forest', 'Fjell', 'Open area', 'Wetland', 'Open bog', 'Freshwater', 'Marine', 'dem', 'nvdi', 'rain', 'temp', 'tree_vol', 'gathering.conversions.dayOfYearBegin']
```

### Models' outputs
1. Predicted probabilities [0,1] and classifidications (0 for outliers, 1 for inliers) for each observation. Note that all models report false-positive outliers, meaning that models always find outliers even from the data that doesn't have any real outliers.
2. Predicted probabilities are visualised to the map and stored to a GeoPackage file. 