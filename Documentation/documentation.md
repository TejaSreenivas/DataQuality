Level 1 API (algos)
-------------------

### 1. Simple gaussian

The first evaluation metric is Simple univariant gaussian distribution,  
1. Fuction: fit_gauss(): returns the mean and std for a normal distribution
modeled on particular column 2. is_anomaly(col, val, param): returns (flag,
plot), \* col - column name \* val - value of the column for a row \* param -
dict containing mean and std for that the column

### 2. Isolation Forest

1.  fit_isolation_forest(self, data): applies isolation forest to the data and
    returns the classifier

    -   Isolation forest is applied for the entire data

    -   Isolation forest scores are in the range of -0.5 to +0.5, the greater
        the score the less the anomalous (Opposite of the original paper)

Level 2 API
-----------

### 1. Column level anomaly detection

1.  compute_columnar_anomaly(self, data, anomalous_rows ): Disaplays all the
    anomalous columns for all potential anomalous_rows

    -   data - complete data (or partial data)

    -   anomalous_rows - rows that the function needs to compute columnar
        anomaly

 

**Input arguments**
-------------------

 

### 1. Initialization

When data is fed to the program for the first time -

-   python DataQuality.py \<config json file\>

outputs the following files:

1.  results.json - output file containing outlier rows

2.  numeric_outlier_detect.joblib - this file is the trained model

 

### 2. Query

When you already have a trained model

-   python DataQuality.py \<model.joblib\> \<dataset.csv\>

outputs the following files:

1.  results.json - output file containing outlier rows

2.  numeric_outlier_detect.joblib - this file is the trained model
