import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm as tqdm_notebook
from sklearn.ensemble import IsolationForest
from scipy.stats import norm
import re
from joblib import dump, load

# scrap imports
from pprint import pprint


class ColumnMeta:
    # assigning meta data for columns
    def __init__(self, **kwarg):
        self.name = kwarg['name']
        self.dtype = kwarg['dtype']
        self.gauss_param = dict()

    def summary(self):
        print("__________________________")
        print("Column parameters: ")
        print("Column name  :", self.name)
        print("Data type    :", self.dtype)
        print("Gaussian para:", self.gauss_param)


class NumericOutlier:
    def summary(self, columns):
        for c in columns:
            if np.issubdtype(self.columns[c].dtype, np.number):
                self.columns[c].summary()

    @staticmethod
    def fit_gauss(data):
        mean, std = norm.fit(data)
        return {"mean": mean, "std": std, 'sensitivity': 2}

    @staticmethod
    def is_gauss_anomaly(col, val, param):
        # print(val)
        if abs(val - param['mean']) > param['sensitivity'] * param['std']:
            return True
        else:
            return False

    @staticmethod
    def column_in(obj, col):
        pos = 0
        for x in obj:
            if col == x['column']:
                return pos
            pos += 1
        return -1

    def multivariate_normal(self, x):
        """pdf of the multivariate normal distribution."""
        d = self.Multivariant_Gauss_param['d']
        mean = self.Multivariant_Gauss_param['mean']
        covariance = self.Multivariant_Gauss_param['cov']
        x_m = x - mean
        return (1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance))) *
                np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

    def fit_isolation_forest(self, data):
        clf = IsolationForest(random_state=0).fit(data.dropna())
        return clf

    def row_filtering(self, data):
        query_data = data[self.numeric_list].dropna().to_numpy()
        anomaly_score = dict()
        print("computing isolation forest scores....", end='')
        anomaly_score['isolation_forest'] = self.isolation_forest_clf.score_samples(query_data)
        print('completed!')

        print("computing multivariant gaussian scores..")
        probs = []
        for x in tqdm_notebook(query_data, desc='rows:'):
            probs.append(self.multivariate_normal(x))
        anomaly_score['multivariant_gauss'] = np.array(probs)
        # calculate net anomaly score
        net_anomaly = np.zeros((query_data.shape[0]), np.float32)
        for key in self.row_level_ensemble_weight.keys():
            # rescale score
            scaled_score = (anomaly_score[key] - anomaly_score[key].min()) / (
                    anomaly_score[key].max() - anomaly_score[key].min())
            weighted_score = scaled_score * self.row_level_ensemble_weight[key]
            net_anomaly += weighted_score
        anomaly_list = np.argsort(net_anomaly)[:int(self.n_percent * net_anomaly.shape[0])]
        # using Multivariant Normal distribution
        return anomaly_list, [net_anomaly[i] for i in anomaly_list]

    def Query(self, data):
        anomalies = dict()
        # detect outlier pipline
        anomaly_list, anomaly_score = self.row_filtering(data)
        anomalies['rows'] = anomaly_list.tolist()
        anomalies['scores'] = anomaly_score
        result = dict()
        result['anomalous_rows_score'] = dict()
        for i in range(len(anomalies['rows'])):
            result['anomalous_rows_score'][str(anomalies['rows'][i])] = str(anomalies['scores'][i])
        result['row_col'] = self.compute_columnar_anomaly(data, anomaly_list)
        result['gaussian_parameters'] = dict()
        for col in self.numeric_list:
            result['gaussian_parameters'][col] = {str(key): str(value) for key, value in self.columns[col].gauss_param.items()}
        return result

    def compute_columnar_anomaly(self, data, anomalous_rows):
        row_cols_pairs = defaultdict(list)
        for row in anomalous_rows:
            temp = []
            # anomalous_cols = []
            # gauss_plots = []
            for c in self.numeric_list:
                b = self.is_gauss_anomaly(c, data.loc[row, c], self.columns[c].gauss_param)
                if b:
                    # gauss_plots.append(plot)
                    temp.append(c)
            if len(temp) == 0:
                print("No Anomaly found at Row: ", row)
            else:
                print("Anomaly found at following columns for row: ", row, end=', ')
                print(temp)
                row_cols_pairs[row] = temp
                # print_nplots(gauss_plots)
        return row_cols_pairs

    def __init__(self, data, parameters):
        self.columns = dict()
        self.numeric_list = []
        # assigning weights to algorithm
        self.row_level_ensemble_weight = dict()
        self.row_level_ensemble_weight['isolation_forest'] = 0.5
        self.row_level_ensemble_weight['multivariant_gauss'] = 0.5
        total_weight = 0
        for algo in parameters['algorithm_weight'].keys():
            total_weight += parameters['algorithm_weight'][algo]
            self.row_level_ensemble_weight[algo] = parameters['algorithm_weight'][algo]
        total_weight += ((len(self.row_level_ensemble_weight.keys()) - len(parameters['algorithm_weight'])) * 0.5)
        for algo in self.row_level_ensemble_weight.keys():
            self.row_level_ensemble_weight[algo] /= total_weight
        pprint(self.row_level_ensemble_weight)
        # top 1% anomalous data
        self.n_percent = parameters['percentage']
        # parameters for multivariant gauss distribution
        self.Multivariant_Gauss_param = dict()
        # setting column sensitivity
        if len(parameters['numeric_columns']) > 0:
            # add given list to numeric columns list
            for c in data.columns:
                self.columns[c] = ColumnMeta(name=c, dtype=data[c].dtype)
                array_position = self.column_in(parameters['numeric_columns'], c)
                if array_position > -1:
                    col = np.array(data[c])
                    self.columns[c].gauss_param = self.fit_gauss(col[~np.isnan(col)])
                    if parameters['numeric_columns'][array_position]['sensitivity'] == 'default':
                        self.columns[c].gauss_param['sensitivity'] = parameters['default_sensitivity']
                    else:
                        self.columns[c].gauss_param['sensitivity'] = parameters['numeric_columns'][array_position]['sensitivity']
                    self.numeric_list.append(c)
                    # self.columns[c].summary()
        else:
            # identify numeric columns
            for c in data.columns:
                self.columns[c] = ColumnMeta(name=c, dtype=data[c].dtype)
                # print(c)
                if np.issubdtype(self.columns[c].dtype, np.number) and len(re.findall('ID', c)) == 0 and len(
                        re.findall('type', c)) == 0:
                    col = np.array(data[c])
                    self.columns[c].gauss_param = self.fit_gauss(col[~np.isnan(col)])
                    self.columns[c].gauss_param['sensitivity'] = parameters['default_sensitivity']
                    self.numeric_list.append(c)
                    # self.columns[c].summary()

        print("Running Isolation forest......", end='')
        self.isolation_forest_clf = self.fit_isolation_forest(data[self.numeric_list].dropna())
        print("completed!")

        print("Running Multivariant Gaussian distribution.....", end='')
        self.Multivariant_Gauss_param['d'] = len(self.numeric_list)
        self.Multivariant_Gauss_param['mean'] = data[self.numeric_list].mean().to_numpy()
        self.Multivariant_Gauss_param['cov'] = data[self.numeric_list].cov().to_numpy()
        print("completed!")


if __name__ == '__main__':

    if len(sys.argv) > 2:
        detect = load(sys.argv[1])
        dataset = pd.read_csv(sys.argv[2])
    elif len(sys.argv) == 2:
        config_json = sys.argv[1]
        with open(config_json) as file:
            parameters = json.load(file)
        pprint(parameters)
        dataset = list()
        for x in parameters['files']:
            dataset.append(pd.read_csv(x))
        dataset = pd.concat(dataset)
        numeric_columns = parameters['numeric_columns']
        sensitivity = parameters['default_sensitivity']
        detect = NumericOutlier(dataset, parameters)
    print(detect.numeric_list)
    result = detect.Query(dataset)
    print(result)
    # result['scores'] = [str(i) for i in result['scores']]
    result['row_col'] = {str(key): value for key, value in result['row_col'].items()}
    # pprint(json.dumps(result))
    with open('results.json', 'w') as json_file:
        json.dump(result, json_file)
    dump(detect, 'numeric_outlier_detector.joblib')

