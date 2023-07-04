from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def get_sorting_key(result):
    # print(get_total_loss(result), -get_best_rsquared(result))
    return get_total_loss(result), -get_best_rsquared(result)


def get_total_loss(result):
    return result.test_results[0] + result.test_results[1] + result.test_results[2]


def get_best_rsquared(result):
    return result.r_squared[0]


class ModelResult:
    def __init__(self, model, test_orig, test_pred, train_orig, train_pred):
        self.test_orig = test_orig
        self.test_pred = test_pred
        self.train_orig = train_orig
        self.train_pred = train_pred
        self.model = model
        self.test_results = self.evaluation(test_orig, test_pred)
        self.train_results = self.evaluation(train_orig, train_pred)
        self.mae = self.test_results[0], self.train_results[0]
        self.mse = self.test_results[1], self.train_results[1]
        self.rmse = self.test_results[2], self.train_results[2]
        self.r_squared = self.test_results[3], self.train_results[3]

    def evaluation(self, y, predictions):
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r_squared = r2_score(y, predictions)
        return mae, mse, rmse, r_squared
