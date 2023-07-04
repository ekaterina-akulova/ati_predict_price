import pickle
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from inference.validate import ModelResult, get_sorting_key
from training.steps.clean_data import clean_data
from training.steps.create_model import model_prediction
from training.steps.load_data import load_data
from training.steps.prepare_data import data_split, standardising_data, add_cargo_weight_volume, cargo_range_dummies

warnings.filterwarnings("ignore")


def save_model(model, output_model_path):
    with open(output_model_path, 'wb') as f:
        pickle.dump(model, f)


def pipeline(use_new_model=False, file='', input_model_path='data/model_predict.pkl',
             output_model_path='data/model_predict.pkl'):
    model_results = []
    models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(),
              AdaBoostRegressor(), GradientBoostingRegressor(), XGBRegressor()]
    predictors = ['cargo_weight_volume', 'Dist', 'cargo_range_Low',	'cargo_range_Medium',
                  'cargo_range_High', 'money_rub']
    num_predictors = ['cargo_weight_volume', 'Dist']
    data = load_data(file)
    data = clean_data(data)

    data = add_cargo_weight_volume(data)
    data = cargo_range_dummies(data)
    data = standardising_data(data, num_predictors)
    data = data[predictors]
    x_train, x_test, y_train, y_test = data_split(data, 'money_rub')
    if use_new_model:
        for model in models:
            res_model, x_train_pred, x_test_pred = model_prediction(model, x_train, y_train, x_test)
            model_results.append(ModelResult(res_model, y_test, x_test_pred, y_train, x_train_pred))
        best_result = sorted(model_results, key=get_sorting_key)[0]
        print("Best model is ", best_result.model, "\nResults on train data (mae, mse, rmse, r_squared) = ",
              best_result.train_results, "\nResults on test data (mae, mse, rmse, r_squared) = ",
              best_result.test_results)
        save_model(best_result.model, output_model_path)
    else:
        model_pickle = open(input_model_path, "rb")
        model = pickle.load(model_pickle)
        print(model)
        # print("Best model is ", model.model, "\nResults on train data (mae, mse, rmse, r_squared) = ",
        #       model.train_results, "\nResults on test data (mae, mse, rmse, r_squared) = ",
        #       model.test_results)


pipeline(use_new_model=True, file='data/data.csv')

