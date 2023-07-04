import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from training.steps.clean_data import delete_outliers

scaler = StandardScaler()


def cargo_range_dummies(data):
    z = round(data.groupby(["cargo_name"])["money_rub"].agg(["mean"]), 2).T
    data = data.merge(z.T, how="left", on="cargo_name")
    cargo_bin = ['Low', 'Medium', 'High']
    data['cargo_range'] = pd.qcut(data['mean'], q=3, labels=cargo_bin)
    new_data = pd.get_dummies(columns=["cargo_range"], data=data)
    return new_data


def add_cargo_weight_volume(data):
    data['cargo_weight_volume'] = data['Cargo_weight'] * data['Cargo_volume']
    data = delete_outliers(data, 'cargo_weight_volume')
    return data


def data_split(data, price):
    x = data.drop(columns=[price])
    y = data[price]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def standardising_data(data, num_cols):
    data[num_cols] = scaler.fit_transform(data[num_cols])
    return data
