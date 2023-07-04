def delete_outliers(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    data = data[(data[col] >= q1 - 1.5*iqr) & (data[col] <= q3 + 1.5*iqr)]
    return data


def delete_spaces(data, col):
    data[col] = data[col].str.strip()
    return data


def object_cols(data):
    cols = data.select_dtypes(include='object').columns
    return cols


def numeric_cols(data):
    cols = data.select_dtypes(include=['float64', 'int64']).columns
    return cols


def to_lower(data, key):
    data[key] = data[key].str.lower()
    return data


def clean_data(data):
    data = data.dropna()
    data.drop_duplicates(inplace=True)
    data = data[(data['Cargo_volume'] != 0) & (data['Cargo_weight'] != 0)]
    for col in object_cols(data):
        data = delete_spaces(data, col)
        data = to_lower(data, col)
    for col in numeric_cols(data):
        data = delete_outliers(data, col)
    return data

