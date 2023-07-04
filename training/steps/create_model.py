def model_prediction(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    x_train_pred = model.predict(x_train)
    x_test_pred = model.predict(x_test)
    return model, x_train_pred, x_test_pred