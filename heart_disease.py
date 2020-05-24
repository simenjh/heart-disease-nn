import data_processing as dproc
from sklearn.model_selection import train_test_split
import model
import dataplot


def heart_disease(data_file, iterations=3000, learning_rate=0.1, reg_param=0.1, plot_learning_curves=False):
    dataset = dproc.read_dataset(data_file)
    X, y = dproc.preprocess(dataset)
    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)
    
    # Standardize data
    X_train_std, X_cv_std = dproc.standardize(X_train, X_cv)

    activation_layers = (25, 1)
    parameters = model.init_params(X.T, activation_layers)

    model.train_model(X_train_std.T, y_train.T, parameters, iterations, learning_rate, reg_param)




    if plot_learning_curves:
        costs_train, costs_cv, m_examples = model.train_various_sizes(X_train_std.T, X_cv_std.T, y_train.T, y_cv.T, parameters, activation_layers, 3000, 0.01, reg_param)
        dataplot.plot_learning_curves(costs_train, costs_cv, m_examples)
        

    
    train_accuracy = model.compute_accuracy(X_train_std.T, y_train.T, parameters)
    cv_accuracy = model.compute_accuracy(X_cv_std.T, y_cv.T, parameters)
    print(f"Train accuracy: {train_accuracy}")
    print(f"CV accuracy: {cv_accuracy}")
    

