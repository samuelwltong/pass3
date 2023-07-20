#Libraries required =======================================
import os
from Query_Data import query_dataset
from Preprocessing_Transformation import data_preprocessing, data_fitnorm, data_Xy, data_transform
from Hyperparameters import hyperparameters
from Save_Load_ML import save_ML, load_ML

import numpy as np
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')


# Enter database path for querying ==================================================
path = input(r'Enter the file path of the database: ')
# path = r"C:\Users\krhen\Desktop\Work Documents\Jobs\AIAP13\data\failure.db"

if os.path.exists(path):
    print('Entered file exists.','\n')
else:
    print('The specified file does NOT exist.','\n')
path = path.replace('\\', '/')
df = query_dataset(path)

# Read txt file for hyperparameters setting ==========================================
variables = []
with open("hyperparameters.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith("#"):
            continue
        else:
            variables.append(line)
file.close()

# Data Preprocessing and Transformations ==============================================================================================
# To remove duplicates, standardise data, drop insignificant features and create more significant features
df = data_preprocessing(df)

# Fit dataset to normal distribution
df = data_fitnorm(df)

# Split dataset into input features, X, and output labels, y
X,y = data_Xy(df)

# Create the preprocessing pipelines for both numeric and categorical data
numeric_features = ["Temperature", "RPM", "Fuel consumption", "Year"]
categorical_features = ["Model", "Factory", "Usage", "Membership"]
transformer = data_transform(numeric_features, categorical_features)

# Training model ==========================================================================================================================
#selection of how to partion X into training set and test set. training set will be further split in cross validation step
variables[0] = float(variables[0]) #test_size for train data split
variables[1] = int(variables[1])#k_fold Cross Validation
variables[2] = int(variables[2]) #no. of processor used for GridSearchCV
#variables[3]  #'auto' fine-tune model that has highest f1 score/ 'all' fine-tune all available models.

# Selection of training model based on nature of problem and sample size
if X.shape[0] < 50:

    print('Dataset is too small to be trained - More data samples have to be collected.')

# Linear Support Vector Classifier, KNeighbors Classifier and Ensemble Classifiers seem suitable for this problem.
elif (X.shape[0] >= 50) and (X.shape[0] < 100_000):
    model_rank = []
    print('Model selection ============================')
    # LinearSVC - Forming preprocessing and transforming pipeline -------------------------------------------------
    pipe = Pipeline(steps=[("transformer", transformer), ("classifier", LinearSVC())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=variables[0], random_state=42)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print('LinearSVC -------------')
    print("model dev score: %.3f" % np.mean(cross_val_score(pipe, X_train, y_train, cv=variables[1])))
    print("model test score: %.3f" % pipe.score(X_test, y_test))
    print("f1 score on test set: %.3f" % f1_score(y_test, y_pred), '\n')
    model_rank.append(f1_score(y_test, y_pred))

    # KNeighbors - Forming preprocessing and transforming pipeline ------------------------------------------------
    pipe = Pipeline(steps=[("transformer", transformer), ("classifier", KNeighborsClassifier())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=variables[0], random_state=42)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print('KNeighborsClassifier -------------')
    print("model dev score: %.3f" % np.mean(cross_val_score(pipe, X_train, y_train, cv=variables[1])))
    print("model test score: %.3f" % pipe.score(X_test, y_test))
    print("f1 score on test set: %.3f" % f1_score(y_test, y_pred), '\n')
    model_rank.append(f1_score(y_test, y_pred))

    # RandomForestClassifier - Forming preprocessing and transforming pipeline ------------------------------------------------
    pipe = Pipeline(steps=[("transformer", transformer), ("classifier", RandomForestClassifier())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=variables[0], random_state=42)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print('RandomForestClassifier -------------')
    print("model dev score: %.3f" % np.mean(cross_val_score(pipe, X_train, y_train, cv=variables[1])))
    print("model test score: %.3f" % pipe.score(X_test, y_test))
    print("f1 score on test set: %.3f" % f1_score(y_test, y_pred), '\n')
    model_rank.append(f1_score(y_test, y_pred))

# SGD Classifier is suitable larger dataset > 100K.
else:
    # SGDClassifier - Forming preprocessing and transforming pipeline -------------------------------------------------
    pipe = Pipeline(steps=[("transformer", transformer), ("classifier", SGDClassifier())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=variables[0], random_state=42)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print('SGDClassifier -------------')
    print("model dev score: %.3f" % np.mean(cross_val_score(pipe, X_train, y_train, cv=variables[1])))
    print("model test score: %.3f" % pipe.score(X_test, y_test))
    print("f1 score on test set: %.3f" % f1_score(y_test, y_pred), '\n')

# Hyperparameter fine tuning ========================================================================================
classifiers = [LinearSVC(), KNeighborsClassifier(), RandomForestClassifier(), SGDClassifier()]
model_num = [0, 1, 2, 3]
if (X.shape[0] >= 50) and (X.shape[0] < 100_000):
    if variables[3] == 'auto':
        if model_rank.index(max(model_rank)) == 0:
            print('LinearSVC model has the highest f1_score - selected to be fine-tuned.')
            model = classifiers[0]
            param_grid = hyperparameters(model_num[0])

        elif model_rank.index(max(model_rank)) == 1:
            print('KNeighborsClassifier model has the highest f1_score -  selected to be fine-tuned.')
            model = classifiers[1]
            param_grid = hyperparameters(model_num[1])

        else:
            print('RandomForestClassifier model has the highest f1_score -  selected to be fine-tuned.')
            model = classifiers[2]
            param_grid = hyperparameters(model_num[2])

        pipe = Pipeline(steps=[("transformer", transformer), ("classifier", model)])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=variables[0], random_state=42)

        search = GridSearchCV(pipe, param_grid, n_jobs=variables[2])  # n_jobs - Number of jobs to run in parallel, change it to 1 if convergence is anot achievable
        search.fit(X_train, y_train.values.ravel())
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)

        # Predict the target ==================================================================================================
        y_pred = search.predict(X_test)
        print('Overall accuracy of the fine-tuned model:', accuracy_score(y_test, y_pred))
        print('Precision of the fine-tuned model:', precision_score(y_test, y_pred))
        print('Recall score of the fine-tuned model:', recall_score(y_test, y_pred))
        print("f1 score of the fine-tuned model: %.3f" % f1_score(y_test, y_pred), '\n')

        # Save trained model ===================================================================================================
        filename = path.replace('/failure.db', '/finalized_model.sav')
        save_ML(search, filename)

        # load the model from disk ===================================================================================================
        loaded_model = load_ML(filename)
        result = loaded_model.score(X_test, y_test)
        print('Accuracy of loaded model:', result, '\n')

    else:
        print('LinearSVC, KNN, RandomForest models will all be fine-tuned.')
        for i in range(len(classifiers)-1):
            if i == 0:
                print('LinearSVC model to be fine-tuned.')
            elif i == 1:
                print('KNeighborsClassifier model to be fine-tuned.')
            else:
                print('RandomForestClassifier model to be fine-tuned.')

            pipe = Pipeline(steps=[("transformer", transformer), ("classifier", classifiers[i])])
            param_grid = hyperparameters(model_num[i])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=variables[0], random_state=42)

            search = GridSearchCV(pipe, param_grid, n_jobs=variables[2])  # n_jobs - Number of jobs to run in parallel, change it to 1 if convergence is anot achievable
            search.fit(X_train, y_train.values.ravel())
            print("Best parameter (CV score=%0.3f):" % search.best_score_)
            print(search.best_params_)

            # Predict the target ==================================================================================================
            y_pred = search.predict(X_test)
            print('Overall accuracy of the fine-tuned model:', accuracy_score(y_test, y_pred))
            print('Precision of the fine-tuned model:', precision_score(y_test, y_pred))
            print('Recall score of the fine-tuned model:', recall_score(y_test, y_pred))
            print("f1 score of the fine-tuned model: %.3f" % f1_score(y_test, y_pred), '\n')

            # Save trained model ===================================================================================================
            filename = path.replace('/failure.db', '/finalized_model')
            filename = filename + str(i) + '.sav'
            save_ML(search, filename)

            # load the model from disk ===================================================================================================
            loaded_model = load_ML(filename)
            result = loaded_model.score(X_test, y_test)
            print('Accuracy of loaded model:', result, '\n')

else:
    print('SGDClassifier model to be fine-tuned.')
    model = classifiers[3]
    param_grid = hyperparameters(model_num[3])

    pipe = Pipeline(steps=[("transformer", transformer), ("classifier", model)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=variables[0], random_state=42)

    search = GridSearchCV(pipe, param_grid, n_jobs=variables[2])  # Number of jobs to run in parallel
    search.fit(X_train, y_train.values.ravel())
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    # Predict the target ==================================================================================================
    y_pred = search.predict(X_test)
    print('Overall accuracy of the fine-tuned model:', accuracy_score(y_test,y_pred))
    print('Precision of the fine-tuned model:', precision_score(y_test,y_pred))
    print('Recall score of the fine-tuned model:', recall_score(y_test,y_pred))
    print("f1 score of the fine-tuned model: %.3f" % f1_score(y_test, y_pred), '\n')

    # Save trained model ===================================================================================================
    filename = path.replace('/failure.db', '/finalized_model.sav')
    save_ML(search, filename)

    # load the model from disk ===================================================================================================
    loaded_model = load_ML(filename)
    result = loaded_model.score(X_test, y_test)
    print('Accuracy of loaded model:', result, '\n')