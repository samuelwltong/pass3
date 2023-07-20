import numpy as np

def hyperparameters(model_num):
    parameters_set = [# Parameters of pipelines can be set using '__' separated parameter names:
                      # Parameters for LinearSVC model
                      {"classifier__penalty": ['l2'],  # l1 does not support hinge loss function.
                      "classifier__C": [0.1, 1, 10, 100],
                      "classifier__loss": ['hinge', 'squared_hinge'],
                      "classifier__max_iter": list(range(1000, 5000)),
                      "classifier__dual": [False],  # n_samples > n_features.
                      },
                      # Parameters for KNN model
                      {"classifier__n_neighbors": list(range(1, 10)),
                       "classifier__weights": ['uniform', 'distance'],
                       "classifier__algorithm": ['auto'],# ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
                       "classifier__leaf_size": list(range(2, 40)),
                       "classifier__p": [1, 2],
                       "classifier__metric": ['minkowski', 'chebyshev'],
                      },
                      # Parameters for RandomForest model
                      {'classifier__n_estimators': [int(x) for x in np.linspace(start=200, stop=1000, num=5)],# Number of trees in random forest
                       'classifier__max_features': ['log2', 'sqrt'],# Number of features to consider at every split - log2(n_features) or sqrt(n_features)
                       'classifier__max_depth': [int(x) for x in np.linspace(10, 110, num=5)],  # Maximum number of levels in tree
                       'classifier__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
                       'classifier__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
                       'classifier__bootstrap': [True, False],  # Method of selecting samples for training each tree
                      },
                      # Parameters for SGDClassifier model
                      {'classifier__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],  # learning rate
                       'classifier__n_iter': [1000, 1500, 2000],  # number of epochs
                       'classifier__loss': ['hinge', 'log_loss', 'squared_error'],  # logistic regression,
                       'classifier__penalty': ['l1', 'l2'],  # Minimum number of samples required to split a node
                      }
                      ]
    return parameters_set[model_num]