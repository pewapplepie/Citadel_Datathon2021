from numpy.core.fromnumeric import mean
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import StratifiedShuffleSplit,  KFold, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, classification_report, cohen_kappa_score, make_scorer, accuracy_score
from datetime import datetime
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import itertools
import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from hyperopt.fmin import generate_trials_to_calculate
import tensorflow as tf
from tensorflow import keras
from collections import Counter
from imblearn.over_sampling import RandomOverSampler as ros


## deprecated
class oversampled_Kfold:
    def __init__(self, n_splits, n_repeats = 2, random_state = 441):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        np.random.seed(random_state)

    def get_n_splits(self, X, y):
        return self.n_splits * self.n_repeats

    def split(self, X, y):
        splits = np.split(np.random.choice(len(X), len(X), replace=False), self.n_splits)
        train, test = [], []
        for _ in range(self.n_repeats):
            for idx in range(len(splits)):
                trainingIdx = np.delete(splits, idx)
                Xidx_r, _ = ros.fit_resample(
                    trainingIdx.reshape((-1,1)),
                    y[trainingIdx])
                train.append(Xidx_r.flatten())
                test.append(splits[idx])
        return list(zip(train, test))

def box_cox_transform(arr, alpha):
    if alpha == 0.0:
        return np.log(arr)
    else:
        return (arr ** alpha - 1) / alpha 

def evaluate_model(predictions, train_predictions, train_labels, test_labels, metric_only):
    # baseline = {}
    # # baseline['recall'] = recall_score(test_labels, [0 for _ in range(len(test_labels))], average='weighted')
    # # baseline['precision'] = precision_score(test_labels, [0 for _ in range(len(test_labels))], average='weighted')
    # baseline['roc'] = 0.5
    # results = {}
    # results['recall'] = recall_score(test_labels, predictions, average='weighted')
    # results['precision'] = precision_score(test_labels, predictions, average='weighted')
    # results['roc'] = roc_auc_score(test_labels, probs, multi_class = "ovo")

    # train_results = {}
    # train_results['recall'] = recall_score(train_labels, train_predictions, average='weighted')
    # train_results['precision'] = precision_score(train_labels, train_predictions, average='weighted')
    # train_results['roc'] = roc_auc_score(train_labels, train_probs, multi_class = "ovo")
    
    # for metric in ['recall', 'precision', 'roc']:
    #     print(f'{metric.capitalize()} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    # base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    # model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    # plt.figure(figsize = (8, 6))
    # plt.rcParams['font.size'] = 16
    
    # # Plot both curves
    # plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    # plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    # plt.legend()
    # plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves')
    if not metric_only:
        print("-------------------------")
        print("Training Classification".upper())
        print(classification_report(train_labels, train_predictions))
        print("Cohen's Kappa =", cohen_kappa_score(train_labels, train_predictions))
        print("-------------------------")
        print("Testing Classification".upper())
        print(classification_report(test_labels, predictions))
        print("Cohen's Kappa =", cohen_kappa_score(test_labels, predictions))
    else:
        return cohen_kappa_score(test_labels, predictions)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

def cohen_scorer():
    return make_scorer(cohen_kappa_score)

class Trainer:
    def __init__(self, feature, target, train_type, verbose, log_text = "Training", filled = False):
        self.feature = feature
        self.target = target
        self.filled = filled
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        self.log_text = train_type + ": " + log_text + "\n" + "Training Time: " + current_time
        if verbose != 0:
            self.training_log()

    def training_log(self):
        print(self.log_text)

    def initiate_cv(
        self, 
        n_splits = 3,
        shuffle = True, 
        random_state = 441):
        self.cv_pair = KFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)

    def override_cv(self, new_cv):
        self.cv_pair = new_cv

    def train_model(self, params = {}):
        self.model = None

    def get_model_params(self):
        if self.model is None:
            print("No model specified yet")
        else:
            return self.model.get_params()
    def get_class_weights(self, alpha, num_output, custom = False,  target_col_index = -1):
        if custom:
            class_count = [[cl, num] for cl, num in Counter(self.target.iloc[:, target_col_index]).items()]
            weight_list = [num / sum(num for cl, num in class_count) for cl, num in sorted(class_count, key = lambda x: x[0])]
            weight_list = box_cox_transform(np.reciprocal(weight_list), alpha = alpha)
            sample_class_weight = {x:y for x, y in zip(range(0,  num_output), weight_list)}
            return sample_class_weight
        else:
            return {x:y for x, y in zip(range(0,  num_output), np.repeat(1,  num_output))}

    def parameter_tune_grid(
        self,
        params,
        scoring,
        n_job = -1,
        verbose = 10,
        cv = 3
        ):
        gsearch = GridSearchCV(
            estimator = self.model, 
            param_grid = params, 
            scoring = scoring, 
            n_jobs = n_job, 
            verbose = verbose, cv = cv)
        gsearch.fit(self.feature, self.target)
        self.model = gsearch.best_estimator_

    def evaluate_model(
        self,
        n_splits = 3,
        n_repeats = 2,
        shuffle = True,
        random_state = 441,
        new_cv = None,
        metric_only = False
        ):
        if new_cv is None:
            self.initiate_cv(n_splits = n_splits, n_repeats = n_repeats, shuffle = shuffle, random_state = random_state)
        else:
            self.override_cv(new_cv)
        metric_result = []
        for train, test in self.cv_pair.split(self.feature, self.target):
            trainset = self.feature.iloc[train, :]
            trainset_label = self.target.iloc[train, :]
            testset = self.feature.iloc[test, :]
            testset_label = self.target.iloc[test, :]
            self.model.fit(trainset, trainset_label)
            
            train_rf_predictions = self.model.predict(trainset)
            # train_rf_probs = self.model.predict_proba(trainset)[:, 1]

            rf_predictions = self.model.predict(testset)
            # rf_probs = self.model.predict_proba(testset)[:, 1]
    
            if not metric_only:
                evaluate_model(rf_predictions, train_rf_predictions, trainset_label, testset_label, metric_only)
                cm = confusion_matrix(testset_label, rf_predictions)
                target_list = self.target.iloc[:,-1].unique()
                plot_confusion_matrix(cm, classes = target_list,
                            title = 'Confusion Matrix')
            else:
                metric = evaluate_model(rf_predictions, train_rf_predictions, trainset_label, testset_label, metric_only)
                metric_result.append(metric)
        return np.mean(metric_result) if metric_only else -1

    def feature_importance(self):
        pass

    def predict(self, new_feature, new_target = None, fitted = False):
        if not fitted:
            self.train_model(self.get_model_params())
            if not isinstance(new_target, pd.core.frame.DataFrame):
                self.model.fit(self.feature, self.target)
            else:
                self.model.fit(new_feature, new_target)
        self.prediction = self.model.predict(new_feature)
        return self.prediction

    def predict_proba(self, new_feature):
        self.model.fit(self.feature, self.target)
        self.prediction_proba = self.model.predict_proba(new_feature)
        return self.prediction_proba

    def bayes_optimize(
        self,
        search_space,
        n_splits = 3,
        shuffle = True,
        random_state = 441,
        max_evals = 1000,
        init_vals = [],
        ):
        def obj_fnc(params):
            self.train_model(params)
            metric = self.evaluate_model(
                n_splits = n_splits,
                shuffle = shuffle,
                random_state = random_state,
                metric_only = True
            )
            return {"loss" : -metric, "status" : STATUS_OK}
        hyper_trials = Trials() if init_vals == [] else generate_trials_to_calculate(init_vals)
        best_params = fmin(obj_fnc, search_space, algo = tpe.suggest, 
        max_evals = max_evals, trials= hyper_trials)
        # current_params = self.get_model_params()
        # for feat in best_params.keys():
        #     current_params[feat] = search_space[feat].pos_args[best_params[feat] + 1]._obj
        print(best_params)
        print(hyper_trials.best_trial['result']['loss'])
        return best_params

class BoundTrainer:
    def __init__(self, feature, target: pd.Series, train_type, verbose, log_text = "Training"):
        self.feature = feature
        self.target = target
        self.target_pse = target.map(lambda x: 0 if x <= 0.5 else 1)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        self.log_text = train_type + ": " + log_text + "\n" + "Training Time: " + current_time
        if verbose != 0:
            self.training_log()

    def training_log(self):
        print(self.log_text)

    def initiate_cv(
        self, 
        n_splits = 3,
        shuffle = True, 
        random_state = 441):
        self.cv_pair = KFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)

    def override_cv(self, new_cv):
        self.cv_pair = new_cv

    def train_model(self, params = {}):
        self.model = None

    def get_model_params(self):
        if self.model is None:
            print("No model specified yet")
        else:
            return self.model.get_params()
    def get_class_weights(self, alpha, num_output, custom = False, target_col_index = -1):
        if custom:
            class_count = [[cl, num] for cl, num in Counter(self.target_pse.iloc[:, target_col_index]).items()]
            weight_list = [num / sum(num for cl, num in class_count) for cl, num in sorted(class_count, key = lambda x: x[0])]
            weight_list = box_cox_transform(np.reciprocal(weight_list), alpha = alpha)
            sample_class_weight = {x:y for x, y in zip(range(0,  num_output), weight_list)}
            return sample_class_weight
        else:
            return {x:y for x, y in zip(range(0,  num_output), np.repeat(1,  num_output))}

    def parameter_tune_grid(
        self,
        params,
        scoring,
        n_job = -1,
        verbose = 10,
        cv = 3
        ):
        gsearch = GridSearchCV(
            estimator = self.model, 
            param_grid = params, 
            scoring = scoring, 
            n_jobs = n_job, 
            verbose = verbose, cv = cv)
        gsearch.fit(self.feature, self.target_pse)
        self.model = gsearch.best_estimator_

    def evaluate_model(
        self,
        n_splits = 3,
        shuffle = True,
        random_state = 441,
        new_cv = None,
        ):
        if new_cv is None:
            self.initiate_cv(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
        else:
            self.override_cv(new_cv)
        metric_result = []
        acc_result = []
        for train, test in self.cv_pair.split(self.feature, self.target_pse):
            trainset = self.feature.iloc[train, :]
            trainset_label = self.target_pse.iloc[train]
            testset = self.feature.iloc[test, :]
            testset_prob = self.target.iloc[test]
            self.model.fit(trainset, trainset_label)

            pred_prob = self.model.predict_proba(testset)[:, 1]
            metric = mean_squared_error(testset_prob, pred_prob)
            acc = r2_score(testset_prob, pred_prob)
            metric_result.append(metric)
            acc_result.append(acc)
        return np.mean(metric_result), np.mean(acc_result)

    def predict(self, new_feature, new_target = None, fitted = False):
        if not fitted:
            self.train_model(self.get_model_params())
            if not isinstance(new_target, pd.core.frame.DataFrame):
                self.model.fit(self.feature, self.target)
            else:
                self.model.fit(new_feature, new_target)
        self.prediction = self.model.predict(new_feature)
        return self.prediction

    def predict_proba(self, new_feature):
        self.model.fit(self.feature, self.target_pse)
        self.prediction_proba = self.model.predict_proba(new_feature)
        return self.prediction_proba

    def bayes_optimize(
        self,
        search_space,
        n_splits = 3,
        shuffle = True,
        random_state = 441,
        max_evals = 1000,
        init_vals = [],
        ):
        def obj_fnc(params):
            self.train_model(params)
            metric = self.evaluate_model(
                n_splits = n_splits,
                shuffle = shuffle,
                random_state = random_state,
            )
            return {"loss" : metric[0], "status" : STATUS_OK}
        hyper_trials = Trials() if init_vals == [] else generate_trials_to_calculate(init_vals)
        best_params = fmin(obj_fnc, search_space, algo = tpe.suggest, 
        max_evals = max_evals, trials= hyper_trials)
        # current_params = self.get_model_params()
        # for feat in best_params.keys():
        #     current_params[feat] = search_space[feat].pos_args[best_params[feat] + 1]._obj
        print(best_params)
        print(hyper_trials.best_trial['result']['loss'])
        return best_params

class Regressor:
    def __init__(self, feature, target: pd.Series, train_type, verbose, log_text = "Training"):
        self.feature = feature
        self.target = target
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        self.log_text = train_type + ": " + log_text + "\n" + "Training Time: " + current_time
        if verbose != 0:
            self.training_log()

    def training_log(self):
        print(self.log_text)

    def initiate_cv(
        self, 
        n_splits = 3,
        shuffle = True, 
        random_state = 441):
        self.cv_pair = KFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)

    def override_cv(self, new_cv):
        self.cv_pair = new_cv

    def train_model(self, params = {}):
        self.model = None

    def get_model_params(self):
        if self.model is None:
            print("No model specified yet")
        else:
            return self.model.get_params()

    def parameter_tune_grid(
        self,
        params,
        scoring,
        n_job = -1,
        verbose = 10,
        cv = 3
        ):
        gsearch = GridSearchCV(
            estimator = self.model, 
            param_grid = params, 
            scoring = scoring, 
            n_jobs = n_job, 
            verbose = verbose, cv = cv)
        gsearch.fit(self.feature, self.target_pse)
        self.model = gsearch.best_estimator_

    def evaluate_model(
        self,
        n_splits = 3,
        shuffle = True,
        random_state = 441,
        new_cv = None,
        ):
        if new_cv is None:
            self.initiate_cv(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
        else:
            self.override_cv(new_cv)
        metric_result = []
        acc_result = []
        for train, test in self.cv_pair.split(self.feature, self.target):
            trainset = self.feature.iloc[train, :]
            trainset_label = self.target.iloc[train]
            testset = self.feature.iloc[test, :]
            testset_prob = self.target.iloc[test]
            self.model.fit(trainset, trainset_label)

            pred_prob = self.model.predict(testset)
            metric = mean_squared_error(testset_prob, pred_prob)
            acc = r2_score(testset_prob, pred_prob)
            metric_result.append(metric)
            acc_result.append(acc)
        return np.mean(metric_result), np.mean(acc_result)

    def predict(self, new_feature, new_target = None, fitted = False):
        if not fitted:
            self.train_model(self.get_model_params())
            if not isinstance(new_target, pd.core.frame.DataFrame):
                self.model.fit(self.feature, self.target)
            else:
                self.model.fit(new_feature, new_target)
        self.prediction = self.model.predict(new_feature)
        return self.prediction

    def predict_proba(self, new_feature):
        self.model.fit(self.feature, self.target_pse)
        self.prediction_proba = self.model.predict_proba(new_feature)
        return self.prediction_proba

    def bayes_optimize(
        self,
        search_space,
        n_splits = 3,
        shuffle = True,
        random_state = 441,
        max_evals = 1000,
        init_vals = [],
        ):
        def obj_fnc(params):
            self.train_model(params)
            metric = self.evaluate_model(
                n_splits = n_splits,
                shuffle = shuffle,
                random_state = random_state,
            )
            return {"loss" : metric[0], "status" : STATUS_OK}
        hyper_trials = Trials() if init_vals == [] else generate_trials_to_calculate(init_vals)
        best_params = fmin(obj_fnc, search_space, algo = tpe.suggest, 
        max_evals = max_evals, trials= hyper_trials)
        # current_params = self.get_model_params()
        # for feat in best_params.keys():
        #     current_params[feat] = search_space[feat].pos_args[best_params[feat] + 1]._obj
        print(best_params)
        print(hyper_trials.best_trial['result']['loss'])
        return best_params


class LWEnsemble(Trainer):
    def __init__(self, feature, target, log_text, filled):
        super(LWEnsemble, self).__init__(feature, target, "Linear Weighted Voter", 1, log_text, filled = filled)
    def train_model(self, params, weight_list):
        self.model = LogisticRegression(**params)
        self.model_params = self.model.get_params()

class LGTrainer(Trainer):
    def __init__(self, feature, target, log_text, filled):
        super(LGTrainer, self).__init__(feature, target, "Softmax Classifier", 1, log_text, filled = filled)
    def train_model(self, params = {}):
        self.model = LogisticRegression(**params)
        self.model_params = self.model.get_params()
class RFTrainer(Trainer):
    def __init__(self, feature, target, log_text, filled):
        super(RFTrainer, self).__init__(feature, target, "Random Forest Classifier", 1, log_text, filled = filled)
    def train_model(self, params = {}):
        self.model = RandomForestClassifier(
            **params)
        self.model_params = self.model.get_params()
    
class XGTrainer(Trainer):
    def __init__(self, feature, target, log_text, filled):
        super(XGTrainer, self).__init__(feature, target, "XGBoost Classifier", 1, log_text, filled = filled)
    def train_model(self, params = {}):
        self.model = XGBClassifier(**params)
        ''' 
        Sample Parameters:
        XGBClassifier(learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softmax',
                    nthread=4,
                    num_class=9,
                    seed=27)
        '''
        self.model_params = self.model.get_params()

class SVMTrainer(Trainer):
    def __init__(self, feature, target, log_text, filled):
        super(SVMTrainer, self).__init__(feature, target, "SVM Classifier", 1, log_text, filled = filled)
    def train_model(self, params = {}):
        self.model = SVC(**params)
        self.model_params = self.model.get_params()

class NNTrainer(Trainer):
    def __init__(self, feature, target, log_text, verbose, filled = False):
        super(NNTrainer, self).__init__(feature, target, "NN Classifier", verbose, log_text, filled)
    def train_model(
        self, 
        learning_rate,
        num_dense_layers,
        num_dense_nodes,
        dropout,
        activation,
        adam_decay,
        batch_size,
        epochs,
        weight_applied = True,
        num_input_nodes = None,
        num_output = None,
        box_cox_alpha = 0.0):
        if num_input_nodes is None:
            num_input_nodes = self.feature.shape[1]
        if num_output is None:
            num_output = len(self.target.iloc[:,-1].unique())
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape = (num_input_nodes)))
        for _ in range(num_dense_layers):
            model.add(keras.layers.Dense(num_dense_nodes, activation = activation))
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(num_output, activation = "softmax"))
        model.compile(
            optimizer = keras.optimizers.Adam(lr = learning_rate, decay = adam_decay),
            loss = 'sparse_categorical_crossentropy',
            metrics = ["accuracy"]
        )
        self.model = model
        self.model_params = dict(
            learning_rate = learning_rate,
            num_input_nodes = num_input_nodes,
            num_output = num_output,
            num_dense_layers = num_dense_layers,
            num_dense_nodes = num_dense_nodes,
            dropout = dropout,
            activation = activation,
            adam_decay = adam_decay,
            batch_size = batch_size,
            epochs = epochs,
            weight_applied = weight_applied,
            box_cox_alpha = box_cox_alpha
        )
    
    # def fitness(
    #     self, 
    #     learning_rate,
    #     num_dense_layers,
    #     num_input_nodes,
    #     num_dense_nodes,
    #     dropout,
    #     activation,
    #     adam_decay,
    #     num_output,
    #     batch_size,
    #     epochs):
    #     self.train_model(
    #         learning_rate,
    #         num_dense_layers,
    #         num_input_nodes,
    #         num_dense_nodes,
    #         dropout,
    #         activation,
    #         adam_decay,
    #         num_output
    #     )
    #     blackbox = self.model.fit(
    #         x = self.feature,
    #         y = self.target,
    #         epochs = epochs,
    #         batch_size = batch_size
    #     )
    #     self.model_params["epochs"] = epochs
    #     self.model_params["batch_size"] = batch_size
    def get_class_weights(self, new_target = None, target_col_index = -1):
        if not isinstance(new_target, pd.core.frame.DataFrame):
            local_target = self.target
        else:
            local_target = new_target
        if self.model_params["weight_applied"]:
            class_count = [[cl, num] for cl, num in Counter(local_target.iloc[:, target_col_index]).items()]
            weight_list = [num / sum(num for cl, num in class_count) for cl, num in sorted(class_count, key = lambda x: x[0])]
            weight_list = box_cox_transform(np.reciprocal(weight_list), alpha = self.model_params["box_cox_alpha"])
            sample_class_weight = {x:y for x, y in zip(range(0, self.model_params["num_output"]), weight_list)}
            return sample_class_weight
        else:
            return {x:y for x, y in zip(range(0, self.model_params["num_output"]), np.repeat(1, self.model_params["num_output"]))}

    def evaluate_model(
        self,
        n_splits = 3,
        n_repeats = 2,
        shuffle = True,
        random_state = 441,
        new_cv = None,
        metric_only = False,
        target_col_index = -1
        ):
        if new_cv is None:
            self.initiate_cv(n_splits = n_splits, n_repeats = n_repeats, shuffle = shuffle, random_state = random_state)
        else:
            self.override_cv(new_cv)
        metric_result = []
        for train, test in self.cv_pair.split(self.feature, self.target):
            self.train_model(**self.model_params)
            trainset = self.feature.iloc[train, :]
            trainset_label = self.target.iloc[train, :]
            testset = self.feature.iloc[test, :]
            testset_label = self.target.iloc[test, :]
            self.model.fit(
                trainset, 
                trainset_label.values.ravel(), 
                epochs = self.model_params["epochs"], 
                batch_size = self.model_params["batch_size"],
                class_weight = self.get_class_weights(trainset_label, target_col_index), verbose = 0)
            
            train_rf_predictions = np.argmax(self.model.predict(trainset), axis=-1)
            # train_rf_probs = self.model.predict_proba(trainset)[:, 1]

            rf_predictions = np.argmax(self.model.predict(testset), axis=-1)
            # rf_probs = self.model.predict_proba(testset)[:, 1]

            if not metric_only:
                evaluate_model(rf_predictions, train_rf_predictions, trainset_label, testset_label, metric_only)
                cm = confusion_matrix(testset_label, rf_predictions)
                target_list = self.target.iloc[:,-1].unique()
                plot_confusion_matrix(cm, classes = target_list,
                            title = 'Confusion Matrix')
            else:
                metric = evaluate_model(rf_predictions, train_rf_predictions, trainset_label, testset_label, metric_only)
                metric_result.append(metric)
        return np.mean(metric_result) if metric_only else -1

    def bayes_optimize(
        self,
        search_space,
        n_splits = 3,
        n_repeats = 2,
        shuffle = True,
        random_state = 441,
        max_evals = 1000,
        init_vals = []
        ):
        def obj_fnc(params):
            self.train_model(**params)
            metric = self.evaluate_model(
                n_splits = n_splits,
                n_repeats = n_repeats,
                shuffle = shuffle,
                random_state = random_state,
                metric_only = True
            )
            return {"loss" : -metric, "status" : STATUS_OK}
        hyper_trials = Trials() if init_vals == [] else generate_trials_to_calculate(init_vals)
        best_params = fmin(obj_fnc, search_space, algo = tpe.suggest, max_evals = max_evals, trials= hyper_trials)
        # current_params = self.get_model_params()
        # for feat in best_params.keys():
        #     current_params[feat] = search_space[feat].pos_args[best_params[feat] + 1]._obj
        print(best_params)
        print(hyper_trials.best_trial['result']['loss'])
        return best_params

    def predict(self, new_feature, new_target = None, fitted = False, target_col_index = -1):
        batch_size = self.model_params["batch_size"]
        epochs = self.model_params["epochs"]
        if not fitted:
            self.train_model(**self.model_params)
            if not isinstance(new_target, pd.core.frame.DataFrame):
                self.model.fit(
                    self.feature, 
                    self.target, 
                    batch_size = batch_size, 
                    class_weight = self.get_class_weights(target_col_index), 
                    epochs = epochs,
                    verbose = 0)
            else:
                self.model.fit(
                    new_feature, 
                    new_target, 
                    batch_size = batch_size, 
                    class_weight = self.get_class_weights(new_target, target_col_index), 
                    epochs = epochs,
                    verbose = 0)
        self.prediction = np.argmax(self.model.predict(new_feature), axis = -1)
        return self.prediction

    def predict_proba(self, new_feature, target_col_index = -1):
        batch_size = self.model_params["batch_size"]
        epochs = self.model_params["epochs"]
        self.model.fit(
            self.feature, 
            self.target, 
            batch_size = batch_size, 
            class_weight = self.get_class_weights(target_col_index), 
            epochs = epochs,
            verbose = 0)
        self.prediction_proba = self.model.predict(new_feature)
        return self.prediction_proba

class NNHybridTrainer(NNTrainer):
    def __init__(self, feature, target, target_real, log_text = "NN Hybrid Classifier", verbose = 0, filled = False):
        super(NNHybridTrainer, self).__init__(feature, target, log_text, verbose, filled)
        self.target_real = target_real
        self.largest = max(self.target.iloc[:,-1])
    def weight_minority(self, target):
        class_count = [[cl, num] for cl, num in Counter(target.iloc[:, -1]).items() if int(cl) >= self.largest]
        class_list = [cl for cl, num in class_count]
        weight_list = [num / sum(num for cl, num in class_count) for cl, num in class_count]
        return class_list, weight_list
    def multinomial_weight_sample(self, class_list, weight_list):
        sample = np.random.multinomial(1, weight_list, size=1)
        result = class_list[np.where(sample == 1)[1][0]]
        return result
    def evaluate_model(
        self,
        n_splits = 3,
        n_repeats = 2,
        shuffle = True,
        random_state = 441,
        new_cv = None,
        metric_only = False,
        target_col_index = -1
        ):
        if new_cv is None:
            self.initiate_cv(n_splits = n_splits, n_repeats = n_repeats, shuffle = shuffle, random_state = random_state)
        else:
            self.override_cv(new_cv)
        metric_result = []
        np.random.seed(random_state)
        for train, test in self.cv_pair.split(self.feature, self.target):
            self.train_model(**self.model_params)
            trainset = self.feature.iloc[train, :]
            trainset_label = self.target.iloc[train, :]
            trainset_label_real = self.target_real.iloc[train, :]
            class_list, weight_list = self.weight_minority(trainset_label_real)
            testset = self.feature.iloc[test, :]
            testset_label = self.target_real.iloc[test, :]
            self.model.fit(
                trainset, 
                trainset_label.values.ravel(), 
                epochs = self.model_params["epochs"], 
                batch_size = self.model_params["batch_size"],
                class_weight = self.get_class_weights(trainset_label, target_col_index), verbose = 0)
            
            train_rf_predictions = np.argmax(self.model.predict(trainset), axis=-1)
            # train_rf_probs = self.model.predict_proba(trainset)[:, 1]
            train_rf_predictions = np.array([self.multinomial_weight_sample(class_list, weight_list) if pred >= self.largest else pred for pred in train_rf_predictions])
            rf_predictions = np.argmax(self.model.predict(testset), axis=-1)
            rf_predictions = np.array([self.multinomial_weight_sample(class_list, weight_list) if pred >= self.largest else pred for pred in rf_predictions])

            # rf_probs = self.model.predict_proba(testset)[:, 1]

            if not metric_only:
                evaluate_model(rf_predictions, train_rf_predictions, trainset_label_real, testset_label, metric_only)
                cm = confusion_matrix(testset_label, rf_predictions)
                target_list = self.target.iloc[:,-1].unique()
                plot_confusion_matrix(cm, classes = target_list,
                            title = 'Confusion Matrix')
            else:
                metric = evaluate_model(rf_predictions, train_rf_predictions, trainset_label_real, testset_label, metric_only)
                metric_result.append(metric)
        return np.mean(metric_result) if metric_only else -1

    def predict(self, new_feature, new_target = None, fitted = False, target_col_index = -1):
        batch_size = self.model_params["batch_size"]
        epochs = self.model_params["epochs"]
        class_list, weight_list = self.weight_minority(self.target_real)
        if not fitted:
            self.train_model(**self.model_params)
            if not isinstance(new_target, pd.core.frame.DataFrame):
                self.model.fit(
                    self.feature, 
                    self.target, 
                    batch_size = batch_size, 
                    class_weight = self.get_class_weights(target_col_index), 
                    epochs = epochs,
                    verbose = 0)
            else:
                self.model.fit(
                    new_feature, 
                    new_target, 
                    batch_size = batch_size, 
                    class_weight = self.get_class_weights(new_target, target_col_index), 
                    epochs = epochs,
                    verbose = 0)
        self.prediction = np.argmax(self.model.predict(new_feature), axis = -1)
        self.prediction = np.array([self.multinomial_weight_sample(class_list, weight_list) if pred >= self.largest else pred for pred in self.prediction])
        return self.prediction


class StackEnsembler:
    def __init__(self, feature, target, trainer_list, stack_trainer, un_pack = False, filled = False):
        self.feature = feature
        self.target = target
        self.trainer_list = trainer_list
        self.stack_trainer = stack_trainer
        self.trainer_num = len(trainer_list)
        self.un_pack = un_pack
        self.filled = filled
        # self.stack_trainer.train_model(self.stack_trainer.model_params)
        self.stacked_model = self.stack_trainer.model
        self.stacked_params = self.stack_trainer.model_params
    def initiate_cv(
        self, 
        n_splits = 3,
        n_repeats = 2, 
        shuffle = True, 
        random_state = 441):
        if self.filled:
            # self.cv_pair = oversampled_Kfold(
            #     n_splits = n_splits, 
            #     n_repeats = n_repeats, 
            #     random_state = random_state)
            self.cv_pair = StratifiedKFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
        else:
            self.cv_pair = KFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
    def stacked_dataset(self, new_feature, new_target = None, fitted = False):
        stackX = None
        for i in range(self.trainer_num):
            # if fitted:
            yhat = self.trainer_list[i].predict(new_feature, new_target, fitted = fitted)
            # else:
            # yhat = self.trainer_list[i].model.predict(new_feature)
            if stackX is None:
                stackX = yhat
            else:
                stackX = np.dstack((stackX, yhat)) 
        return stackX[0]

    def fit_stacked_model(self, new_feature, new_target):
        # create dataset using ensemble
        stackedX = self.stacked_dataset(new_feature, new_target)
        # fit standalone model
        if self.un_pack:
            self.stack_trainer.model_params["num_input_nodes"] = self.trainer_num
            self.stack_trainer.train_model(**self.stack_trainer.model_params)
            model = self.stack_trainer.model
        else:
            self.stack_trainer.train_model(self.stack_trainer.model_params)
            model = self.stack_trainer.model
        self.stacked_train_set = stackedX
        if self.un_pack:
            model.fit(
                stackedX,
                new_target.values.ravel(),
                epochs = self.stack_trainer.model_params["epochs"], 
                batch_size = self.stack_trainer.model_params["batch_size"],
                class_weight = self.stack_trainer.get_class_weights(new_target), verbose = 0
            )
        else:
            model.fit(stackedX, new_target)
        self.stacked_model = model
        self.stacked_params = self.stack_trainer.model_params

    def predict(self, new_feature, new_target = None, fitted = False):
        stackedX = self.stacked_dataset(new_feature, new_target, fitted = fitted)
        if self.un_pack:
            yhat = np.argmax(self.stacked_model.predict(stackedX), axis = -1)
        else:
            yhat = self.stacked_model.predict(stackedX)
        return yhat 

    def evaluate_model(
        self,
        n_splits = 3,
        n_repeats = 2,
        shuffle = True,
        random_state = 441,
        new_cv = None,
        metric_only = False
        ):
        if new_cv is None:
            self.initiate_cv(n_splits = n_splits, n_repeats = n_repeats, shuffle = shuffle, random_state = random_state)
        else:
            self.override_cv(new_cv)
        metric_result = []
        for train, test in self.cv_pair.split(self.feature, self.target):
            trainset = self.feature.iloc[train, :]
            trainset_label = self.target.iloc[train, :]
            testset = self.feature.iloc[test, :]
            testset_label = self.target.iloc[test, :]
            self.fit_stacked_model(trainset, trainset_label)
            if self.un_pack:
                yhat = np.argmax(self.stacked_model.predict(self.stacked_train_set), axis = -1)
            else:
                yhat = self.stacked_model.predict(self.stacked_train_set)
            train_rf_predictions = yhat
            # train_rf_probs = self.model.predict_proba(trainset)[:, 1]

            rf_predictions = self.predict(testset, fitted = True)
            # rf_probs = self.model.predict_proba(testset)[:, 1]
    
            if not metric_only:
                evaluate_model(rf_predictions, train_rf_predictions, trainset_label, testset_label, metric_only)
                cm = confusion_matrix(testset_label, rf_predictions)
                target_list = self.target.iloc[:,-1].unique()
                plot_confusion_matrix(cm, classes = target_list,
                            title = 'Confusion Matrix')
            else:
                metric = evaluate_model(rf_predictions, train_rf_predictions, trainset_label, testset_label, metric_only)
                metric_result.append(metric)
        return np.mean(metric_result) if metric_only else -1
    def bayes_optimize(
        self,
        search_space,
        n_splits = 3,
        shuffle = True,
        random_state = 441,
        max_evals = 1000,
        init_vals = []
        ):
        def obj_fnc(params):
            if self.un_pack:
                params["num_input_nodes"] = self.trainer_num
                self.stack_trainer.train_model(**params)
            else:
                self.stack_trainer.train_model(params)
            self.stacked_model = self.stack_trainer.model
            metric = self.evaluate_model(
                n_splits = n_splits,
                shuffle = shuffle,
                random_state = random_state,
                metric_only = True
            )
            return {"loss" : -metric, "status" : STATUS_OK}
        hyper_trials = Trials() if init_vals == [] else generate_trials_to_calculate(init_vals)
        best_params = fmin(obj_fnc, search_space, algo = tpe.suggest, 
        max_evals = max_evals, trials= hyper_trials)
        # current_params = self.get_model_params()
        # for feat in best_params.keys():
        #     current_params[feat] = search_space[feat].pos_args[best_params[feat] + 1]._obj
        print(best_params)
        print(hyper_trials.best_trial['result']['loss'])
        return best_params


class NNRegressor(NNTrainer):
    def __init__(self, feature, target, log_text, verbose):
        super(NNRegressor, self).__init__(feature, target, "NN Regressor", verbose, log_text)

    def train_model(
        self, 
        learning_rate,
        num_dense_layers,
        num_dense_nodes,
        dropout,
        activation,
        adam_decay,
        batch_size,
        epochs,
        num_input_nodes = None,
        num_output = None):
        if num_input_nodes is None:
            num_input_nodes = self.feature.shape[1]
        if num_output is None:
            num_output = len(self.target.iloc[:,-1].unique())
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape = (num_input_nodes)))
        for _ in range(num_dense_layers):
            model.add(keras.layers.Dense(num_dense_nodes, activation = activation))
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(num_output))
        model.compile(
            optimizer = keras.optimizers.Adam(lr = learning_rate, decay = adam_decay),
            loss = 'mean_squared_error'
        )
        self.model = model
        self.model_params = dict(
            learning_rate = learning_rate,
            num_input_nodes = num_input_nodes,
            num_output = num_output,
            num_dense_layers = num_dense_layers,
            num_dense_nodes = num_dense_nodes,
            dropout = dropout,
            activation = activation,
            adam_decay = adam_decay,
            batch_size = batch_size,
            epochs = epochs,
        )
    def evaluate_model(
        self,
        n_splits = 3,
        n_repeats = 2,
        shuffle = True,
        random_state = 441,
        new_cv = None,
        metric_only = False,
        target_col_index = -1
        ):
        if new_cv is None:
            self.initiate_cv(n_splits = n_splits, n_repeats = n_repeats, shuffle = shuffle, random_state = random_state)
        else:
            self.override_cv(new_cv)
        metric_result = []
        acc_result = []
        for train, test in self.cv_pair.split(self.feature, self.target):
            self.train_model(**self.model_params)
            trainset = self.feature.iloc[train, :]
            trainset_label = self.target.iloc[train]
            testset = self.feature.iloc[test, :]
            testset_label = self.target.iloc[test]
            self.history = self.model.fit(
                trainset, 
                trainset_label.values.ravel(), 
                epochs = self.model_params["epochs"], 
                batch_size = self.model_params["batch_size"],
                verbose = 0)

            pred = self.model.predict(testset)
            # rf_probs = self.model.predict_proba(testset)[:, 1]
            metric = mean_squared_error(testset_label, pred)
            metric_result.append(metric)
            acc = r2_score(testset_label, pred)
            acc_result.append(acc)
        return np.mean(metric_result), np.mean(acc_result)

    def bayes_optimize(
        self,
        search_space,
        n_splits = 3,
        n_repeats = 2,
        shuffle = True,
        random_state = 441,
        max_evals = 1000,
        init_vals = []
        ):
        def obj_fnc(params):
            self.train_model(**params)
            metric = self.evaluate_model(
                n_splits = n_splits,
                n_repeats = n_repeats,
                shuffle = shuffle,
                random_state = random_state,
            )
            return {"loss" : metric[0], "status" : STATUS_OK}
        hyper_trials = Trials() if init_vals == [] else generate_trials_to_calculate(init_vals)
        best_params = fmin(obj_fnc, search_space, algo = tpe.suggest, max_evals = max_evals, trials= hyper_trials)
        # current_params = self.get_model_params()
        # for feat in best_params.keys():
        #     current_params[feat] = search_space[feat].pos_args[best_params[feat] + 1]._obj
        print(best_params)
        print(hyper_trials.best_trial['result']['loss'])
        return best_params

    def predict(self, new_feature, new_target = None, fitted = False, target_col_index = -1):
        batch_size = self.model_params["batch_size"]
        epochs = self.model_params["epochs"]
        if not fitted:
            self.train_model(**self.model_params)
            if not isinstance(new_target, pd.core.frame.DataFrame):
                self.model.fit(
                    self.feature, 
                    self.target, 
                    batch_size = batch_size, 
                    class_weight = self.get_class_weights(target_col_index), 
                    epochs = epochs,
                    verbose = 0)
            else:
                self.model.fit(
                    new_feature, 
                    new_target, 
                    batch_size = batch_size, 
                    epochs = epochs,
                    verbose = 0)
        self.prediction = np.argmax(self.model.predict(new_feature), axis = -1)
        return self.prediction

    def predict_proba(self, new_feature, target_col_index = -1):
        batch_size = self.model_params["batch_size"]
        epochs = self.model_params["epochs"]
        self.model.fit(
            self.feature, 
            self.target, 
            batch_size = batch_size, 
            class_weight = self.get_class_weights(target_col_index), 
            epochs = epochs,
            verbose = 0)
        self.prediction_proba = self.model.predict(new_feature)
        return self.prediction_proba


class XGRegressor(Regressor):
    def __init__(self, feature, target, log_text):
        super(XGRegressor, self).__init__(feature, target, "XGBoost Regressor", 1, log_text)
    def train_model(self, params = {}):
        self.model = XGBRegressor(**params)
        ''' 
        Sample Parameters:
        XGBClassifier(learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softmax',
                    nthread=4,
                    num_class=9,
                    seed=27)
        '''
        self.model_params = self.model.get_params()

class LGBoundRegressor(Regressor):
    def __init__(self, feature, target, log_text):
        super(LGBoundRegressor, self).__init__(feature, target, "LG Regressor", 1, log_text)
    def train_model(self, params = {}):
        self.model = LogisticRegression(**params)
        self.model_params = self.model.get_params()

class RFRegressor(Regressor):
    def __init__(self, feature, target, log_text):
        super(RFRegressor, self).__init__(feature, target, "RF Bound Regressor", 1, log_text)
    def train_model(self, params = {}):
        self.model = RandomForestRegressor(**params)
        self.model_params = self.model.get_params()

class NNRegressor(Regressor):
    def __init__(self, feature, target, log_text, verbose):
        super(NNRegressor, self).__init__(feature, target, "NN Regressor", verbose, log_text)
    def train_model(
        self, 
        learning_rate,
        num_dense_layers,
        num_dense_nodes,
        dropout,
        activation,
        adam_decay,
        batch_size,
        epochs,
        num_input_nodes = None,
        num_output = None):
        if num_input_nodes is None:
            num_input_nodes = self.feature.shape[1]
        if num_output is None:
            num_output = 1
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape = (num_input_nodes)))
        for _ in range(num_dense_layers):
            model.add(keras.layers.Dense(num_dense_nodes, activation = activation))
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(num_output,  activation = "linear"))
        model.compile(
            optimizer = keras.optimizers.Adam(lr = learning_rate, decay = adam_decay),
            loss = 'mse'
        )
        self.model = model
        self.model_params = dict(
            learning_rate = learning_rate,
            num_input_nodes = num_input_nodes,
            num_output = num_output,
            num_dense_layers = num_dense_layers,
            num_dense_nodes = num_dense_nodes,
            dropout = dropout,
            activation = activation,
            adam_decay = adam_decay,
            batch_size = batch_size,
            epochs = epochs,
        )

    def evaluate_model(
        self,
        n_splits = 3,
        shuffle = True,
        random_state = 441,
        new_cv = None,
        ):
        if new_cv is None:
            self.initiate_cv(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
        else:
            self.override_cv(new_cv)
        metric_result = []
        acc_result = []
        for train, test in self.cv_pair.split(self.feature, self.target):
            self.train_model(**self.model_params)
            trainset = self.feature.iloc[train, :]
            trainset_label = self.target.iloc[train]
            testset = self.feature.iloc[test, :]
            testset_prob = self.target.iloc[test]
            self.model.fit(
                trainset, 
                trainset_label.values.ravel(), 
                epochs = self.model_params["epochs"], 
                batch_size = self.model_params["batch_size"],
                verbose = 0)

            pred_prob = self.model.predict(testset)

            metric = mean_squared_error(testset_prob, pred_prob)
            acc_result.append(r2_score(testset_prob, pred_prob))
            metric_result.append(metric)
        return np.mean(metric_result), np.mean(acc_result)

    def bayes_optimize(
        self,
        search_space,
        n_splits = 3,
        shuffle = True,
        random_state = 441,
        max_evals = 1000,
        init_vals = []
        ):
        def obj_fnc(params):
            self.train_model(**params)
            metric = self.evaluate_model(
                n_splits = n_splits,
                shuffle = shuffle,
                random_state = random_state,
            )
            return {"loss" : metric[0], "status" : STATUS_OK}
        hyper_trials = Trials() if init_vals == [] else generate_trials_to_calculate(init_vals)
        best_params = fmin(obj_fnc, search_space, algo = tpe.suggest, max_evals = max_evals, trials= hyper_trials)
        # current_params = self.get_model_params()
        # for feat in best_params.keys():
        #     current_params[feat] = search_space[feat].pos_args[best_params[feat] + 1]._obj
        print(best_params)
        print(hyper_trials.best_trial['result']['loss'])
        return best_params

    def predict(self, new_feature, new_target = None, fitted = False, target_col_index = -1):
        batch_size = self.model_params["batch_size"]
        epochs = self.model_params["epochs"]
        if not fitted:
            self.train_model(**self.model_params)
            if not isinstance(new_target, pd.core.frame.DataFrame):
                self.model.fit(
                    self.feature, 
                    self.target, 
                    batch_size = batch_size, 
                    epochs = epochs,
                    verbose = 0)
            else:
                self.model.fit(
                    new_feature, 
                    new_target, 
                    batch_size = batch_size, 
                    epochs = epochs,
                    verbose = 0)
        self.prediction = np.argmax(self.model.predict(new_feature), axis = -1)
        return self.prediction

    def predict_proba(self, new_feature, target_col_index = -1):
        batch_size = self.model_params["batch_size"]
        epochs = self.model_params["epochs"]
        self.model.fit(
            self.feature, 
            self.target, 
            batch_size = batch_size, 
            epochs = epochs,
            verbose = 0)
        self.prediction_proba = self.model.predict(new_feature)
        return self.prediction_proba