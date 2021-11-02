import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from joblib import dump, load

class E2epp():

    """
    Predictor E2epp which use multiple decision tree to to predict accuracy of CNN
    """
    def __init__(self, nb_trees,training_data):
        self.trees = np.zeros(nb_trees)
        self.features = np.zeros(nb_trees)
        self.predictor_training(training_data=training_data)



    def load_data(self,data):
        d = pd.read_csv(data).values
        data = d[:, 1:-1]
        label = d[:, -1]
        return data, label

    def make_decision_trees(self,train_data, train_label):
        feature_record = []

        for tree in range(len(self.trees)):
            sample_idx = np.arange(train_data.shape[0])
            np.random.shuffle(sample_idx)
            train_data = train_data[sample_idx, :]
            train_label = train_label[sample_idx]

            feature_idx = np.arange(train_data.shape[1])
            np.random.shuffle(feature_idx)
            n_feature = np.random.randint(1, train_data.shape[1] + 1)
            selected_feature_ids = feature_idx[0:n_feature]
            print(n_feature)
            feature_record.append(selected_feature_ids)

            self.trees[tree] = DecisionTreeRegressor()
            self.trees[tree].fit(train_data[:, selected_feature_ids], train_label)
        return feature_record
    def predictor_training(self,training_data):
        """
        Train the random forest with some CNN already trained
        """
        data,label = self.load_data("e2epp.txt")
        features = self.make_decision_trees(data,label)
        self.save_model(features)

    def predict_performance(self, net):
        """
        Predict the accuracy of a CNN
        """
        pass

    def save_model(self,features):
        """
        save model
        Returns
        -------

        """
        model = [self.trees, features]
        joblib.dump(model, 'E2eep_model.pkl')

    def load_saved_model(self):
        """
        load model
        Returns
        -------

        """
        self.model, self.features = joblib.load('E2eep_model.pkl')