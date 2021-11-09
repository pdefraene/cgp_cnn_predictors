import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

class E2epp():

    """
    Predictor E2epp which use multiple decision tree to to predict accuracy of CNN
    """
    def __init__(self, nb_trees,training_data):
        self.trees = np.zeros(nb_trees,dtype=DecisionTreeRegressor)
        self.features = np.zeros(nb_trees)
        self.test_one_this_run(training_data)



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

    def predict(self,test_data, feature_ids):
        """
        give the list of prediction for each tree

        Parameters
        ----------
        feature_ids

        Returns
        -------

        """
        predict_list = []
        for tree, feature in zip(self.trees, feature_ids):
            predict_y = tree.predict(test_data[:, feature])
            predict_list.append(predict_y)
        return predict_list

    def test_one_this_run(self,training_data):
        data, label = self.load_data(training_data)
        idx = np.arange(label.shape[0])
        #     np.random.shuffle(idx)
        #     data = data[idx,:]
        #     label = label[idx]
        train_num = int(idx.shape[0] * 0.8)
        print("Train num : ", train_num)
        train_data = data[0:train_num, :]
        train_label = label[0:train_num]
        test_data = data[train_num:, :]
        test_label = label[train_num:]
        self.one_run_for_each(train_data, train_label, test_data, test_label)

    def one_run_for_each(self,train_data, train_label, test_data, test_label):
        n_tree = 1000
        features = self.make_decision_trees(train_data, train_label)
        test_num = test_data.shape[0]
        predict_labels = np.zeros(test_num)
        for i in range(test_num):
            this_test_data = test_data[i, :]
            predict_this_list = np.zeros(n_tree)

            for j, (tree, feature) in enumerate(zip(self.trees, features)):
                predict_this_list[j] = tree.predict([this_test_data[feature]])[0]

            # find the top 100 prediction
            predict_this_list = np.sort(predict_this_list)
            predict_this_list = predict_this_list[::-1]
            this_predict = np.mean(predict_this_list[0:100])
            predict_labels[i] = this_predict
        print(np.sqrt(np.mean(np.square(predict_labels - test_label))))
        print(np.mean(np.abs(predict_labels - test_label)))

        plt.plot(predict_labels, label='predict')
        plt.plot(test_label, label='true')
        plt.legend()
        plt.show()

    def predict_performance(self, net):
        """
        Predict the accuracy of a CNN
        """
        pass

    def train_final_model(self,training_data):
        """
        train the final model and save it to be able to load it without training again
        Parameters
        ----------
        training_data : training data for train the random forest

        Returns
        -------

        """
        self.trees = np.zeros(5000,dtype=DecisionTreeRegressor)
        train_data, train_label = self.load_data(training_data)
        features = self.make_decision_trees(train_data, train_label)
        model = [self.trees, features]
        joblib.dump(model, 'E2eep_model.pkl')

    def load_saved_model(self):
        """
        load model
        Returns
        -------

        """
        self.model, self.features = joblib.load('E2eep_model.pkl')