import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from joblib import dump, load

class E2epp():

    """
    Predictor E2epp which use multiple decision tree to to predict accuracy of CNN
    """
    def __init__(self, nb_trees):
        self.trees = np.zeros(nb_trees)
        self.features = np.zeros(nb_trees)



    def predictor_training(self,training_data):
        """
        Train the random forest with some CNN already trained
        """

        for tree in range(len(self.trees)):



            self.trees[tree] = DecisionTreeRegressor()
            self.trees[tree].fit()

        self.save_model()

    def predict_performance(self):
        """
        Predict the accuracy of a CNN
        """
        pass

    def save_model(self):
        """
        save model
        Returns
        -------

        """
        model = [self.trees,self.features]
        joblib.dump(model, 'E2eep_model.pkl')

    def load_saved_model(self):
        """
        load model
        Returns
        -------

        """
        self.model, self.features = joblib.load('E2eep_model.pkl')