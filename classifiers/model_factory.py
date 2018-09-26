from classifiers import deep_dense_network,Random_forest,SVM

class model_factory:
    def __init__(self,model):
        self.model = model

    def factory(self):

        if (self.model=="random_forest"):
            return Random_forest.Random_forest()

        if (self.model=="neural_networks"):
            return deep_dense_network.Neural_network()

        if (self.model=="SVM"):
            return SVM.SVM()
        else:
            print ("write correct spell")

    def fit_predict(self,train_data,train_labels,test_data):
        model = self.factory()
        predictions = model.fit_predict(train_data,train_labels,test_data)
        return predictions