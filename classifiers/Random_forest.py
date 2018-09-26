from sklearn.ensemble import RandomForestClassifier

class Random_forest():

    def fit_predict(self,train_data,train_labels,test_data):
        model_obj = RandomForestClassifier()
        model_obj.fit(train_data,train_labels)
        return model_obj.predict(test_data)
