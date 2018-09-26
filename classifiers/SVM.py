from sklearn import svm

class SVM():


    def fit_predict(self,train_data,train_labels,test_data):
        model_obj = svm.SVC()
        model_obj.fit(train_data,train_labels)
        return model_obj.predict(test_data)