import sys
import pandas as pd
import numpy as np
import timeit
import Constants
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

# This method is used to categorize the data
# Input: Data Frame,List of Categories
# Output: Categorized Data
def Data_Pre_Processing(data,list):
    categorizer=preprocessing.LabelEncoder()
    for i in list:
        data[i]=categorizer.fit_transform(data[i])
    return data

def Splitting_Data_On_Stages(data,won,lost):
    data_train=data[(data.Stage == won) | (data.Stage == lost)]
    data_test=data[~((data.Stage == won) | (data.Stage == lost))]
    data_test=data_test.reset_index(drop=True)
    data_train=data_train.reset_index(drop=True)
    return data_train,data_test
# This method is used for splitting dependent and independent variables
# Input: Training and Testing Data, Categorizer (Categories that needs to be classified) and Identifier(that should be avoided in  modelling)
# Output: Returns DataFrames of Dependent and In-Dependent variable
def Model_Data_Preperation(train_data,test_data,Categorizer,Identifier):
    headers=list(train_data.columns.values)
    headers.remove(Categorizer)
    X_dataframe,X_df=Model_Data_Preperation_Slicing(headers,train_data,Identifier)
    Categorizer_list=[]
    Categorizer_list.append(Categorizer)
    y_df=train_data[Categorizer_list]
    headers1=list(test_data.columns.values)
    headers1.remove(Categorizer)
    X_pred_dataframe,X_pred=Model_Data_Preperation_Slicing(headers1,test_data,Identifier)
    return X_dataframe,X_df,X_pred_dataframe,X_pred,y_df

def Model_Data_Preperation_Slicing(headers1,data,Identifier):
    X_dataframe=data[headers1]
    headers1.remove(Identifier)
    X_df=X_dataframe[headers1]
    return X_dataframe,X_df

# This method is used to select best classifier algorithm suited for the data set
# Input: Dependendent,Independent variables and the values of dependent variables that has to be classified
# Output:Predicted values
def ClassifierAlgortithms(X,y,X_pred):
    options = {0 : Knn,1 : Logistics,2 : RandomForest,3 : GradientBoosting}
    j=[]
    j=range(0,4)
    recall=[]
    for i in j:
        recalls,_,_=options[i](X,y,X_pred)
        recall.append(recalls)
    option_index=recall.index(max(recall))
    recall,prediction,proba_df=options[option_index](X,y,X_pred)
    print "Recal Value:"
    print recall
    return prediction,proba_df
# This method is used to select best regressor algorithm suited for the data set
# Input: Dependendent,Independent variables and the values of dependent variables that has to be classified
# Output:Predicted values
def RegressorAlgorithms(X,y,X_pred):
    options = {0 : Regression,1 : Gbm_Regressor,2 : Randomforest_Regressor,3 : Bag_Regression}
    j=[]
    j=range(0,4)
    score=[]
    for i in j:
        scores,_=options[i](X,y,X_pred)
        score.append(scores)
    option_index=score.index(max(score))
    score,prediction=options[option_index](X,y,X_pred)
    prediction=prediction.astype(int)
    print "R Square Value:"
    print score
    return prediction
# This method does Knearest Neighbour classification
# Input: Dependendent,Independent variables and the values of dependent variables that has to be classified
# Output:Predicted values
def Knn(X,y,X_pred):
    k_range = range(1, 10)
    param_grid = dict(n_neighbors=k_range)
    knn=KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    grid.fit(X, y)
    model_accuracy=grid.best_estimator_.predict(X)
    recal=recall_score(y,model_accuracy)
    y_pred_knn=grid.best_estimator_.predict(X_pred)
    proba_x_pred=grid.predict_proba(X_pred)
    proba_df=pd.DataFrame(proba_x_pred)
    return recal,y_pred_knn,proba_df
# This method does Logistic Regression
# Input: Dependendent,Independent variables and the values of dependent variables that has to be classified
# Output:Predicted values
def Logistics(X,y,X_pred):
    logistics = LogisticRegression()
    pca=decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistics', logistics)])
    n_components = [1, 2, 3, 4, 5]
    Cs = np.logspace(-4, 4, 3)
    estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistics__C=Cs))
    estimator.fit(X, y)
    model_accuracy=estimator.best_estimator_.predict(X)
    recal=recall_score(y,model_accuracy)
    y_pred_logistics=estimator.best_estimator_.predict(X_pred)
    proba_x_pred=estimator.predict_proba(X_pred)
    proba_df=pd.DataFrame(proba_x_pred)
    return recal,y_pred_logistics,proba_df
# This method does Random Forest classification
# Input: Dependendent,Independent variables and the values of dependent variables that has to be classified
# Output:Predicted values
def RandomForest(X,y,X_pred):
    rf=RandomForestClassifier()
    n_estimators_range=range(1,5)
    param_grid = dict(n_estimators=n_estimators_range)
    grid_random_forest = GridSearchCV(rf, param_grid, cv=10, scoring='accuracy')
    grid_random_forest.fit(X, y)
    model_accuracy=grid_random_forest.best_estimator_.predict(X)
    recal=recall_score(y,model_accuracy)
    y_pred_randomforest=grid_random_forest.best_estimator_.predict(X_pred)
    proba_x_pred=grid_random_forest.predict_proba(X_pred)
    proba_df=pd.DataFrame(proba_x_pred)
    return recal,y_pred_randomforest,proba_df
# This method does Gradient Boosting classification
# Input: Dependendent,Independent variables and the values of dependent variables that has to be classified
# Output:Predicted values
def GradientBoosting(X,y,X_pred):
    gbm=GradientBoostingClassifier()
    n_estimators_range=range(1,10)
    learning_rate= [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]
    param_grid=dict(n_estimators=n_estimators_range,learning_rate=learning_rate)
    grid_gbm=GridSearchCV(gbm,param_grid,cv=10,scoring='accuracy')
    grid_gbm.fit(X,y)
    model_accuracy=grid_gbm.best_estimator_.predict(X)
    recal=recall_score(y,model_accuracy)
    y_pred_gradient_boosting=grid_gbm.best_estimator_.predict(X_pred)
    proba_x_pred=grid_gbm.predict_proba(X_pred)
    proba_df=pd.DataFrame(proba_x_pred)
    return recal,y_pred_gradient_boosting,proba_df
# This method is used for slicing  data which are actually preedicted as won and have age greater than zero
# Input: Data,Classified Column and the regressor
# Output: Sliced Data
def Data_For_Regression(data,Predicted,Regressor):
    Category=Predicted
    data=data[(data[Category] == 1)]
    data=data[(data[Regressor] != 0)]
    return data
# This method is used to split the trainin and testing set
# Input: Data,Won case, Categorizer,Important Features
# Output: Training and Testing set
def Regression_Data_Split(data,Won_case,Categorizer,DATA_PRE_PROCESSING):
    data=data[DATA_PRE_PROCESSING]
    train_data=data[(data[Categorizer] == Won_case)]
    test_data=data[~(data[Categorizer] == Won_case)]
    train_data=train_data.reset_index(drop=True)
    test_data=test_data.reset_index(drop=True)
    return  train_data,test_data
# This method is used to split the data as dependent and independt variables suitable for modelling
# Input: Training data, Testing Data, Regressor feature and
# Output: Data frames suitable for modelling
def Regressor_Data_Preperation(train_data,test_data,Regressor,Identifier):
    headers=list(train_data.columns.values)
    headers.remove(Regressor)
    X_dataframe,X_df=Model_Data_Preperation_Slicing(headers,train_data,Identifier)
    Regressor_list=[]
    Regressor_list.append(Regressor)
    y_df=train_data[Regressor_list]
    headers1=list(test_data.columns.values)
    headers1.remove(Regressor)
    X_pred_dataframe,X_pred=Model_Data_Preperation_Slicing(headers1,test_data,Identifier)
    return X_dataframe,X_df,X_pred_dataframe,X_pred,y_df
# This method is used to do feature engineering
# Input: Numpy array of Dependendt and Independent Variables
# Output: Numpy array of important Dependent variables
def Select_Features_For_Regression(X,y,X_pred):
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X=sel.fit_transform(X)
    X_pred=sel.transform(X_pred)
    clf = LinearRegression()
    sfm = SelectFromModel(clf, threshold=0.05)
    sfm_pred=SelectFromModel(clf, threshold=0.05)
    fm_fit=sfm.fit(X, y)
    X_new=Feature_Extraction(X,y,sfm,fm_fit)
    fm_fit=sfm_pred.fit(X,y)
    X_pred=Feature_Extraction(X_pred,y,sfm_pred,fm_fit)
    return X_new,X_pred
# This method is used to do feature extraction
# Input: Numpy array of Dependendt and Independent Variables and models of feature extraction
# Output: Re-shaped Features
def Feature_Extraction(X,y,sfm,fm_fit):
    n_features = sfm.transform(X).shape[1]
    disclosed_features=n_features-1
    while n_features > disclosed_features:
        sfm.threshold += 0.1
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]
    return X_transform
# This method is used to normalize the features to reduce the error
# Input: Independent variables of training and testing set
# Output: Normalized independent variables
def Normalizing_Features(X,X_pred):
    max_abs_scaler = preprocessing.MaxAbsScaler()
    X=max_abs_scaler.fit_transform(X)
    X_pred=max_abs_scaler.transform(X_pred)
    return X,X_pred
# This method is used to do linear regression to find the age of the deal
# Input: Dependendent,Independent variables and the values of dependent variables that has to be classified
# Output:Predicted values
def Regression(X,y,X_pred):
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict(X)
    score=r2_score(y, prediction)
    result=model.predict(X_pred)
    return score,result
# This method is used to do GBM regression to find the age of the deal
# Input: Dependendent,Independent variables and the values of dependent variables that has to be classified
# Output:Predicted values
def Gbm_Regressor(X,y,X_pred):
    gbm=GradientBoostingRegressor()
    gbm.fit(X,y)
    score=gbm.score(X,y)
    y_pred_gradient_boosting=gbm.predict(X_pred)
    return score,y_pred_gradient_boosting
# This method is used to do random forest regression to find the age of the deal
# Input: Dependendent,Independent variables and the values of dependent variables that has to be classified
# Output:Predicted values
def Randomforest_Regressor(X,y,X_pred):
    rf_regressor=RandomForestRegressor(n_estimators=10)
    rf_regressor.fit(X,y)
    score=rf_regressor.score(X,y)
    y_pred_random_forest=rf_regressor.predict(X_pred)
    return score,y_pred_random_forest
# This method is used to do bag regression to find the age of the deal
# Input: Dependendent,Independent variables and the values of dependent variables that has to be classified
# Output:Predicted values
def Bag_Regression(X,y,X_pred):
    bag=BaggingRegressor()
    bag.fit(X,y)
    score=bag.score(X,y)
    y_pred_bag=bag.predict(X_pred)
    return score,y_pred_bag


# This method is used to split the data in required format as requested for einsights system
def Final_slicing(data,Identifier,Predicted_category,Regressor_op,Probablity):
        headers=[]
        headers.append(Identifier)
        headers.append(Predicted_category)
        headers.append(Regressor_op)
        headers.append(Probablity)
        data=data[headers]
        final_headers=list(data.columns.values)
        final_headers.remove(Predicted_category)
        final_headers.remove(Regressor_op)
        final_headers.remove(Probablity)
        final_headers.append('Predicted Category')
        final_headers.append('Predicted Date')
        final_headers.append('Won Probability')
        data.columns=final_headers
        return data




def main():
    start=timeit.default_timer()
    # Reading Single Flat file from CRM System for Analytics
    for arg in sys.argv[1:]:
        location=arg
    DATA_PRE_PROCESSING=Constants.DATA_PRE_PROCESSING
    WON_CASE=Constants.WON_CASE
    LOST_CASE=Constants.LOST_CASE
    Categorizer=Constants.Categorizer
    Identifier=Constants.Identifier
    Predicted_Category=Constants.Predicted_Category
    Regressor=Constants.Regressor
    Regressor_op=Constants.Regressor_op
    Date=Constants.Date
    Probablity=Constants.Probablity
    #location="/Users/aj/Desktop/Project_Automation/StepChangeData_1.csv"
    data=pd.read_csv(location)
    data_new=data.copy()
    # In this file there is ambiguity in stage names which was avoided
    data_new=data_new.replace({'Proposals lost': 'Lost overall'}, regex=True)
    # Slicing the features required
    data_new=data_new[DATA_PRE_PROCESSING]
    # Splitting data in random as training data and testing data
    data_train,data_test=Splitting_Data_On_Stages(data_new,WON_CASE,LOST_CASE)
    # Categorizing the features
    categories_list = list(data_new.select_dtypes(include=['object']).columns)
    data_train=Data_Pre_Processing(data_train,categories_list)
    data_test=Data_Pre_Processing(data_test,categories_list)
    final_headers=list(data_test.columns.values)
    # Splitting Dependent and Independent Variables
    X_dataframe,X_df,X_pred_dataframe,X_pred,y_df=Model_Data_Preperation(data_train,data_test,Categorizer,Identifier)
    # Making all the dataframes as Numpy array that enables to be fed in the model
    X=np.array(X_df.as_matrix(columns = None))
    y=np.array(y_df[Categorizer].values)
    X_pred=np.array(X_pred.as_matrix())
    # Classification Algorithm
    y_pred,proba_df=ClassifierAlgortithms(X,y,X_pred)
    test_id=data_test[Identifier]
    proba_df=proba_df.join(test_id)
    data_test[Predicted_Category]=y_pred
    data_train[Predicted_Category]=data_train[Categorizer]
    merged_data=pd.merge(data_train,data_test,how='outer')
    merged_data=pd.merge(merged_data,proba_df,how='left',left_on=Identifier,right_on=Identifier)
    merged_data=merged_data.reset_index(drop=True)
    merged_data=merged_data[[Identifier,Predicted_Category,Probablity]]
    data=pd.merge(data,merged_data,how='left',left_on=Identifier,right_on=Identifier)
    data=Data_For_Regression(data,Predicted_Category,Regressor)
    data=data.reset_index(drop=True)
    data_train,data_test=Regression_Data_Split(data,WON_CASE,Categorizer,DATA_PRE_PROCESSING)
    categories_list = list(data_train.select_dtypes(include=['object']).columns)
    data_train=Data_Pre_Processing(data_train,categories_list)
    data_test=Data_Pre_Processing(data_test,categories_list)
    X_dataframe,X_df,X_pred_dataframe,X_pred,y_df=Regressor_Data_Preperation(data_train,data_test,Regressor,Identifier)
    X=np.array(X_df.as_matrix(columns = None))
    y=np.array(y_df[Regressor].values)
    X_pred=np.array(X_pred.as_matrix())
    X,X_pred=Select_Features_For_Regression(X,y,X_pred)
    X,X_pred=Normalizing_Features(X,X_pred)
    y_pred=RegressorAlgorithms(X,y,X_pred)
    y_pred[y_pred < 0] = 0
    data_train,data_test=Regression_Data_Split(data,WON_CASE,Categorizer,DATA_PRE_PROCESSING)
    data_test[Regressor_op]=y_pred
    merged_data=pd.merge(data_train,data_test,how='outer')
    new_header=[]
    new_header.append(Identifier)
    new_header.append(Regressor_op)
    merged_data=merged_data[new_header]
    data=pd.merge(data,merged_data,how='left',left_on=Identifier,right_on=Identifier)
    data[Date]=pd.to_datetime(data[Date],dayfirst=True)
    data[Regressor_op] = data[Date] + pd.TimedeltaIndex(data[Regressor_op], unit='D')
    actual_data=pd.read_csv("/Users/aj/PycharmProjects/StepChange/StepChange_1.csv")
    data=Final_slicing(data,Identifier,Predicted_Category,Regressor_op,Probablity)
    actual_data=pd.merge(actual_data,data,how='left',left_on=Identifier,right_on=Identifier)
    actual_data.to_csv("StepChangePrediction.csv",index=False)
    stop=timeit.default_timer()
    print "Algorithm Running Time of Analytical Engine:"
    print stop-start

if __name__ == "__main__": main()