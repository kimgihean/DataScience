"""
'Eccomerce Shipping Data'

This file is finding best score function after preprocessing yourself
    Algorithm Method
        Classifing  : Decision Tree, KNN, LogisticRegresison
        Regression  : Polynomial Regression, Multiple Regression
        Ensemble    : Random Forest, Gradient Boosting
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, r2_score, auc, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import time


def autoML(data, X, y):
    """
    This function automatically combines and executes 
        Scaling, Modeling, and Evaluation, 
    and returns 
        the result in the type of a DataFrame.

    Args:
        data (DataFrame) : Data in the type of a DataFrame after removing dirty data
                          , The target data is binary classification.
        X (Training dataset)
        y (Target dataset)

    Returns:
        Data frame : DataFrame with columns 
                    'Model', 
                    'Scaler', 
                    'Parameter_dict', 
                    'score_cv', 
                    'score_auc', 
                    'score_ac', 
                    'score_r2', 
                    'runtime', 
                    'model_'.
                    
    Example:
    >>> import OpenSource_12_김서영_김부경_김기현_김원종 as test
    >>> data.autoML()
    >>> <function OpenSource_12_김서영_김부경_김기현_김원종.autoML(data, X, y)>
    >>> data.autoML(DataFrame, Training, Test)
    >>> Model            Scaler                        Parameter_dict  \
92   RandomForest  StandardScaler()   {'n_estimators': 5, 'max_depth': 3}   
145  RandomForest    MinMaxScaler()   {'n_estimators': 8, 'max_depth': 4}   
157  RandomForest    MinMaxScaler()  {'n_estimators': 11, 'max_depth': 4}   
188  RandomForest    RobustScaler()   {'n_estimators': 9, 'max_depth': 3}   
108  RandomForest  StandardScaler()   {'n_estimators': 9, 'max_depth': 3}   

     score_cv  score_auc  score_ac score_r2   runtime  \
92   0.690480   0.737257  0.685152           0.021019   
145  0.688402   0.732980  0.685152           0.034000   
157  0.688142   0.723200  0.682424           0.042003   
188  0.687752   0.719793  0.683939           0.032011   
108  0.687493   0.736067  0.686061           0.034000   

                                                model_  
92   (DecisionTreeClassifier(max_depth=3, max_featu...  
145  (DecisionTreeClassifier(max_depth=4, max_featu...  
157  (DecisionTreeClassifier(max_depth=4, max_featu...  
188  (DecisionTreeClassifier(max_depth=3, max_featu...  
108  (DecisionTreeClassifier(max_depth=3, max_featu...  
    """
    
    scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]
    regression_name = ['polynomialRegression', 'multipleRegression']
    classification_name = ['DecisionTree(gini)', 'DecisionTree(entropy)', 'KNeighborsClassifier', 'LogisticRegression']
    ensemble_name = ['RandomForest','GradientBoosting']

    result_regression = []
    result_classification = []
    result_ensemble = []
    result = []
    
    def Scaling(X_train, X_test, scaler):
        scaled_train = pd.DataFrame(scaler.fit_transform(X_train))
        scaled_test = pd.DataFrame(scaler.fit_transform(X_test))
        return scaled_train, scaled_test

    def Classifier(X, y, model):
        return_res=[]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        for i in scalers:
            X_train, X_test = Scaling(X_train, X_test, i)

            param = range(2, 12)
            for p in param:
                result = []
                t = time.time()
                result.append(model)
                result.append(i)
                
                if model.find("KNeighbor")!=-1 and p%2==1:
                    m = KNeighborsClassifier(n_neighbors=p)
                    result.append({'n_neighbors':p})
                    
                elif model.find("Tree")!=-1:
                    if model.find("gini")!=-1:
                        m = DecisionTreeClassifier(criterion = 'gini', max_depth = p)
                        result.append({'criterion':'gini', 'max_depth':p})
                    elif model.find("entropy")!=-1:
                        m = DecisionTreeClassifier(criterion = 'entropy', max_depth = p)
                        result.append({'criterion':'entropy', 'max_depth':p})
                elif model.find("Logistic")!=-1 and p==2:
                    m = LogisticRegression()
                    result.append('')
                    
                else:
                    continue


                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                
                runtime = time.time() - t
                
                scores = cross_val_score(m, X, y, cv=5)
                score_cv = scores.mean()
                fper, tper, thresholds = roc_curve(y_test, y_pred)
                score_auc = auc(fper, tper)
                score_ac = accuracy_score(y_test, y_pred)
                score_r2 = ''
                
                result.extend([score_cv, score_auc, score_ac, score_r2, runtime, m])
                return_res.append(result)
                
        return return_res
    def Regression(X, y, model):
        return_res=[]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
        for i in scalers:
            result = []
            t = time.time()
            
            X_train, X_test = Scaling(X_train, X_test, i)

            if model == 'polynomialRegression':
                poly = PolynomialFeatures(degree = 2, include_bias = True)
                X_train_ = poly.fit_transform(X_train)
                X_test_ = poly.fit_transform(X_test)
            elif model == 'multipleRegression':
                X_train_ = X_train
                X_test_ = X_test
            else :
                continue
            
            result.extend([model, i, ''])
            model = LinearRegression()
            
            model.fit(X_train_, y_train)
            y_pred = model.predict(X_test_)
            runtime = time.time() - t
            
            scores = cross_val_score(model, X_train_, y_train, cv=5)
            score_cv = scores.mean()
            fper, tper, thresholds = roc_curve(y_test, y_pred)
            score_auc = auc(fper, tper)
            score_ac = ''
            score_r2 = r2_score(y_test, y_pred)

            result.extend([score_cv, score_auc, score_ac, score_r2, runtime, model])
            return_res.append(result)
                
        return return_res
    
    def Ensemble(X, y, model):
        return_res=[]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        for i in scalers:
            X_train, X_test = Scaling(X_train, X_test, i)

            n_estimator_range = range(2, 12)
            max_depth_range = range(3,7)
            
            for n_estimator_param in n_estimator_range:
                for max_depth_param in max_depth_range:
                    result = []
                    
                    param_dict = {}
                    t = time.time()
                    result.append(model)
                    result.append(i)
                    
                    param_dict = {'n_estimators': n_estimator_param, 'max_depth': max_depth_param}

                    if model == 'RandomForest':
                        m = RandomForestClassifier(n_estimators = n_estimator_param, max_depth = max_depth_param)
                    elif model == 'GradientBoosting':
                        m = GradientBoostingClassifier(learning_rate = 1, n_estimators = n_estimator_param, max_depth = max_depth_param)
                    else:
                        continue
                    
                    result.append(param_dict)

                    m.fit(X_train, y_train)
                    y_pred = m.predict(X_test)
                    runtime = time.time() - t
                    
                    scores = cross_val_score(m, X_train, y_train, cv=5)
                    score_cv = scores.mean()
                    fper, tper, thresholds = roc_curve(y_test, y_pred)
                    score_auc = auc(fper, tper)
                
                    score_ac = accuracy_score(y_test, y_pred)
                    score_r2 = ''
                
                    result.extend([score_cv, score_auc, score_ac, score_r2, runtime, m])
                    return_res.append(result)
                
        return return_res
 

    for model in regression_name:
        result_regression.append(Regression(X, y, model))
    result_regression = sum(result_regression, [])

    for model in classification_name:
        result_classification.append(Classifier(X, y, model))
    result_classification = sum(result_classification, [])

    for model in ensemble_name:
        result_ensemble.append(Ensemble(X, y, model))
    result_ensemble = sum(result_ensemble, [])

    final_list = result_regression+ result_classification + result_ensemble

    final_result = pd.DataFrame(final_list, columns=
                        ['Model', 'Scaler', 'Parameter_dict', 'score_cv', 'score_auc', 'score_ac', 'score_r2', 'runtime', 'model_'])

    return final_result
    

def best_n(df, ft_name, n):
    """
    This function automatically Find best score
    and returns DataFrame after descending sorting.

    Args:
        df (DataFrame)      : Final result(final_result) after all process
        ft_name (String)    : What kinds of scoring (cv, auc, ac, r2)
        n (integer)         : The highest rank in order n (1 ~ n)

    Returns:
        Data frame
    """
    df.sort_values(by = ft_name, ascending = False, inplace = True)
    return df[:n]
    

"""
[ The Feature ]
ID                  : ID Number of Customers
Warehouse block     : The Company have big Warehouse which is divided in to block 
                      such as A, B, C, D, F.
Mode of shipment    : The Company Ships the products in multiple way such as Ship, Flight and Road.
Customer care calls : The number of calls made from enquiry for enquiry of the shipment.
Customer rating     : The company has rated from every customer. 
                      1 is the lowest (Worst), 5 is the highest (Best).
Cost of the product : Cost of the Product in US Dollars
Prior purchases     : The Number of Prior Purchase
Product importance  : The company has categorized the product in the various parameter 
                      such as low, medium, high
Gender              : Male and Female
Discount offered    : Discount offered on that specific product
Weight in gms       : It is the weight in grams
Reached on time     : It is the target variable, 
                      where 1 Indicates that the product has NOT reached on time
                      and 0 indicates it has reached on time

[ The Target Feature ]
Reached on time
"""

    
# main code
data = pd.read_csv("/ecommerce_shipping.csv")
target_feature = 'Reached.on.Time_Y.N'
other_feature = ['Warehouse_block', 
                'Mode_of_Shipment',
                'Customer_care_calls',
                'Customer_rating',
                'Cost_of_the_Product',
                'Prior_purchases',
                'Product_importance',
                'Gender',
                'Discount_offered',
                'Weight_in_gms'
                 ]

label_encoder = LabelEncoder()
cateogical_feature_name = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
for feature_name in cateogical_feature_name:
    data[feature_name] = label_encoder.fit_transform(data[feature_name])

y = data[target_feature]
X = data.drop(target_feature, axis=1)

final_result = autoML(data, X, y)
print(best_n(final_result, 'score_cv', 5))
print(best_n(final_result, 'score_auc', 5))
