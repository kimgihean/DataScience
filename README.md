# DataScience TermProject_2022
### Auto ML for regression & classification & ensemble<br>and find best models<br><br>


## def autoML
#### This function automatically combines and executes Scaling, Modeling, and Evaluation, and returns the result in the type of a DataFrame.<br><br>


>- __Scalers__ __ <em>def Scaling</em>
>  -  StandardScaler
>  - MinMaxScaler
>  - RobustScaler
>
>- __Models__ __ <em>def Regression, def Classifier, def Ensemble</em>
>
>| Regression | Classification<br>(Parameters) | Ensemble<br>(Parameters) |
>|:---:|:---:|:---:|
>| `polynomialRegression` | `DecisionTree`<br>(criterion: {"gini", "entropy"}, max_depth: range(2,12)) | `RandomForest`<br>(n_estimators: range(2,12), max_depth: range(3,7)) |
>| `multipleRegression` | `KNeighborsClassifier`<br>(n_neighbors: range(3,12,2)) | `GradientBoosting`<br>(n_estimators: range(2,12), max_depth: range(3,7)) |
>|  | `LogisticRegression` |  |
>
>- __Evaluations__
>  - cross_val_score(score_cv)
>  - auc(score_auc)
>  - accuracy_score(score_ac)
>  - r2_score(score_r2)<br>

- __Parameters__
  - `data` : Data in the type of a DataFrame with dirty data removed, The target data is binary classification.
  - `X` : Training dataset
  - `y` : Target data

- __return__
  - `final_result` : DataFrame with columns 'Model', 'Scaler', 'Parameter_dict', 'score_cv', 'score_auc', 'score_ac', 'score_r2', 'runtime', 'model_'.

<br>


## def best_n
#### Find best scores<br>

- __Parameters__
  - `df` : Final result(`final_result`) after all process
  - `ft_name` : What kinds of scoring (cv, auc, ac, r2)
  - `n` : The highest rank in order n (1 ~ n)


- __return__ : DataFrame after descending sorting




