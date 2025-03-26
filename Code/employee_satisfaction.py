#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 10:11:01 2025

@author: beto
"""
# For data manipulation
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

# For data modeling
from xgboost import XGBClassifier 
from xgboost import XGBRegressor 
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report 
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

# For saving models
import pickle



# df0 = pd.read_csv("HR_capstone_dataset.csv")
# df0.head()

####################################################################################################################################################################################
################################# Cleaning Protocols ################################################################################################
####################################################################################################################################################################################

df0.info()
df0.describe()
df0.columns
df0 = df0.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})
df0.isna().sum()
df0.duplicated().sum()
df0[df0.duplicated()].head()


df1 = df0.drop_duplicates(keep='first')
df1.head()



# Percentile analysis 
percentile25 = df1['tenure'].quantile(0.25)
percentile75 = df1['tenure'].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]

print("Number of rows in the data containing outliers in `tenure`:",len(outliers))




####################################################################################################################################################################################
################################# EDA ################################################################################################
####################################################################################################################################################################################

print(df1['left'].value_counts())
print()
print(df1['left'].value_counts(normalize=True))


# Create plotting
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing `average_monthly_hours` distributions for␣ ,`number_project`, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='average_monthly_hours', y='number_project', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly hours by number of projects', fontsize='14')

# Create histogram showing distribution of `number_project`, comparing␣ ,employees who stayed versus those who left
tenure_stay = df1[df1['left']==0]['number_project']
tenure_left = df1[df1['left']==1]['number_project']
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge',shrink=2, ax=ax[1])
ax[1].set_title('Number of projects histogram', fontsize='14')

plt.show()


df1[df1['number_project']==7]['left'].value_counts()

plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--') 
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed']) 
plt.title('Monthly hours by last evaluation score', fontsize='14');


# Bar plots and boxplots 
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Boxplot showing distributions of `satisfaction_level` by tenure, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by tenure', fontsize='14')

# Histogram showing distribution of `tenure`, comparing employees who, stayed versus those who left
tenure_stay = df1[df1['left']==0]['tenure']
tenure_left = df1[df1['left']==1]['tenure']
sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5,ax=ax[1])
ax[1].set_title('Tenure histogram', fontsize='14')
plt.show();



# Mean - median of satisfaction level
df1.groupby(['left'])['satisfaction_level'].agg([np.mean,np.median])

# Scatterplot of `average_monthly_hours` versus `last_evaluation`
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='last_evaluation', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--') 
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed']) 
plt.title('Monthly hours by last evaluation score', fontsize='14');


# Correlation heatmap
plt.figure(figsize=(16, 9))
heatmap = sns.heatmap(df0.corr(), vmin=-1, vmax=1, annot=True, cmap=sns,color_palette("vlag", as_cmap=True)
heatmap.set_title('Correlation Heatmap', 
                  fontdict={'fontsize':14}, 
                  pad=12)


####################################################################################################################################################################################
################################# LOGSITIC REGRESSION ################################################################################################
####################################################################################################################################################################################


df_logreg = df_enc[(df_enc['tenure'] >= lower_limit) & (df_enc['tenure'] <=␣ ,→upper_limit)]
# df_logreg.head()
y = df_logreg['left']
X = df_logreg.drop('left', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
y_pred = log_clf.predict(X_test)

log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm,
                                  display_labels=log_clf.classes_)
log_disp.plot(values_format='')
plt.show()

# Value counts for model balance
df_logreg['left'].value_counts(normalize=True)
target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))


# Tree based model 
y = df_enc['left']
X = df_enc.drop('left', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
tree = DecisionTreeClassifier(random_state=0)

cv_params = {'max_depth':[4, 6, 8, None], 'min_samples_leaf': [2, 5, 1], 'min_samples_split': [2, 4, 6]}

scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')

##############################
# Only for Jupyter
%time%
tree1.fit(X_train, y_train)
###############################


tree1.best_params_
tree1.best_score_


def make_results(model_name:str, model_object, metric:str): 
'''
Arguments:
model_name (string): what you want the model to be called in the output, table
model_object: a fit GridSearchCV object
metric (string): precision, recall, f1, accuracy, or auc
    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.

'''
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
}
    cv_results = pd.DataFrame(model_object.cv_results_)
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]]. ,→idxmax(), :]
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
    table = pd.DataFrame()
    
    table = pd.DataFrame({'model': [model_name],
                      'precision': [precision],
    )}
    return table

# Instantiate model 
rf = RandomForestClassifier(random_state=0)
cv_params = {'max_depth': [3,5, None], 'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
}
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')

# Path to save the model. Edit depending your working directory
#path = '~ home/jovyan/work/'


rf1.best_score_
rf1.best_params_

rf1_cv_results = make_results('random forest cv', rf1, 'auc')
print(tree1_cv_results)
print(rf1_cv_results)



def get_scores(model_name:str, model, X_test_data, y_test_data): 
'''
    Generate a table of test scores.
    In:
        model_name (string): How you want your model to be named in the output,table
        model:
            X_test_data:
            y_test_data:
            A fit GridSearchCV object
            numpy array of X_test data
            numpy array of y_test data
    Out: pandas df of precision, recall, f1, accuracy, and AUC scores for your␣ ,→model
'''
   preds = model.best_estimator_.predict(X_test_data)
   auc = roc_auc_score(y_test_data, preds)
   accuracy = accuracy_score(y_test_data, preds)
   precision = precision_score(y_test_data, preds)
   recall = recall_score(y_test_data, preds)
   f1 = f1_score(y_test_data, preds)
   
   table = pd.DataFrame({'model': [model_name],
                         'precision': [precision],
                         'recall': [recall],
                         'f1': [f1],
                         'accuracy': [accuracy],
                         'AUC': [auc]
     })

    return table


# Get predictions on test data
rf1_test_scores = get_scores('random forest1 test', rf1, X_test, y_test)
rf1_test_scores


# Feature Engineering for robustness
df2['overworked'] = df2['average_monthly_hours']
print('Max hours:', df2['overworked'].max())
print('Min hours:', df2['overworked'].min())

df2['overworked'] = (df2['overworked'] > 175).astype(int)
df2 = df2.drop('average_monthly_hours', axis=1)

# Model (re)implementation
y = df2['left']
X = df2.drop('left', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,stratify=y, random_state=0)

tree = DecisionTreeClassifier(random_state=0)
cv_params = {'max_depth':[4, 6, 8, None], 
             'min_samples_leaf': [2, 5, 1], 
             'min_samples_split': [2, 4, 6]}

scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

tree2 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')

tree2.best_params_
tree2.best_score_


tree2_cv_results = make_results('decision tree2 cv', tree2, 'auc')
print(tree1_cv_results)
print(tree2_cv_results)


####################################################################################################################################################################################
################################# RANDOM FOREST ################################################################################################
####################################################################################################################################################################################

rf = RandomForestClassifier(random_state=0)
# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None], 'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }

scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

rf2 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')

rf2.best_params_
rf2.best_score_

rf2_cv_results = make_results('random forest2 cv', rf2, 'auc')
print(tree2_cv_results)
print(rf2_cv_results)


rf2_test_scores = get_scores('random forest2 test', rf2, X_test, y_test)
#rf2_test_scores

preds = rf2.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rf2.classes_)

# Plot confusion matrix

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf2.classes_)
# disp.plot(values_format='');
# plt.figure(figsize=(85,20))
# plot_tree(tree2.best_estimator_, max_depth=6, fontsize=14, feature_names=X.columns,
# class_names={0:'stayed', 1:'left'}, filled=True);
# plt.show()


# Feature importance through Gini analysis

# tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, columns=X.columns)
tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_,
                                 columns=['gini_importance'],
                                 index=X.columns)
tree2_importances = tree2_importances.sort_values(by='gini_importance',ascending=False)
# Only extract the features with importances > 0
tree2_importances = tree2_importances[tree2_importances['gini_importance'] != 0]
tree2_importances

sns.barplot(data=tree2_importances, x="gini_importance", y=tree2_importances. index, orient='h')
plt.title("Decision Tree: Feature Importances for Employee Leaving",fontsize=12)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.show()



# Feature importance for Random forest

# Get feature importances
feat_impt = rf2.best_estimator_.feature_importances_ # Get indices of top 10 features
ind = np.argpartition(rf2.best_estimator_.feature_importances_, -10)[-10:] # Get column labels of top 10 features
feat = X.columns[ind]
# Filter `feat_impt` to consist of top 10 feature importances
feat_impt = feat_impt[ind]
y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
y_sort_df = y_df.sort_values("Importance")
fig = plt.figure()

ax1 = fig.add_subplot(111)
y_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")
ax1.set_title("Random Forest: Feature Importances for Employee Leaving",fontsize=12)
ax1.set_ylabel("Feature")
ax1.set_xlabel("Importance")
plt.show()

