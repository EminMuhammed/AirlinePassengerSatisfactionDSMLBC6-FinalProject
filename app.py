# Dataset: https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction

from project.preprocessing import get_data, asel_mebaysan_preprocess
from project.model import random_forest_classifier, xgbm_classifier, random_forest_classifier_basemodel, \
    xgbm_classifier_basemodel
from project.prediction import predict, get_test_data
from helpers.model_evaluation import plot_importance

#################################
#### * Train ####
#################################

df = get_data()
df = asel_mebaysan_preprocess(df, 'group')
X = df.drop('satisfaction_satisfied', axis=1)
y = df['satisfaction_satisfied']

result_dict_rf = random_forest_classifier(X, y)
result_dict_xgbm = xgbm_classifier(X, y)

# PARAMETREYİ KENDİMİZ VERDİK
parameters_rf = {"max_features": 7,
                 "n_estimators": 500,
                 "random_state": 34}

parameters_xgb = {"booster": "gbtree",
                  "learning_rate": 0.01,
                  "max_depth": 12,
                  "n_estimators": 1000}

result_dict_rf_base = random_forest_classifier_basemodel(parameters_rf, X, y)
result_dict_xgbm_base = xgbm_classifier_basemodel(parameters_xgb, X, y)

# RF GRİDSEARCH RESULT
# {'model': RandomForestClassifier(max_features=7, n_estimators=500, random_state=34),
#  'train_accuracy': 0.9652467792071682,
#  'train_f1': 0.9593570918291565,
#  'train_roc_auc': 0.9947164095559189}

# XGBOOST GRİDSEARCH RESULT
# {'model': XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#                colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
#                importance_type='gain', interaction_constraints='',
#                learning_rate=0.01, max_delta_step=0, max_depth=12,
#                min_child_weight=1, missing=nan, monotone_constraints='()',
#                n_estimators=1000, n_jobs=4, num_parallel_tree=1, random_state=17,
#                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#                tree_method='exact', validate_parameters=1, verbosity=None),
#  'train_accuracy': 0.9673544906866453,
#  'train_f1': 0.9617683619923908,
#  'train_roc_auc': 0.9959605868892137}
#################################
#### * Test ####
#################################

test_df, test_X, test_y = get_test_data(na='group')

test_dict_rf = predict(test_X, test_y, result_dict_rf['model'])
test_dict_xgbm = predict(test_X, test_y, result_dict_xgbm['model'])

test_dict_rf_base = predict(test_X, test_y, result_dict_rf_base['model'])
test_dict_xgbm_base = predict(test_X, test_y, result_dict_xgbm_base['model'])

#################################
#### * Plot Importance for Random Forest ####
#################################
plot_importance(test_dict_xgbm_base['model'], test_X, num=15)

#################################
#### * Plot Importance for XGBM ####
#################################
plot_importance(test_dict_xgbm['model'], test_X, num=15)
