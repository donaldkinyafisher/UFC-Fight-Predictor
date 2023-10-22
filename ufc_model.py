import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from ufc_data import X_train, X_test, y_train, y_test

#Gradient Boost Model using hyper parameter obtained through Optuna trial tuning
gradientBoostParams = {'lambda': 0.02346830703895387, 'alpha': 9.956365178312454, 'tree_method': 'gpu_hist', 'objective': 'binary:logistic', 'verbosity': 0, 'n_jobs': -1, 'learning_rate': 0.030042259864304366, 'min_child_weight': 19, 'max_depth': 12, 'max_delta_step': 8, 'subsample': 0.21684497186514273, 'colsample_bytree': 0.7003826597151147, 'gamma': 0.02156806207076522, 'n_estimators': 147, 'eta': 0.15337337361538475}
gradientBoostModel = xgb.XGBClassifier(**gradientBoostParams)

#Set model to use
model = gradientBoostModel

#Train model
model.fit(X_train, y_train)

#Get model accuracy
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'test_accuracy:{accuracy}')

#Save model
#model.save_model("UFC_MODEL_XGBOOST_V1.json")