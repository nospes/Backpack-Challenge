import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# Carregar os dados
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Separar features e target
X = train.drop(columns=['id', 'Price'])  # Usando 'Price' como variável alvo
y = train['Price']

# Listar colunas categóricas
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Converter colunas categóricas para o tipo 'category' para XGBoost
X_xgb = X.copy()
test_xgb = test.copy()
X_xgb[categorical_cols] = X_xgb[categorical_cols].astype('category')
test_xgb[categorical_cols] = test_xgb[categorical_cols].astype('category')

# Divisão dos dados para treino e validação
X_train_xgb, X_valid_xgb, y_train, y_valid = train_test_split(X_xgb, y, test_size=0.2, random_state=42)

# XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, enable_categorical=True)
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}
random_search_xgb = RandomizedSearchCV(xgb_model, param_distributions=param_grid_xgb, n_iter=10, 
                                       scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
random_search_xgb.fit(X_train_xgb, y_train)
best_xgb = random_search_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_valid_xgb)
mse_xgb = mean_squared_error(y_valid, y_pred_xgb)
print(f'MSE do XGBoost: {mse_xgb}')

# Fazer previsões no conjunto de teste
test_preds_xgb = best_xgb.predict(test_xgb.drop(columns=['id']))

# Criar submissão
submission_xgb = pd.DataFrame({'id': test['id'], 'Price': test_preds_xgb})
submission_xgb.to_csv("submission_xgb.csv", index=False)
print("Submissão salva como submission_xgb_basic.csv.")


###################################################################################################################

# Preparação para CatBoost
X_cat = X.copy()
test_cat = test.copy()

# Tratar valores ausentes e converter colunas categóricas para string
X_cat[categorical_cols] = X_cat[categorical_cols].astype(str).fillna("missing")
test_cat[categorical_cols] = test_cat[categorical_cols].astype(str).fillna("missing")

# Divisão dos dados para treino e validação
X_train_cat, X_valid_cat, _, _ = train_test_split(X_cat, y, test_size=0.2, random_state=42)

# CatBoost
cat_model = cb.CatBoostRegressor(loss_function='RMSE', random_state=42, cat_features=categorical_cols, verbose=0)
param_grid_cat = {
    'iterations': [500, 1000, 1500],
    'learning_rate': [0.005, 0.01, 0.03, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 10],
    'border_count': [32, 64, 128, 254],
    'bagging_temperature': [0.5, 1, 2, 5]
}
random_search_cat = RandomizedSearchCV(cat_model, param_distributions=param_grid_cat, n_iter=10, 
                                       scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
random_search_cat.fit(X_train_cat, y_train)
best_cat = random_search_cat.best_estimator_
y_pred_cat = best_cat.predict(X_valid_cat)
mse_cat = mean_squared_error(y_valid, y_pred_cat)
print(f'MSE do CatBoost: {mse_cat}')

# Fazer previsões no conjunto de teste
test_preds_cat = best_cat.predict(test_cat.drop(columns=['id']))

# Criar submissão
submission_cat = pd.DataFrame({'id': test['id'], 'Price': test_preds_cat})
submission_cat.to_csv("submission_cat_basic.csv", index=False)
print("Submissão salva como submission_cat.csv.")

print("Melhores hiperparâmetros do XGBoost:", random_search_xgb.best_params_)
print("Melhores hiperparâmetros do CatBoost:", random_search_cat.best_params_)

