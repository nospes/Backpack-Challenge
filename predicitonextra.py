import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Carregar os dados
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
training_extra = pd.read_csv("training_extra.csv")  # Adicionando training_extra

# Unindo training_extra ao conjunto de treino
train = pd.concat([train, training_extra], ignore_index=True)

# Separar features e target
X = train.drop(columns=['id', 'Price'])  # Usando 'Price' como variável alvo
y = train['Price']

# Listar colunas categóricas
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Converter colunas categóricas para o tipo 'category'
X[categorical_cols] = X[categorical_cols].astype('category')
test[categorical_cols] = test[categorical_cols].astype('category')

# Divisão dos dados para treino e validação
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir modelo XGBoost com os parâmetros fixos
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    random_state=42, 
    enable_categorical=True,
    n_estimators=400,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.9
)

# Treinar o modelo
xgb_model.fit(X_train, y_train)

# Avaliação no conjunto de validação
y_pred = xgb_model.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred)
print(f'MSE no conjunto de validação (XGBoost): {mse}')

# Fazer previsões no conjunto de teste
test_preds = xgb_model.predict(test.drop(columns=['id']))

xgb.plot_importance(xgb_model)
plt.show()

# Criar submissão
submission = pd.DataFrame({
    'id': test['id'], 
    'Price': test_preds  
})
submission.to_csv("submissionXGB.csv", index=False)
print("Submissão salva como submissionXGB.csv no formato correto.")

# Preparação para CatBoost
X_cat = X.copy()
test_cat = test.copy()

# Tratar valores ausentes e converter colunas categóricas para string
X_cat[categorical_cols] = X_cat[categorical_cols].astype(str).fillna("missing")
test_cat[categorical_cols] = test_cat[categorical_cols].astype(str).fillna("missing")

# Divisão dos dados para treino e validação
X_train_cat, X_valid_cat, _, _ = train_test_split(X_cat, y, test_size=0.2, random_state=42)

# CatBoost com parâmetros fixos
cat_model = cb.CatBoostRegressor(
    loss_function='RMSE', 
    random_state=42, 
    cat_features=categorical_cols, 
    verbose=0,
    iterations=1500,
    learning_rate=0.03,
    depth=4,
    l2_leaf_reg=10,
    border_count=254,
    bagging_temperature=1
)

# Treinar o modelo CatBoost
cat_model.fit(X_train_cat, y_train)

# Avaliação no conjunto de validação
y_pred_cat = cat_model.predict(X_valid_cat)
mse_cat = mean_squared_error(y_valid, y_pred_cat)
print(f'MSE do CatBoost: {mse_cat}')

# Fazer previsões no conjunto de teste
test_preds_cat = cat_model.predict(test_cat.drop(columns=['id']))

# Criar submissão
submission_cat = pd.DataFrame({'id': test['id'], 'Price': test_preds_cat})
submission_cat.to_csv("submissionCAT.csv", index=False)
print("Submissão salva como submissionCAT.csv.")
