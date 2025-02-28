# Backpack-Challenge
📊 Kaggle - Playground Series S5E2
Este repositório contém minhas soluções para a competição Kaggle Playground Series - Season 5, Episode 2. O objetivo da competição é prever os preços dos produtos com base em diferentes características.

📂 Estrutura do Repositório
predicition.py → Implementação de modelos XGBoost e CatBoost com RandomizedSearchCV para otimização de hiperparâmetros.
predicitonextra.py → Versão alternativa, incluindo um conjunto de dados extra para aumentar o volume de treino.
submissionXGB.csv → Previsões geradas pelo modelo XGBoost.
submissionCAT.csv → Previsões geradas pelo modelo CatBoost.
🚀 Metodologia
🔍 Pré-processamento
Carregamento dos dados train.csv e test.csv (além de training_extra.csv na versão predicitonextra.py).
Exclusão da coluna id.
Definição de variáveis categóricas (Brand, Material, Size, Laptop Compartment, Waterproof, Style, Color).
Conversão dessas variáveis para categorias no XGBoost e para strings no CatBoost.
Divisão de dados em treino e validação (80/20).
🎯 Modelagem
Utilizei dois modelos de regressão:

XGBoost 🎯

Na versão predicition.py, utilizei RandomizedSearchCV para buscar os melhores hiperparâmetros entre:
n_estimators: [100, 200, 300]
learning_rate: [0.01, 0.05, 0.1]
max_depth: [3, 5, 7]
subsample: [0.7, 0.8, 0.9]
colsample_bytree: [0.7, 0.8, 0.9]
Na versão predicitonextra.py, defini parâmetros fixos sem otimização.
CatBoost 🐱📈

Similarmente, predicition.py usou RandomizedSearchCV, explorando hiperparâmetros como:
iterations: [500, 1000, 1500]
learning_rate: [0.005, 0.01, 0.03, 0.1]
depth: [4, 6, 8, 10]
l2_leaf_reg: [1, 3, 5, 10]
A versão predicitonextra.py teve um ajuste manual de hiperparâmetros.
📈 Avaliação
Métricas utilizadas: Erro Quadrático Médio (MSE) sobre o conjunto de validação.
Importância das features visualizada para XGBoost (xgb.plot_importance).
Melhor combinação de hiperparâmetros exibida após RandomizedSearchCV.
