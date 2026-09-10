# Titanic Survival Prediction - Machine Learning

Projeto de Machine Learning desenvolvido utilizando o dataset Titanic,
com o objetivo de prever a sobrevivência dos passageiros.

## Objetivo

Construir e comparar modelos de classificação capazes de prever
`Survived`, onde:

- 0 = não sobreviveu
- 1 = sobreviveu

## Etapas do projeto

- Análise inicial dos dados
- Análise Exploratória dos Dados (EDA)
- Tratamento de valores ausentes
- Feature Engineering
- One-Hot Encoding
- Padronização
- Definição de baseline
- Comparação de modelos
- Validação cruzada
- Pipeline para prevenção de data leakage
- Hyperparameter Tuning
- Avaliação final

## Modelos avaliados

- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- SVM

## Modelo final

O Random Forest otimizado foi selecionado como modelo final,
utilizando ROC-AUC como principal critério de seleção.

## Resultados finais

| Métrica | Resultado |
|---|---:|
| Accuracy | 82.12% |
| Precision | 83.64% |
| Recall | 66.67% |
| F1-score | 74.19% |
| ROC-AUC | 84.53% |

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
