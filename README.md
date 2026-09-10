# Titanic - Previsão de Sobrevivência com Machine Learning

Projeto de Machine Learning que constrói e avalia modelos de classificação capazes de prever se um passageiro do Titanic sobreviveu ou não, utilizando a base `train.csv` da competição [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) do Kaggle.

## 🎯 Objetivo

Prever a variável alvo `Survived`:

- `0` → Não sobreviveu
- `1` → Sobreviveu

O projeto percorre todo o ciclo de um problema de classificação supervisionada: análise exploratória, tratamento de valores ausentes, feature engineering, encoding, padronização, comparação de modelos, validação cruzada, otimização de hiperparâmetros e avaliação final em dados não vistos.

## 📊 Sobre os dados

| Coluna | Significado |
|---|---|
| `PassengerId` | Identificador único do passageiro |
| `Survived` | Variável-alvo (`0` = Não, `1` = Sim) |
| `Pclass` | Classe da passagem (1ª, 2ª ou 3ª) |
| `Name` | Nome do passageiro |
| `Sex` | Sexo do passageiro |
| `Age` | Idade em anos |
| `SibSp` | Nº de irmãos/cônjuges a bordo |
| `Parch` | Nº de pais/filhos a bordo |
| `Ticket` | Código do bilhete |
| `Fare` | Tarifa paga |
| `Cabin` | Cabine |
| `Embarked` | Porto de embarque |

## 🔍 Principais descobertas da EDA

- **Sexo** foi a variável com maior poder preditivo aparente: mulheres tiveram taxa de sobrevivência muito superior à dos homens.
- **Classe (`Pclass`)** também é altamente relevante: passageiros de 1ª classe sobreviveram proporcionalmente mais. Correlação com `Survived` ≈ **-0,34**.
- **Fare** teve correlação positiva de ≈ **0,26** com `Survived` (tarifas mais altas → maior chance de sobrevivência), e correlação negativa forte com `Pclass` (≈ **-0,55**), como esperado.
- `SibSp` e `Parch` apresentaram correlação positiva entre si (≈ **0,41**), sugerindo que famílias tendiam a viajar juntas.
- A combinação de sexo e classe mostrou efeito de interação sobre a sobrevivência (ex: mulheres de 1ª classe tiveram taxa de sobrevivência muito mais alta que homens de 3ª classe).

## 🛠️ Pipeline de pré-processamento

1. **Extração de `Title`** a partir de `Name` (Mr, Mrs, Miss, Master, etc.), usado apenas como apoio para imputar idade — depois removido do conjunto final.
2. **Imputação de `Age`** em cascata, aprendida somente com o treino:
   - mediana por `Title` + `Pclass`;
   - fallback: mediana por `Title`;
   - fallback final: mediana geral.
3. **Imputação de `Embarked`** com a moda calculada no treino.
4. **One-Hot Encoding** para `Sex`, `Embarked` e `Pclass`.
5. **Padronização (`StandardScaler`)** das variáveis numéricas (`Age`, `SibSp`, `Parch`, `Fare`).
6. Remoção de `PassengerId`, `Name`, `Ticket` e `Cabin` (não utilizadas nesta versão do modelo).

Para eliminar qualquer risco de **data leakage**, todo esse fluxo foi reconstruído com `Pipeline` + `ColumnTransformer` do scikit-learn, incluindo um **transformer customizado (`AgeImputer`)** que aprende as medianas apenas dentro de cada fold de validação cruzada.

## 🤖 Modelagem

**Baseline:** classificador ingênuo prevendo sempre a classe majoritária, usado como referência mínima de desempenho.

**Modelos comparados** (parâmetros padrão, depois otimizados):

- Regressão Logística
- KNN
- Árvore de Decisão
- Random Forest
- SVM

**Validação:** `StratifiedKFold` com 5 folds × 5 seeds diferentes (25 avaliações por modelo), para reduzir a variância de uma única divisão treino/teste.

**Otimização de hiperparâmetros:** `GridSearchCV` (métrica: AUC-ROC) para os dois modelos finalistas — Random Forest e SVM.

| Modelo | Melhores parâmetros | AUC-ROC (CV) |
|---|---|---|
| Random Forest | `n_estimators=100`, `max_depth=None`, `min_samples_split=10`, `min_samples_leaf=4` | 0,8821 |
| SVM | `C=10`, `kernel=rbf`, `gamma=auto` | 0,8561 |

O **Random Forest otimizado** foi escolhido como modelo final por apresentar a maior AUC-ROC média e maior estabilidade (menor desvio padrão) entre as validações.

## 📈 Resultados finais (conjunto de teste)

| Métrica | Valor |
|---|---|
| Accuracy | 82,12% |
| Precision | 83,64% |
| Recall | 66,67% |
| F1-score | 74,19% |
| AUC-ROC | 84,53% |

**Matriz de confusão:**

|  | Previsto: Não sobreviveu | Previsto: Sobreviveu |
|---|---|---|
| **Real: Não sobreviveu** | 101 (VN) | 9 (FP) |
| **Real: Sobreviveu** | 23 (FN) | 46 (VP) |

Os resultados no teste ficaram próximos das médias obtidas na validação cruzada, indicando boa capacidade de generalização. A Precision alta mostra que o modelo é confiável ao apontar sobreviventes, enquanto o Recall de ~67% indica que ainda deixa de identificar uma parte real dos sobreviventes — um trade-off possível de ajustar via threshold de classificação.

## 🚀 Possíveis melhorias futuras

- Usar `Title` diretamente como variável preditora, não só como apoio para imputação.
- Extrair informação da cabine (`Cabin`) e criar `FamilySize` a partir de `SibSp` + `Parch`.
- Testar algoritmos de boosting (XGBoost, LightGBM, Gradient Boosting).
- Ampliar a busca de hiperparâmetros (`RandomizedSearchCV` / Bayesian Optimization).
- Analisar diferentes thresholds de classificação para equilibrar Precision e Recall conforme o objetivo do problema.

## 🧰 Tecnologias utilizadas

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook / Google Colab

## ▶️ Como executar

1. Clone este repositório:
   ```bash
   git clone <url-do-repositorio>
   cd <nome-do-repositorio>
   ```
2. Abra o notebook `titanic_machine_learning.ipynb` no [Google Colab](https://colab.research.google.com/) ou localmente com Jupyter.
3. Faça o upload do arquivo `train.csv` (disponível na página da [competição no Kaggle](https://www.kaggle.com/competitions/titanic/data)) quando solicitado na primeira célula.
4. Execute as células em ordem.

## 📁 Estrutura do notebook

1. Carregamento da base de dados
2. Análise inicial dos dados
3. Análise Exploratória (EDA)
4. Extração de títulos a partir dos nomes
5. Separação entre variáveis independentes e alvo
6. Separação treino/teste
7. Tratamento de `Age`
8. Tratamento de `Embarked`
9. Remoção da variável auxiliar `Title`
10. Codificação das variáveis categóricas
11. Padronização das variáveis numéricas
12. Definição do baseline
13. Comparação inicial de modelos
14. Avaliação inicial dos modelos
15. Validação cruzada com múltiplas seeds
16. Reconstrução do fluxo de pré-processamento com Pipelines
17. Construção dos Pipelines dos modelos finalistas
18. Hyperparameter Tuning - Random Forest
19. Hyperparameter Tuning - SVM
20. Validação cruzada após o tuning
21. Seleção do modelo final
22. Avaliação final no conjunto de teste
23. Análise dos resultados finais
24. Conclusão

## ✍️ Autor

Gabriel
