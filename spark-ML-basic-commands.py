# Databricks notebook source
# -----------------------------------#
# 1o - Ingestão de dados             #
# 2o - Criação do modelo             #
# 3o - Treinamento do modelo         #
# 4o - Teste do modelo               #
# 5o - Medir a performance do modelo #
# -----------------------------------#

# Casse = o que eu quero prever ou classificar, também é um atributo/classe
# Dimensão ou atributo = características ou colunas
# Instãncia =  observação ou linha
# Relação = conjunto de dados

# Classificação binária = classificação de sim ou não em uma única classe/coluna
# Classificação multiclasse = mais de uma classificação em uma mesma classe/coluna
# Classificação multilabel = classificação por meio de mais de uma classe/coluna

# Variáveis independentes devem estar em uma única coluna
# Variáveis dependentes = a qual se quer avaliar
# One Hot Encoding = todos os dados devem ser convertidos em números

# COMMAND ----------

# -------------------------------------------------------------------------#
# Usando modelos de regressão Linear Regression e Random Forest Regression #
# Dado um projeto de um carro, queremos prever qual a potência HP do mesmo #
# -------------------------------------------------------------------------#

# COMMAND ----------

# bibliotecas para gerar modelos de regressão
from pyspark.ml.regression import LinearRegression, RandomForestRegressor

# biblioteca para avaliar a performance do modelo de regressão
from pyspark.ml.evaluation import RegressionEvaluator

# biblioteca para transformar dados categoricos em númericos e agrupa variáveis independentes em uma única coluna
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# Ingestão de dados
carros_tmp = spark.read.csv('/FileStore/tables/Carros.csv', inferSchema=True, header=True, sep=";")

# Criando um subconjunto de dados
carros = carros_tmp.select("Consumo", "Cilindros", "Cilindradas", "HP")
carros.show(5)

# COMMAND ----------

# Criar um objeto que agrupa variáveis independentes em uma úica coluna
vector = VectorAssembler(inputCols=[("Consumo"),("Cilindros"),("Cilindradas")], outputCol="VectorColumn")

# Aplica os dados do df carros ao objeto de transformação vector
carros = vector.transform(carros)
carros.show(5)

# COMMAND ----------

# Dividir a massa de dados para treino 70% e para teste 30%
carros_treino, carros_teste = carros.randomSplit([0.7,0.3])
print(carros_treino.count())
print(carros_teste.count())

# COMMAND ----------

# Cria objeto do tipo Linear Regression usando as colunas de variáveis independentes e de variável dependente
obj_lr = LinearRegression(featuresCol="VectorColumn", labelCol="HP")
# Cria o modelo aplicando o objeto LR aos dados de treino
modelo_lr = obj_lr.fit(carros_treino)

# Cria objeto do tipo Random Forest Regression usando as colunas de variáveis independentes e de variável dependente
obj_rf = RandomForestRegressor(featuresCol="VectorColumn", labelCol="HP")
# Cria o modelo aplicando o objeto RFRegression aos dados de treino
modelo_rf = obj_rf.fit(carros_treino)

# COMMAND ----------

# Realizando a previsão usando modelos criados com base nos dados de teste
previsao_lr = modelo_lr.transform(carros_teste)
previsao_lr.show()

previsao_rf = modelo_rf.transform(carros_teste)
previsao_rf.show()

# COMMAND ----------

# Cria objeto para avaliação da previsão dos modelos utilizados com dados de teste
obj_avaliar = RegressionEvaluator(predictionCol="prediction", labelCol="HP", metricName="rmse")

# Avalia as previsões feitas com os modelos LR e RF com base nos dados de teste
# Na métrica RMSE quanto menor o valor, mais precisa é a previsão
avaliacao_lr = obj_avaliar.evaluate(previsao_lr)
print(avaliacao_lr)

avaliacao_rf = obj_avaliar.evaluate(previsao_rf)
print(avaliacao_rf)

# COMMAND ----------

# --------------------------------------------------------------------------#
# Usando modelos de classificação binária, classe com valores True ou False #
# Dado um determinado cliente, prever se ele vai deixar de comprar ou não   #
# --------------------------------------------------------------------------#

# COMMAND ----------

# biblioteca para criar um modelos de classificação
from pyspark.ml.feature import RFormula

# biblioteca para cirar algoritmo de classificação árvore de decisão
from pyspark.ml.classification import DecisionTreeClassifier

# biblioteca para avaliar a performance do modelo de classificação binária
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

# Ingestão de dados
churn = spark.read.csv('/FileStore/tables/Churn.csv', inferSchema=True, header=True, sep=";")
churn.show()

# COMMAND ----------

# Criando objeto R Fórmula para criar modelo de classificação
formula = RFormula(formula="Exited ~ .", featuresCol="featuresCol", labelCol="labelCol", handleInvalid="skip")

# Realizando One Hot Encoding dos dados usando o modelo criado
churn_transformado = formula.fit(churn).transform(churn).select("featuresCol", "labelCol")
churn_transformado.show(truncate=False)

# COMMAND ----------

# Dividindo os dados em treino e teste
churnTreino, churnTeste = churn_transformado.randomSplit([0.7,0.3])
print(churnTreino.count())
print(churnTeste.count())

# COMMAND ----------

# Criando objeto para geração do modelo de classificação
objeto_dt = DecisionTreeClassifier(labelCol="labelCol", featuresCol="featuresCol")

# criando modelo de classificação
modelo_dt = objeto_dt.fit(churnTreino)

# COMMAND ----------

# Realizando previsões com o modelo criado
previsao_dt = modelo_dt.transform(churnTeste)
previsao_dt.show(truncate=False)

# COMMAND ----------

# Avaliar a performance do modelo usando métrica de área sobre a curva, quanto mais próximo de 1 melhor
obj_avaliar_dt = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="labelCol", metricName="areaUnderROC")
avaliar_dt = obj_avaliar_dt.evaluate(previsao_dt)
print(avaliar_dt)
