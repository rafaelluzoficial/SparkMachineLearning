# Spark-Machine-Learning
Code repository with Spark Machine Learning commands

## Code repository for studying Spark Framework

# Preparação do ambiente linux para execução do SPARK

## Atualizando o ambiente
sudo apt update
sudo apt -y upgrade

## Instalando JAVA
sudo apt install curl mlocate default-jdk -y

## Baixando SPARK
wget [link de download do SPARK]

## Descomprimindo pacote TAR do SPARK
tar xvf [nome do arquivo .tgz]

## Movendo SPARK para pasta OPT
sudo mv [nome da pasta]/ /opt/spark

## Definindo variáveis de ambiente com GEDIT
sudo gedit ~/.bashrc

[inserir no arquivo bashrc]

export SPARK_HOME=/opt/spark

export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

[salvar arquivo bashrc]

source ~/.bashrc

## Iniciando nó MASTER do SPARK no modo standalone
start-master.sh
Teste no navegador digitando http://localhost:8080/

## Iniciando workprocess do SPARK
/opt/spark/sbin/start-slave.sh spark://localhost:7077

## Iniciando SPARK shell com linguagem SCALA
spark-shell

## Iniciando SPARK com linguagem PYTHON (PYSPARK)
pyspark

## Instalando bibliotecas adicionais Python

## Instalando gerenciador de pacotes PIP
sudo apt install python3-pip

## Instalando pacotes adicionais Numpy e Pandas
pip install numpy
pip install pandas
