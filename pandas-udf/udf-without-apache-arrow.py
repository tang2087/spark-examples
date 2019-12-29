from pyspark import SparkConf
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType, DecimalType, FloatType
from pyspark.sql.functions import udf, collect_list, struct, explode
from decimal import Decimal
import random
import pandas as pd
import numpy as np

appName = "Python Example - UDF without Apache Arrow"
master = 'local'

# Create Spark session
conf = SparkConf().setMaster(master)
spark = SparkSession.builder.config(conf=conf) \
    .getOrCreate()

# Construct the data frame directly (without reading from HDFS)
cust_count = 10
txn_count = 100
data = [(i, j, Decimal(i*j*random.random()*random.choice((-1, 1)))) for j in range(txn_count)
        for i in range(cust_count)]

# Create a schema for the dataframe
schema = StructType([
    StructField('CustomerID', IntegerType(), False),
    StructField('TransactionID', IntegerType(), False),
    StructField('Amount', DecimalType(scale=2), True)
])

# Create the data frame
df = spark.createDataFrame(data, schema=schema)

# Function 1 - Scalar function - dervice a new column with value as Credit or Debit.


def calc_credit_debit_func(amount):
    return "Credit" if amount.any >= 0 else "Debit"


fn_credit_debit = udf(calc_credit_debit_func, returnType=StringType())

df = df.withColumn("CreditOrDebit", fn_credit_debit(df.Amount))
df.show()

# Function 2 - Group map function - calculate the difference from mean
attributes = [
    StructField('TransactionID', IntegerType(), False),
    StructField('Amount', DecimalType(scale=2), False),
    StructField('CreditOrDebit', StringType(), False),
    StructField('Diff', DecimalType(scale=2), False)
]
attribute_names = [a.name for a in attributes]


@udf(ArrayType(StructType(attributes)))
def fn_calc_diff_from_mean(txn):
    dict_list = [row.asDict() for row in txn]
    pdf = pd.DataFrame(dict_list)
    amount = pdf.Amount
    pdf = pdf.assign(Diff=amount-Decimal(amount.mean()))
    return [[r[attr] if attr in r else None for attr in attribute_names] for r in pdf.to_dict(orient='records')]


df_map = df.groupby("CustomerID")\
    .agg(collect_list(struct(['TransactionID', 'Amount', 'CreditOrDebit'])).alias('Transactions')) \
    .withColumn("EnrichedTransactions", fn_calc_diff_from_mean("Transactions"))
df_map.show(10)
df_map_expanded = df_map.withColumn("transactions_exploded", explode("EnrichedTransactions")) \
    .select("CustomerID", "transactions_exploded.*")
df_map_expanded.show(100)

# Function 3 - Group aggregate function - calculate mean only
@udf(DecimalType(scale=2))
def mean_udf(amount):
    return np.mean(amount)


df_agg = df.groupby("CustomerID").agg(collect_list("Amount").alias("Amounts"))\
    .withColumn("Mean", mean_udf("Amounts"))
df_agg.show()
