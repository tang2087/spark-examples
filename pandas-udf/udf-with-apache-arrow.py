from pyspark import SparkConf
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType, DecimalType, FloatType
from pyspark.sql.functions import udf, collect_list, struct, explode, pandas_udf, PandasUDFType, col
from decimal import Decimal
import random
import pandas as pd
import numpy as np

appName = "Python Example - UDF with Apache Arrow (Pandas UDF)"
master = 'local'

# Create Spark session
conf = SparkConf().setMaster(master)
spark = SparkSession.builder.config(conf=conf) \
    .getOrCreate()

# Enable Arrow optimization and fallback if there is no Arrow installed
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.fallback.enabled", "true")

# Construct the data frame directly (without reading from HDFS)
cust_count = 10
txn_count = 100
data = [(i, j, i * j * random.random() * random.choice((-1, 1)))
        for j in range(txn_count) for i in range(cust_count)]

# Create a schema for the dataframe
schema = StructType([
    StructField('CustomerID', IntegerType(), False),
    StructField('TransactionID', IntegerType(), False),
    StructField('Amount', FloatType(), True)
])

# Create the data frame
df = spark.createDataFrame(data, schema=schema)

# Function 1 - Scalar function - dervice a new column with value as Credit or Debit.


def calc_credit_debit_func(amount):
    return pd.Series(["Credit" if a >= 0 else "Debit" for a in amount])


fn_credit_debit = pandas_udf(calc_credit_debit_func, returnType=StringType())

df = df.withColumn("CreditOrDebit", fn_credit_debit(df.Amount))
df.show()

# Function 2 - Group map function - calculate the difference from mean
attributes = [
    StructField('CustomerID', IntegerType(), False),
    StructField('TransactionID', IntegerType(), False),
    StructField('Amount', FloatType(), False),
    StructField('CreditOrDebit', StringType(), False),
    StructField('Diff', FloatType(), False)
]
attribute_names = [a.name for a in attributes]


@pandas_udf(StructType(attributes), PandasUDFType.GROUPED_MAP)
def fn_calc_diff_from_mean(txn):
    pdf = txn
    amount = pdf.Amount
    pdf = pdf.assign(Diff=amount - amount.mean())
    return pdf


df_map = df.groupby("CustomerID").apply(fn_calc_diff_from_mean)
df_map.show(100)


# Function 3 - Group aggregate function - calculate mean only
@pandas_udf(FloatType(), PandasUDFType.GROUPED_AGG)
def mean_udf(amount):
    return np.mean(amount)


df_agg = df.groupby("CustomerID").agg(mean_udf(df['Amount']).alias("Mean"))
df_agg.show()

# Function 4 - Group aggregate function - Windowing function

w = Window \
    .partitionBy('CustomerID') \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
df.withColumn('Mean', mean_udf(df['Amount']).over(w)).show()

