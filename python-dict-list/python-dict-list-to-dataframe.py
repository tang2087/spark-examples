from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType,\
    IntegerType, FloatType
from decimal import Decimal

appName = "Python Example - Python Dictionary List to Spark Data Frame"
master = "local"

# Create Spark session
spark = SparkSession.builder \
    .appName(appName) \
    .master(master) \
    .getOrCreate()

# List
data = [{"Category": 'Category A', 'ItemID': 1, 'Amount': 12.40},
        {"Category": 'Category B', 'ItemID': 2, 'Amount': 30.10},
        {"Category": 'Category C', 'ItemID': 3, 'Amount': 100.01},
        {"Category": 'Category A', 'ItemID': 4, 'Amount': 110.01},
        {"Category": 'Category B', 'ItemID': 5, 'Amount': 70.85}
        ]


def infer_schema():
    # Create data frame
    df = spark.createDataFrame(data)
    print(df.schema)
    df.show()


def explicit_schema():
    # Create a schema for the dataframe
    schema = StructType([
        StructField('Category', StringType(), False),
        StructField('ItemID', IntegerType(), False),
        StructField('Amount', FloatType(), True)
    ])

    # Create data frame
    df = spark.createDataFrame(data, schema)
    print(df.schema)
    df.show()


if __name__ == "__main__":
    infer_schema()
    explicit_schema()
