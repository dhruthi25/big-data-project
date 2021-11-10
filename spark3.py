import pyspark
lines=pyspark.sql.streaming.DataStreamReader.json('http://localhost:6100/',pyspark.sql.types.StructType)
print(lines.isStreaming)
print(lines.schema)