# Databricks notebook source
# MAGIC %md
# MAGIC # Haritha Thipparapu 
# MAGIC ### Week 07 - Spark Application

# COMMAND ----------

df1 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/haritha.thipparapu@slu.edu/OP_REMOVED_DELETED_PGYR2023_P01302025_01212025.csv")
df2 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/haritha.thipparapu@slu.edu/OP_DTL_OWNRSHP_PGYR2023_P01302025_01212025.csv")

# Additional files uploaded
# dbfs:/FileStore/shared_uploads/haritha.thipparapu@slu.edu/OP_PGYR2023_README_P01302025.txt


# COMMAND ----------

df_txt = spark.read.text("dbfs:/FileStore/shared_uploads/haritha.thipparapu@slu.edu/OP_PGYR2023_README_P01302025.txt")


# COMMAND ----------

df3 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/haritha.thipparapu@slu.edu/OP_DTL_RSRCH_PGYR2023_P01302025_01212025.csv")

# COMMAND ----------

df4 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/haritha.thipparapu@slu.edu/OP_CVRD_RCPNT_PRFL_SPLMTL_P01302025_01212025.csv")

# Additional files uploaded
# dbfs:/FileStore/shared_uploads/haritha.thipparapu@slu.edu/OP_CVRD_RCPNT_PRFL_SPLMTL_README_P01302025.txt

# COMMAND ----------

df_txt1 = spark.read.text("dbfs:/FileStore/shared_uploads/haritha.thipparapu@slu.edu/OP_CVRD_RCPNT_PRFL_SPLMTL_README_P01302025.txt")


# COMMAND ----------

dflarge = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/haritha.thipparapu@slu.edu/OP_DTL_GNRL_PGYR2023_P01302025_01212025copy.csv")

# COMMAND ----------

from pyspark.sql import functions as F

# 1. Nature of Payments with reimbursement amounts greater than $1,000 ordered by count
dflarge.filter(F.col("Total_Amount_of_Payment_USDollars") > 1000) \
    .groupBy("Nature_of_Payment_or_Transfer_of_Value") \
    .count() \
    .orderBy(F.desc("count")) \
    .show(10, truncate=False)

# 2. Top ten Nature of Payments by count
dflarge.groupBy("Nature_of_Payment_or_Transfer_of_Value") \
    .count() \
    .orderBy(F.desc("count")) \
    .show(10, truncate=False)

# 3. Top ten Nature of Payments by total amount
dflarge.groupBy("Nature_of_Payment_or_Transfer_of_Value") \
    .agg(F.sum("Total_Amount_of_Payment_USDollars").alias("Total_Payment")) \
    .orderBy(F.desc("Total_Payment")) \
    .show(10, truncate=False)

# 4. Top ten physician specialties by total amount
dflarge.groupBy("Covered_Recipient_Primary_Type_1") \
    .agg(F.sum("Total_Amount_of_Payment_USDollars").alias("Total_Payment")) \
    .orderBy(F.desc("Total_Payment")) \
    .show(10, truncate=False)

# 5. Top ten physicians by total amount
dflarge.groupBy("Covered_Recipient_First_Name", "Covered_Recipient_Last_Name") \
    .agg(F.sum("Total_Amount_of_Payment_USDollars").alias("Total_Payment")) \
    .orderBy(F.desc("Total_Payment")) \
    .show(10, truncate=False)


# COMMAND ----------

from pyspark.sql import functions as F

# 1. Nature of Payments with reimbursement amounts greater than $1,000 ordered by count
dflarge.filter(F.col("Total_Amount_of_Payment_USDollars") > 1000) \
    .groupBy("Nature_of_Payment_or_Transfer_of_Value") \
    .count() \
    .orderBy(F.desc("count")) \
    .show(10, truncate=False)


# COMMAND ----------

# 2. Top ten Nature of Payments by count
dflarge.groupBy("Nature_of_Payment_or_Transfer_of_Value") \
    .count() \
    .orderBy(F.desc("count")) \
    .show(10, truncate=False)

# COMMAND ----------

# 3. Top ten Nature of Payments by total amount
dflarge.groupBy("Nature_of_Payment_or_Transfer_of_Value") \
    .agg(F.sum("Total_Amount_of_Payment_USDollars").alias("Total_Payment")) \
    .orderBy(F.desc("Total_Payment")) \
    .show(10, truncate=False)


# COMMAND ----------

# 4. Top ten physician specialties by total amount
dflarge.groupBy("Covered_Recipient_Primary_Type_1") \
    .agg(F.sum("Total_Amount_of_Payment_USDollars").alias("Total_Payment")) \
    .orderBy(F.desc("Total_Payment")) \
    .show(10, truncate=False)

# COMMAND ----------

# 5. Top ten physicians by total amount
dflarge.groupBy("Covered_Recipient_First_Name", "Covered_Recipient_Last_Name") \
    .agg(F.sum("Total_Amount_of_Payment_USDollars").alias("Total_Payment")) \
    .orderBy(F.desc("Total_Payment")) \
    .show(10, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 
# MAGIC Nature of Payments with Reimbursement Amounts Greater Than $1,000 (Ordered by Count)
# MAGIC    According to their accounting data services generate the highest number of payments which include speaking engagements totaling 10,375 along with consulting fees across 5,676 records and honoraria amounting to 3,760. Travel and lodging expenses comprise a noteworthy 1,706 payments while royalties and grants made up numbers of 877 and 274 respectively.  
# MAGIC
# MAGIC The top ten payment types based on their appearance frequencies
# MAGIC    Payments for food and beverages represent the majority of 966,577 transactions while travel and lodging payments amount to 29,314 and education-related expenditures stand at 18,743. Among all payments for services compensation and consulting fees and honoraria each reach more than 5000 counts.  
# MAGIC
# MAGIC The distribution of payments demonstrates food and beverage payments as the leading nature of payment with a total amount of $58.45 million. 
# MAGIC    Among the top payments university sport received royalties amounting to $58.45 million while compensation for services reached $47.76 million and consulting fees reached $29.02 million. Total payments for food and beverages reached $27.87M and represented a major portion of disclosed payments next to honorariums combined with travel reimbursements.  
# MAGIC
# MAGIC Medical Doctors received the largest share of payments as a specialty group followed by unspecified physicians and Nurse Practitioners in the statistics.
# MAGIC    Medical Doctors procured the largest total payments ($141.97M) which exceeded unspecified recipients ($36.29M) and totaled $11.17M for Nurse Practitioners. The payment amounts to Physician Assistants, Osteopaths and Dentists were marked as significant.  
# MAGIC
# MAGIC  Medical doctors and unspecified personnel emerged as the top ten physicians who received the largest total payments.
# MAGIC   An unspecified individual received the most compensation of $36.29 million to date but the highest paid doctor was David Zamierowski who earned $3.46 million. Among major payees Hyun Bae stands at the second position with $2.74 million followed by Thomas Fehring receiving $2.61 million and third place holder is Chitranjan Ranawat who obtained $2.58 million.  
# MAGIC
