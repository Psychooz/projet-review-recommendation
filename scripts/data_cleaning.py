from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, regexp_replace, when, monotonically_increasing_id
from pyspark.sql.types import IntegerType, FloatType
import logging

def clean_reviews(input_path, output_path):
    spark = SparkSession.builder.appName("DataCleaning").getOrCreate()
    
    # Lecture des données
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # 1. Suppression des lignes avec valeurs manquantes critiques
    df = df.dropna(subset=['productId', 'userId', 'score'])
    
    # 2. Conversion des scores en float et nettoyage
    df = df.withColumn('score_clean', 
                      col('score').cast(FloatType()))
    
    # 3. Filtrage des scores valides (1-5) et non nuls
    df = df.filter(
        (col('score_clean') >= 1) & 
        (col('score_clean') <= 5) &
        col('score_clean').isNotNull()
    )
    
    # 4. Nettoyage des prix
    df = df.withColumn('price_clean', 
                      regexp_replace(col('price'), r'[^0-9.]', '').cast(FloatType()))
    
    # 5. Création d'IDs numériques
    user_mapping = df.select('userId').distinct() \
        .withColumn('userNumericId', monotonically_increasing_id().cast(IntegerType()))
    
    product_mapping = df.select('productId').distinct() \
        .withColumn('productNumericId', monotonically_increasing_id().cast(IntegerType()))
    
    # 6. Jointure avec les mappings
    df = df.join(user_mapping, 'userId') \
           .join(product_mapping, 'productId')
    
    # 7. Sélection des colonnes finales
    clean_df = df.select(
        col('productNumericId').alias('productId'),
        col('userNumericId').alias('userId'),
        col('score_clean').alias('score'),
        col('title'),
        col('price_clean').alias('price'),
        col('timestamp')
    )
    
    # Vérification finale des valeurs nulles
    if clean_df.filter(col('score').isNull()).count() > 0:
        raise ValueError("Il reste des scores nuls après le nettoyage")
    
    # Sauvegarde
    clean_df.write.mode('overwrite').csv(output_path, header=True)
    user_mapping.write.mode('overwrite').csv(output_path + "_user_mapping", header=True)
    product_mapping.write.mode('overwrite').csv(output_path + "_product_mapping", header=True)
    
    return clean_df

if __name__ == "__main__":
    input_path = "data/reviews.csv"  # Mettez le bon chemin vers votre fichier
    output_path = "processed_data/cleaned_reviews.csv"
    clean_reviews(input_path, output_path)