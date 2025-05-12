from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, lit, avg, desc
import logging
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RecommendationEngine")

class RecommendationEngine:
    def __init__(self, data_path):
        self.spark = SparkSession.builder \
            .appName("EcommerceRecommendation") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        
        # Chargement des données et des mappings
        self.df = self.spark.read.csv(data_path, header=True, inferSchema=True)
        
        # Charger les mappings utilisateurs et produits
        base_path = os.path.dirname(data_path)
        self.user_mapping = self.spark.read.csv(
            os.path.join(base_path, "cleaned_reviews.csv_user_mapping"), 
            header=True, 
            inferSchema=True
        )
        self.product_mapping = self.spark.read.csv(
            os.path.join(base_path, "cleaned_reviews.csv_product_mapping"), 
            header=True, 
            inferSchema=True
        )
        
        # Entraînement du modèle
        self.als_model = self._train_model()
        self.top_products = self._prepare_top_products()

    def _train_model(self):
        als = ALS(
            maxIter=5,
            regParam=0.01,
            userCol="userId",
            itemCol="productId",
            ratingCol="score",
            coldStartStrategy="drop",
            nonnegative=True,
            seed=42
        )
        return als.fit(self.df.select("userId", "productId", "score"))

    def _prepare_top_products(self):
        """Prépare les produits les mieux notés pour les nouveaux utilisateurs"""
        return self.df.groupBy("productId", "title", "price") \
                    .agg(avg("score").alias("avg_score")) \
                    .orderBy(desc("avg_score")) \
                    .limit(100) \
                    .collect()

    def _get_numeric_id(self, original_id, mapping_df, id_col='userId'):
        """Convertit un ID original en ID numérique"""
        try:
            result = mapping_df.filter(col(id_col) == original_id).collect()
            return result[0]['userNumericId'] if result else None
        except Exception as e:
            logging.error(f"Erreur lors de la conversion de l'ID: {e}")
            return None

    def recommend_for_user(self, user_id, num_recommendations=5):
        """Génère des recommandations pour un utilisateur"""
        try:
            # Conversion de l'ID utilisateur
            numeric_id = self._get_numeric_id(user_id, self.user_mapping)
            
            if numeric_id is None:
                logging.warning(f"Utilisateur {user_id} non trouvé - retour des meilleurs produits")
                return self.top_products[:num_recommendations]
            
            # Génération des recommandations
            user_df = self.spark.createDataFrame([(numeric_id,)], ["userId"])
            recs = self.als_model.recommendForUserSubset(user_df, num_recommendations).collect()
            
            if not recs:
                return self.top_products[:num_recommendations]
                
            # Conversion des IDs produits
            product_recs = recs[0]['recommendations']
            numeric_ids = [rec['productId'] for rec in product_recs]
            
            # Récupération des détails des produits
            products = self.product_mapping.filter(col('productNumericId').isin(numeric_ids))
            details = products.join(
                self.df.select('productNumericId', 'title', 'price').distinct(),
                'productNumericId'
            ).collect()
            
            return details if details else self.top_products[:num_recommendations]
            
        except Exception as e:
            logging.error(f"Erreur dans recommend_for_user: {e}")
            return self.top_products[:num_recommendations]

    def recommend_for_product(self, product_id, num_recommendations=5):
        """Génère des produits similaires"""
        try:
            # Conversion de l'ID produit
            numeric_id = self._get_numeric_id(product_id, self.product_mapping, 'productId')
            
            if numeric_id is None:
                logging.warning(f"Produit {product_id} non trouvé - retour des meilleurs produits")
                return self.top_products[:num_recommendations]
            
            # Trouver les utilisateurs qui ont aimé ce produit
            product_users = self.df.filter(
                (col('productId') == numeric_id) & 
                (col('score') >= 4)
            ).select('userId').distinct()
            
            if product_users.count() == 0:
                return self.top_products[:num_recommendations]
            
            # Générer les recommandations
            user_recs = self.als_model.recommendForUserSubset(product_users, num_recommendations)
            all_recs = []
            for row in user_recs.collect():
                all_recs.extend(row['recommendations'])
            
            # Compter les occurrences des produits recommandés
            from collections import defaultdict
            product_counts = defaultdict(int)
            for rec in all_recs:
                if rec['productId'] != numeric_id:
                    product_counts[rec['productId']] += 1
            
            # Trier et prendre les meilleurs
            sorted_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
            top_numeric_ids = [x[0] for x in sorted_products[:num_recommendations]]
            
            # Récupérer les détails
            products = self.product_mapping.filter(col('productNumericId').isin(top_numeric_ids))
            details = products.join(
                self.df.select('productNumericId', 'title', 'price').distinct(),
                'productNumericId'
            ).collect()
            
            return details if details else self.top_products[:num_recommendations]
            
        except Exception as e:
            logging.error(f"Erreur dans recommend_for_product: {e}")
            return self.top_products[:num_recommendations]
    
    def _get_top_rated_products(self, num_recommendations):
        """Retourne les produits les mieux notés en cas d'utilisateur inconnu"""
        logger.info("Retour des produits les mieux notés")
        
        top_products = self.df.groupBy("productId", "title", "price") \
                             .agg({"score": "avg"}) \
                             .orderBy("avg(score)", ascending=False) \
                             .limit(num_recommendations) \
                             .collect()
        
        return top_products
    
    def recommend_for_product(self, product_id, num_recommendations=5):
        logger.info(f"Génération de produits similaires pour le produit {product_id}")
        
        # Trouver les utilisateurs qui ont aimé ce produit
        product_users = self.df.filter((col("productId") == product_id) & 
                              (col("score") >= 4)) \
                              .select("userId") \
                              .distinct()
        
        if product_users.count() == 0:
            logger.warning(f"Aucun utilisateur n'a aimé le produit {product_id}")
            return self._get_top_rated_products(num_recommendations)
        
        # Obtenir les recommandations pour ces utilisateurs
        user_recs = self.als_model.recommendForUserSubset(product_users, num_recommendations)
        
        # Compter les occurrences des produits recommandés
        all_recs = []
        for row in user_recs.collect():
            all_recs.extend(row['recommendations'])
        
        # Trouver les produits les plus fréquemment recommandés
        from collections import defaultdict
        product_counts = defaultdict(int)
        for rec in all_recs:
            if rec['productId'] != product_id:
                product_counts[rec['productId']] += 1
        
        # Trier et prendre les meilleurs
        sorted_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
        top_product_ids = [x[0] for x in sorted_products[:num_recommendations]]
        
        # Récupérer les détails des produits
        product_details = self.df.filter(col("productId").isin(top_product_ids)) \
                                .select("productId", "title", "price") \
                                .distinct() \
                                .collect()
        
        return product_details

if __name__ == "__main__":
    data_path = "processed_data/cleaned_reviews.csv"
    engine = RecommendationEngine(data_path)
    
    # Exemple d'utilisation
    user_id = "A31KXTOQNTWUVM"  # Utilisateur de l'exemple
    print("\nRecommandations pour l'utilisateur:", user_id)
    user_recs = engine.recommend_for_user(user_id)
    if user_recs:
        for product in user_recs:
            print(f"- {product['title']} (Prix: ${product['price']})")
    
    product_id = "B000EENAE0"  # Produit de l'exemple
    print("\nProduits similaires à:", product_id)
    product_recs = engine.recommend_for_product(product_id)
    if product_recs:
        for product in product_recs:
            print(f"- {product['title']} (Prix: ${product['price']})")