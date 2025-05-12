import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self, csv_path):
        """
        Initialize recommendation engine with robust data loading
        
        Args:
            csv_path (str): Path to the reviews CSV file
        """
        try:
            # Read CSV with flexible parsing
            self.df = self._load_csv(csv_path)
            
            # Clean and prepare data
            self._prepare_data()
            
            logger.info(f"Loaded {len(self.df)} reviews")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _load_csv(self, csv_path):
        """
        Load CSV with robust parsing
        
        Args:
            csv_path (str): Path to CSV file
        
        Returns:
            pandas.DataFrame: Cleaned DataFrame
        """
        # Custom converters to handle problematic columns
        def safe_float_convert(val):
            """
            Safely convert value to float
            
            Args:
                val (str): Input value
            
            Returns:
                float or np.nan: Converted value
            """
            if pd.isna(val) or val == 'unknown':
                return np.nan
            
            # Remove any non-numeric characters except decimal point
            try:
                # Remove currency symbols, commas, etc.
                clean_val = re.sub(r'[^\d.]', '', str(val))
                return float(clean_val) if clean_val else np.nan
            except (ValueError, TypeError):
                return np.nan

        def safe_str_convert(val):
            """
            Safely convert to string, handling NaN
            
            Args:
                val (any): Input value
            
            Returns:
                str: Converted value or empty string
            """
            if pd.isna(val):
                return ''
            return str(val).strip()

        # Read CSV with custom converters
        df = pd.read_csv(
            csv_path, 
            dtype={
                'productId': str, 
                'userId': str
            },
            converters={
                'price': safe_float_convert,
                'score': safe_float_convert,
                'title': safe_str_convert
            }
        )
        
        return df

    def _prepare_data(self):
        """
        Clean and prepare data for recommendations
        """
        # Remove rows with missing critical information
        self.df.dropna(subset=['productId', 'userId', 'score'], inplace=True)
        
        # Filter out invalid scores and prices
        self.df = self.df[
            (self.df['score'] >= 1) & 
            (self.df['score'] <= 5) & 
            (self.df['price'] > 0)
        ]
        
        # Remove duplicate reviews
        self.df.drop_duplicates(subset=['productId', 'userId'], inplace=True)
        
        # Create product-level aggregations
        self.product_stats = self.df.groupby('productId').agg({
            'title': 'first',  # First unique title
            'price': 'mean',   # Average price
            'score': ['count', 'mean']  # Review count and average score
        }).reset_index()
        
        self.product_stats.columns = ['productId', 'title', 'price', 'review_count', 'avg_score']

    def recommend_for_user(self, user_id, n_recommendations=5):
        """
        Generate recommendations for a specific user
        
        Args:
            user_id (str): User ID
            n_recommendations (int): Number of recommendations
        
        Returns:
            DataFrame: Recommended products
        """
        try:
            # Validate user_id
            if not isinstance(user_id, str):
                user_id = str(user_id)
            
            # Get products user has already reviewed
            user_reviewed_products = set(
                self.df[self.df['userId'] == user_id]['productId']
            )
            
            # Get user's reviews
            user_ratings = self.df[self.df['userId'] == user_id]
            
            # If user not found, return top-rated products
            if len(user_ratings) == 0:
                logger.warning(f"User {user_id} not found. Returning top products.")
                return self._get_top_products(n_recommendations)
            
            # Find similar users based on rating patterns
            similar_users_ratings = self.df[
                ~self.df['productId'].isin(user_reviewed_products)
            ]
            
            # Score recommendations
            def compute_similarity(group):
                """
                Compute recommendation score for a product
                """
                try:
                    # Compare ratings distribution
                    return np.corrcoef(
                        group['score'], 
                        user_ratings['score']
                    )[0,1]
                except Exception:
                    return 0
            
            recommendation_scores = similar_users_ratings.groupby('productId').apply(compute_similarity)
            
            # Get top recommendations
            top_recommendations = recommendation_scores.nlargest(n_recommendations)
            
            # Merge with product stats
            recommended_products = self.product_stats[
                self.product_stats['productId'].isin(top_recommendations.index)
            ].copy()
            
            # Add recommendation score
            recommended_products['recommendation_score'] = top_recommendations.values
            
            return recommended_products.sort_values('recommendation_score', ascending=False)
        
        except Exception as e:
            logger.error(f"Recommendation error for user {user_id}: {e}")
            return self._get_top_products(n_recommendations)

    def recommend_similar_products(self, product_id, n_recommendations=5):
        """
        Find similar products based on title
        
        Args:
            product_id (str): Product ID to find similar products for
            n_recommendations (int): Number of recommendations
        
        Returns:
            DataFrame: Similar products
        """
        try:
            # Validate product_id
            if not isinstance(product_id, str):
                product_id = str(product_id)
            
            # Get current product details
            current_product = self.product_stats[
                self.product_stats['productId'] == product_id
            ]
            
            if len(current_product) == 0:
                logger.warning(f"Product {product_id} not found")
                return self._get_top_products(n_recommendations)
            
            # Prepare titles for similarity
            titles = self.product_stats['title'].fillna('')
            
            # Create TF-IDF vectorizer
            tfidf = TfidfVectorizer(stop_words='english')
            
            # Compute TF-IDF matrix
            tfidf_matrix = tfidf.fit_transform(titles)
            
            # Find similar products using cosine similarity
            current_product_title = current_product['title'].values[0]
            current_product_tfidf = tfidf.transform([current_product_title])
            cosine_similarities = cosine_similarity(current_product_tfidf, tfidf_matrix)[0]
            
            # Get top similar product indices
            similar_indices = cosine_similarities.argsort()[::-1][1:n_recommendations+1]
            
            # Return similar products
            similar_products = self.product_stats.iloc[similar_indices].copy()
            similar_products['similarity_score'] = cosine_similarities[similar_indices]
            
            return similar_products.sort_values('similarity_score', ascending=False)
        
        except Exception as e:
            logger.error(f"Similar product error for {product_id}: {e}")
            return self._get_top_products(n_recommendations)

    def _get_top_products(self, n_products=5):
        """
        Get top-rated products
        
        Args:
            n_products (int): Number of top products to return
        
        Returns:
            DataFrame: Top-rated products
        """
        return (
            self.product_stats
            .sort_values(['avg_score', 'review_count'], ascending=[False, False])
            .head(n_products)
        )

def main():
    # Example usage
    try:
        # Path to your CSV file
        csv_path = 'data/reviews.csv'
        
        # Initialize recommendation engine
        engine = RecommendationEngine(csv_path)
        
        # Example user recommendations
        print("\n=== User Recommendations ===")
        user_id = "A31KXTOQNTWUVM"
        user_recs = engine.recommend_for_user(user_id)
        print(user_recs[['title', 'price', 'recommendation_score']])
        
        # Example similar products
        print("\n=== Similar Products ===")
        product_id = "B000EENAE0"
        similar_products = engine.recommend_similar_products(product_id)
        print(similar_products[['title', 'price', 'similarity_score']])
    
    except Exception as e:
        logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()