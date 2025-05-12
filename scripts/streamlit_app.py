import streamlit as st
import sys
import os

# Add script directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import the recommendation engine
from spark_recommendation import RecommendationEngine

def load_recommendation_engine(csv_path):
    """
    Load recommendation engine with error handling
    
    Args:
        csv_path (str): Path to reviews CSV
    
    Returns:
        RecommendationEngine or None
    """
    try:
        return RecommendationEngine(csv_path)
    except Exception as e:
        st.error(f"Failed to load recommendation engine: {e}")
        return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Product Recommendation System", 
        page_icon="🛍️", 
        layout="wide"
    )
    
    # Title and description
    st.title("🎯 Product Recommendation System")
    st.markdown("""
    ### Discover Personalized Product Recommendations
    
    Choose between:
    - User-based Recommendations
    - Similar Product Recommendations
    """)
    
    # Load recommendation engine
    csv_path = "data/reviews.csv"  # Adjust path as needed
    engine = load_recommendation_engine(csv_path)
    
    if engine is None:
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("Recommendation Options")
    rec_type = st.sidebar.radio(
        "Select Recommendation Type",
        ["User Recommendations", "Similar Products"],
        index=0
    )
    
    # Example IDs
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Example IDs:\n"
        "- User ID: A31KXTOQNTWUVM\n"
        "- Product ID: B000EENAE0"
    )
    
    # User Recommendations Section
    if rec_type == "User Recommendations":
        st.header("👤 Personalized Recommendations")
        
        user_id = st.text_input(
            "Enter User ID", 
            value="A31KXTOQNTWUVM",
            help="Input a specific user ID"
        )
        
        if st.button("Get Recommendations", key="user_rec"):
            with st.spinner("Generating recommendations..."):
                try:
                    recommendations = engine.recommend_for_user(user_id)
                    
                    if len(recommendations) > 0:
                        st.subheader(f"Recommendations for User {user_id}")
                        st.dataframe(recommendations[['title', 'price', 'recommendation_score']])
                    else:
                        st.warning("No recommendations found.")
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
    
    # Similar Products Section
    else:
        st.header("🔍 Find Similar Products")
        
        product_id = st.text_input(
            "Enter Product ID", 
            value="B000EENAE0",
            help="Input a product ID to find similar products"
        )
        
        if st.button("Find Similar Products", key="product_rec"):
            with st.spinner("Searching for similar products..."):
                try:
                    similar_products = engine.recommend_similar_products(product_id)
                    
                    if len(similar_products) > 0:
                        st.subheader(f"Products Similar to {product_id}")
                        st.dataframe(similar_products[['title', 'price', 'similarity_score']])
                    else:
                        st.warning("No similar products found.")
                except Exception as e:
                    st.error(f"Error finding similar products: {e}")

    # Footer
    st.markdown("---")
    st.markdown("*Powered by Machine Learning Recommendation Engine*")

if __name__ == "__main__":
    main()