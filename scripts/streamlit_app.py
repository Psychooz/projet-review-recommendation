import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def show_statistics(engine):
    """Display various statistics about the data"""
    st.header("📊 Data Statistics and Insights")
    
    # Basic statistics
    st.subheader("Basic Dataset Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(engine.df))
    col2.metric("Unique Users", len(engine.user_ids))
    col3.metric("Unique Products", len(engine.product_ids))
    
    # Create a copy of the dataframe for visualization
    viz_df = engine.df.copy()
    
    # Convert score to numeric if it's not already
    viz_df['score'] = pd.to_numeric(viz_df['score'], errors='coerce')
    
    # Rating distribution
    st.subheader("Rating Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='score', data=viz_df.dropna(subset=['score']), ax=ax)
    ax.set_title("Distribution of Product Ratings")
    st.pyplot(fig)
    
    # Price distribution
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(viz_df['price'].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Product Prices")
    ax.set_xlabel("Price")
    st.pyplot(fig)
    
    # Top rated products
    st.subheader("Top Rated Products (with most reviews)")
    top_products = engine.product_stats.sort_values(
        ['avg_score', 'review_count'], 
        ascending=[False, False]
    ).head(10)
    st.dataframe(top_products[['title', 'price', 'avg_score', 'review_count']])
    
    # Most active users
    st.subheader("Most Active Users")
    user_activity = engine.df['userId'].value_counts().reset_index()
    user_activity.columns = ['userId', 'review_count']
    st.dataframe(user_activity.head(10))
    
    # Price vs. Rating analysis
    st.subheader("Price vs. Rating Analysis")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x='price', 
        y='avg_score', 
        size='review_count',
        sizes=(20, 200),
        alpha=0.6,
        data=engine.product_stats,
        ax=ax
    )
    ax.set_title("Price vs. Average Rating")
    st.pyplot(fig)

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
    - Data Statistics
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
        ["User Recommendations", "Similar Products", "Data Statistics"],
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
    elif rec_type == "Similar Products":
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
    
    # Statistics Section
    else:
        show_statistics(engine)

    # Footer
    st.markdown("---")
    st.markdown("*Made By Ziad Boukhalkhal - Khalil Hamdaoui*")

if __name__ == "__main__":
    main()