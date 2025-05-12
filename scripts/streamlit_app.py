import streamlit as st
from spark_recommendation import RecommendationEngine
import logging
import pandas as pd

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamlitApp")

# Initialiser le moteur de recommandation
@st.cache_resource
def load_recommendation_engine():
    return RecommendationEngine("processed_data/cleaned_reviews.csv")

def display_recommendations(recommendations):
    if not recommendations:
        st.warning("Aucune recommandation trouvée.")
        return
    
    # Convert to DataFrame with proper null handling
    rec_list = []
    for rec in recommendations:
        price = rec['price']
        rec_list.append({
            'Produit': rec['title'],
            'Prix': f"${price:.2f}" if price is not None else "N/A",
            'ID Produit': rec['productId']
        })
    
    rec_df = pd.DataFrame(rec_list)
    st.table(rec_df)

def main():
    st.set_page_config(page_title="Système de Recommandation E-commerce", layout="wide")
    st.title("🎯 Système de Recommandation E-commerce")
    
    engine = load_recommendation_engine()
    
    st.sidebar.header("Options")
    option = st.sidebar.radio(
        "Type de recommandation",
        ["Pour un utilisateur", "Pour un produit"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Exemples d'IDs à tester:\n"
        "- Utilisateur: A31KXTOQNTWUVM\n"
        "- Produit: B000EENAE0"
    )
    
    if option == "Pour un utilisateur":
        st.header("🔍 Recommandations personnalisées")
        user_id = st.text_input("Entrez l'ID de l'utilisateur:", "A31KXTOQNTWUVM")
        
        if st.button("Générer des recommandations", key="user_rec"):
            with st.spinner("Recherche des recommandations..."):
                try:
                    recommendations = engine.recommend_for_user(user_id)
                    st.subheader("Résultats")
                    display_recommendations(recommendations)
                except Exception as e:
                    st.error(f"Une erreur est survenue: {str(e)}")
    
    else:
        st.header("🛍️ Produits similaires")
        product_id = st.text_input("Entrez l'ID du produit:", "B000EENAE0")
        
        if st.button("Trouver des produits similaires", key="product_rec"):
            with st.spinner("Recherche de produits similaires..."):
                try:
                    similar_products = engine.recommend_for_product(product_id)
                    st.subheader("Résultats")
                    display_recommendations(similar_products)
                except Exception as e:
                    st.error(f"Une erreur est survenue: {str(e)}")
                    
if __name__ == "__main__":
    main()