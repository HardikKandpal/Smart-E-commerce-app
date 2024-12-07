import streamlit as st
import pandas as pd
from recc.content_based import content_based_recc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recc.collaborative import collaborative_filtering_recommendations
from  price_comp import scrape_amazon, scrape_flipkart
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from db import *





@st.cache_data
def load_data():
    train_data = pd.read_csv('data/product_review.tsv', sep='\t')
    train_data = train_data[['Uniq Id', 'Product Id', 'Product Rating', 'Product Reviews Count', 
                             'Product Category', 'Product Brand', 'Product Name', 
                             'Product Image Url', 'Product Description', 'Product Tags']]
    train_data.rename(columns={
        'Uniq Id': 'ID',
        'Product Id': 'ProdID',
        'Product Rating': 'Rating',
        'Product Reviews Count': 'ReviewCount',
        'Product Category': 'Category',
        'Product Brand': 'Brand',
        'Product Name': 'Name',
        'Product Image Url': 'ImageURL',
        'Product Description': 'Description',
        'Product Tags': 'Tags',
    }, inplace=True)

    # Handle missing values
    train_data['Rating'].fillna(train_data['Rating'].mean(), inplace=True)
    train_data['ReviewCount'].fillna(0, inplace=True)
    train_data['Category'].fillna('', inplace=True)
    train_data['Brand'].fillna('', inplace=True)
    train_data['Description'].fillna('', inplace=True)

    # Extract numeric parts from ID and ProdID
    train_data['ID'] = train_data['ID'].str.extract(r'(\d+)').astype(float)
    train_data['ProdID'] = train_data['ProdID'].str.extract(r'(\d+)').astype(float)

    # Extract only the first image URL
    train_data['ImageURL'] = train_data['ImageURL'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else x)

    # Generate tags for better content-based filtering
    nlp = spacy.load("en_core_web_sm")
    def clean_and_extract_tags(text):
        doc = nlp(text.lower())
        tags = [token.text for token in doc if token.text.isalnum() and token.text not in STOP_WORDS]
        return ', '.join(tags)

    columns_to_extract_tags_from = ['Category', 'Brand', 'Description']
    for column in columns_to_extract_tags_from:
        train_data[column] = train_data[column].apply(clean_and_extract_tags)

    train_data['Tags'] = train_data[columns_to_extract_tags_from].apply(lambda row: ', '.join(row), axis=1)

    return train_data


# Recommendation Functions
def get_trending_products(train_data, top_n=5):
    """Get top trending products based on review counts and ratings."""
    return train_data.sort_values(['ReviewCount', 'Rating'], ascending=False).head(top_n)


def content_based_recc(train_data, item_name, top_n=10):
    """Generate content-based recommendations."""
    if item_name not in train_data['Name'].values:
        return pd.DataFrame({'Error': [f"Item '{item_name}' not found in the database."]})

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]

    return train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)

    user_similarity = cosine_similarity(user_item_matrix)

    if target_user_id not in user_item_matrix.index:
        return pd.DataFrame({'Error': [f"User ID '{target_user_id}' not found in the dataset."]})

    target_user_index = user_item_matrix.index.get_loc(target_user_id)
    user_similarities = user_similarity[target_user_index]
    similar_users_indices = user_similarities.argsort()[::-1][1:]

    recommended_items = []

    for user_index in similar_users_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user > 0) & (user_item_matrix.iloc[target_user_index] == 0)

        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])

    recommended_items_details = train_data[train_data['ProdID'].isin(recommended_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details.head(top_n)

def hybrid_recommendations(train_data, target_user_id, item_name, top_n):
    # Ensure the queried product is included
    queried_product = train_data[train_data['Name'] == item_name][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]
    
    if queried_product.empty:
        return pd.DataFrame({'Error': [f"Item '{item_name}' not found in the database."]})
    
    # Content-based recommendations
    content_based_rec = content_based_recc(train_data, item_name, top_n)
    
    # Collaborative filtering recommendations
    collaborative_filtering_rec = collaborative_filtering_recommendations(train_data, target_user_id, top_n)
    
    # Combine both recommendations
    hybrid_rec = pd.concat([queried_product, content_based_rec, collaborative_filtering_rec]).drop_duplicates()
    
    # Ensure queried product appears at the top
    hybrid_rec = pd.concat([queried_product, hybrid_rec[hybrid_rec['Name'] != item_name]])
    
    # Return only the top_n recommendations
    return hybrid_rec.head(top_n)



data = load_data()
trend = pd.read_csv('data/trending_products.csv')

st.title("🛍️ Smart Retail Store")

# Display trending products
st.subheader("🔥 Trending Products This Week")
trending_products = get_trending_products(trend, top_n=5)
for _, row in trending_products.iterrows():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(row['ImageURL'], width=100)
    with col2:
        st.write(f"**{row['Name']}**")
        st.write(f"⭐ Rating: {row['Rating']} | 💬 Reviews: {row['ReviewCount']}")
    st.markdown("---")

# Sidebar Inputs
item_name = st.sidebar.selectbox(
    "Search and select a product:",
    options=data['Name'].unique(),
    help="Choose a product for recommendations.",
)
user_id = st.sidebar.number_input("Enter your User ID:", min_value=1, step=1, value=4)  # Ensure user_id is defined
top_n = st.sidebar.slider("Number of recommendations to show:", 1, 20, 10)

if st.sidebar.button("Get Hybrid Recommendations"):
    if not item_name or user_id <= 0: 
        st.warning("Please provide valid inputs for both the product name and user ID.")
    else:
        hybrid_recs = hybrid_recommendations(data, user_id, item_name, top_n)
        
        if "Error" in hybrid_recs.columns:
            st.error(hybrid_recs['Error'].iloc[0])
        else:
            st.subheader(f"Top {top_n} Hybrid Recommendations for User : {user_id} and Item : '{item_name}'")
            
            # Display each recommendation
            for i, row in hybrid_recs.iterrows():
                if i == 0:
                    st.markdown("### **Queried Product:**")
                else:
                    st.markdown("---")
                    st.markdown(f"### **Recommendation #{i}:**")
                    
                st.image(row['ImageURL'], width=150)
                st.write(f"**{row['Name']}**")
                st.write(f"Brand: {row['Brand']} | Rating: {row['Rating']} | Reviews: {row['ReviewCount']}")