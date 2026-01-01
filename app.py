import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)



ABBREVIATIONS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "dl": "deep learning"
}


# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set page config
st.set_page_config(
    page_title="Duplicate Questions Detector",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("🔍 Duplicate Questions Detector")
st.markdown("""
This NLP-based app detects whether two questions are semantically similar (duplicates) or not.
The model uses semantic search techniques to compare question meaning.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
**How it works:**
1. Enter two questions in the text boxes
2. The app preprocesses and analyzes them
3. Semantic features are extracted
4. A trained ML model predicts similarity
5. Results with confidence score are displayed
""")

st.sidebar.header("Model Info")
st.sidebar.text("Algorithm: Random Forest")
st.sidebar.text("Features: TF-IDF, text statistics")
st.sidebar.text("Accuracy: 69.30%")

# Load model and resources
@st.cache_resource
def load_model():
    """Load the trained model and resources"""
    try:
        model = joblib.load('models/duplicate_model.pkl')
        tfidf = joblib.load('models/tfidf_vectorizer.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        return model, tfidf, feature_cols
    except:
        st.error("Model files not found. Please run train_model.py first!")
        return None, None, None

model, tfidf, feature_cols = load_model()

def preprocess_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def extract_features_from_pair(q1, q2, tfidf):
    """Extract features from a question pair"""
    # Preprocess questions
    q1_clean = preprocess_text(q1)
    q2_clean = preprocess_text(q2)
    
    # Basic text features
    q1_len = len(q1_clean)
    q2_len = len(q2_clean)
    len_diff = abs(q1_len - q2_len)
    
    q1_word_count = len(q1_clean.split())
    q2_word_count = len(q2_clean.split())
    word_count_diff = abs(q1_word_count - q2_word_count)
    
    # Common words feature
    common_words = len(set(q1_clean.split()) & set(q2_clean.split()))
    
    # TF-IDF features and cosine similarity
    try:
        # Transform questions
        q1_tfidf = tfidf.transform([q1_clean])
        q2_tfidf = tfidf.transform([q2_clean])
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(q1_tfidf, q2_tfidf)[0][0]
    except:
        cosine_sim = 0
    
    # Create feature dictionary
    features = {
        'q1_len': q1_len,
        'q2_len': q2_len,
        'len_diff': len_diff,
        'q1_word_count': q1_word_count,
        'q2_word_count': q2_word_count,
        'word_count_diff': word_count_diff,
        'common_words': common_words,
        'cosine_similarity': cosine_sim
    }
    
    return features, q1_clean, q2_clean

def main():
    """Main app interface"""
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        question1 = st.text_area(
            "Enter first question:",
            "What is machine learning?",
            height=100,
            key="q1"
        )
    
    with col2:
        question2 = st.text_area(
            "Enter second question:",
            "How does ML work?",
            height=100,
            key="q2"
        )
    
    # Add analyze button
    analyze_button = st.button("🔍 Analyze Questions", type="primary", use_container_width=True)
    
    if analyze_button and model is not None:
        with st.spinner("Analyzing questions..."):
            # Extract features
            features, q1_clean, q2_clean = extract_features_from_pair(question1, question2, tfidf)
            
            # Convert to dataframe for prediction
            features_df = pd.DataFrame([features])[feature_cols]
            
            # Make prediction
            prediction = model.predict(features_df)[0]
            prediction_proba = model.predict_proba(features_df)[0]
            
            # Display results
            st.markdown("---")
            
            # Result in a nice container
            result_container = st.container()
            
            with result_container:
                if prediction == 1:
                    st.success(f"✅ **The questions are DUPLICATE (similar)**")
                else:
                    st.error(f"❌ **The questions are NOT DUPLICATE (different)**")
                
                # Confidence score
                confidence = prediction_proba[prediction] * 100
                st.info(f"**Confidence:** {confidence:.1f}%")
            
            # Display features and visualization
            st.markdown("---")
            st.subheader("📊 Analysis Details")
            
            # Create three columns for details
            col_analysis1, col_analysis2, col_analysis3 = st.columns(3)
            
            with col_analysis1:
                st.markdown("**Preprocessed Questions:**")
                st.text(f"Q1: {q1_clean[:50]}..." if len(q1_clean) > 50 else f"Q1: {q1_clean}")
                st.text(f"Q2: {q2_clean[:50]}..." if len(q2_clean) > 50 else f"Q2: {q2_clean}")
            
            with col_analysis2:
                st.markdown("**Key Features:**")
                st.text(f"Cosine Similarity: {features['cosine_similarity']:.3f}")
                st.text(f"Common Words: {features['common_words']}")
                st.text(f"Length Difference: {features['len_diff']}")
            
            with col_analysis3:
                st.markdown("**Prediction Probabilities:**")
                st.text(f"Duplicate: {prediction_proba[1]*100:.1f}%")
                st.text(f"Not Duplicate: {prediction_proba[0]*100:.1f}%")
            
            # Visualization
            st.markdown("---")
            st.subheader("📈 Visualizations")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Create similarity gauge
                fig1, ax1 = plt.subplots(figsize=(6, 3))
                
                # Simple bar chart for similarity
                labels = ['Cosine Similarity']
                values = [features['cosine_similarity']]
                
                ax1.bar(labels, values, color=['skyblue' if values[0] < 0.5 else 'lightgreen'])
                ax1.set_ylim(0, 1)
                ax1.set_ylabel('Similarity Score')
                ax1.set_title('Semantic Similarity Score')
                
                # Add threshold line
                ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
                ax1.legend()
                
                st.pyplot(fig1)
            
            with viz_col2:
                # Create probability chart
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                
                categories = ['Not Duplicate', 'Duplicate']
                probabilities = prediction_proba * 100
                
                colors = ['lightcoral', 'lightgreen']
                bars = ax2.bar(categories, probabilities, color=colors)
                
                ax2.set_ylim(0, 100)
                ax2.set_ylabel('Probability (%)')
                ax2.set_title('Prediction Probabilities')
                
                # Add value labels on bars
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{prob:.1f}%', ha='center', va='bottom')
                
                st.pyplot(fig2)
            
            # Show saved visualizations from training
            st.markdown("---")
            st.subheader("📚 Model Performance Visualizations")
            
            try:
                viz_col3, viz_col4 = st.columns(2)
                
                with viz_col3:
                    st.image("models/confusion_matrix.png", 
                            caption="Confusion Matrix")
                
                with viz_col4:
                    st.image("models/feature_importance.png", 
                            caption="Feature Importance")
                
                st.image("models/similarity_distribution.png", 
                        caption="Similarity Distribution")
            except:
                st.info("Training visualizations not available. Run train_model.py to generate them.")
    
    elif analyze_button and model is None:
        st.error("Model not loaded. Please run train_model.py first to train and save the model.")
    

if __name__ == "__main__":
    main()