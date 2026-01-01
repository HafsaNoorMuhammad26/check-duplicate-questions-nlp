import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics.pairwise import cosine_similarity

ABBREVIATIONS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "dl": "deep learning"
}


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()

    # Expand abbreviations
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf'\b{abbr}\b', full, text)

    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def create_synthetic_data():
    """Create synthetic training data for demo purposes"""
    print("Creating synthetic training data...")
    
    # Sample questions
    questions = [
        # Similar pairs (label = 1) - 50 pairs
        ("What is machine learning?", "What is ML?", 1),
        ("How to learn Python?", "Best way to learn Python programming", 1),
        ("What is artificial intelligence?", "Define AI", 1),
        ("How to cook pasta?", "What's the recipe for pasta?", 1),
        ("Best programming language?", "Which programming language is best?", 1),
        ("How to lose weight?", "Ways to reduce weight", 1),
        ("What is climate change?", "Explain global warming", 1),
        ("How to invest in stocks?", "Stock market investment tips", 1),
        ("What is data science?", "Define data science", 1),
        ("How to make coffee?", "Coffee making method", 1),
        ("What is blockchain?", "Explain blockchain technology", 1),
        ("How to meditate?", "Meditation techniques", 1),
        ("What is COVID-19?", "Coronavirus definition", 1),
        ("How to study effectively?", "Effective study methods", 1),
        ("What is cloud computing?", "Cloud technology explained", 1),
        ("How to bake a cake?", "Cake baking recipe", 1),
        ("What is deep learning?", "Deep learning definition", 1),
        ("How to write a resume?", "Resume writing tips", 1),
        ("What is quantum computing?", "Quantum computer explanation", 1),
        ("How to learn English?", "English learning methods", 1),
        ("What is cryptocurrency?", "Digital currency explanation", 1),
        ("How to start a business?", "Business startup guide", 1),
        ("What is neural network?", "Neural networks explained", 1),
        ("How to cook rice?", "Rice cooking method", 1),
        ("What is big data?", "Big data analytics", 1),
        ("How to improve memory?", "Memory enhancement techniques", 1),
        ("What is IoT?", "Internet of Things explained", 1),
        ("How to make tea?", "Tea preparation method", 1),
        ("What is 5G technology?", "5G network explanation", 1),
        ("How to exercise at home?", "Home workout routine", 1),
        ("What is virtual reality?", "VR technology explained", 1),
        ("How to save money?", "Money saving tips", 1),
        ("What is cybersecurity?", "Cyber security definition", 1),
        ("How to learn coding?", "Programming learning guide", 1),
        ("What is augmented reality?", "AR technology explained", 1),
        ("How to make pizza?", "Pizza recipe at home", 1),
        ("What is machine vision?", "Computer vision explained", 1),
        ("How to manage time?", "Time management techniques", 1),
        ("What is robotics?", "Robotics engineering", 1),
        ("How to grow plants?", "Plant growth tips", 1),
        ("What is natural language processing?", "NLP explained", 1),
        ("How to paint a room?", "Room painting guide", 1),
        ("What is computer programming?", "Coding definition", 1),
        ("How to swim?", "Swimming techniques", 1),
        ("What is software engineering?", "Software development", 1),
        ("How to cook chicken?", "Chicken recipes", 1),
        ("What is data mining?", "Data extraction techniques", 1),
        ("How to drive a car?", "Car driving lessons", 1),
        ("What is web development?", "Website creation", 1),
        ("How to make bread?", "Bread baking recipe", 1),
        
        # Dissimilar pairs (label = 0) - 50 pairs
        ("What is machine learning?", "How to cook pizza?", 0),
        ("Python programming basics", "Best pizza recipes", 0),
        ("AI and its applications", "Weather today", 0),
        ("Data science course", "Football match results", 0),
        ("How to learn coding?", "Gardening tips for beginners", 0),
        ("What is blockchain?", "Chicken curry recipe", 0),
        ("Deep learning algorithms", "Car maintenance guide", 0),
        ("Natural language processing", "Yoga exercises", 0),
        ("Cloud computing services", "Cake decorating ideas", 0),
        ("Quantum physics", "Coffee brewing techniques", 0),
        ("Software development", "Football team rankings", 0),
        ("Machine learning models", "Hair styling methods", 0),
        ("Data analysis techniques", "Movie reviews", 0),
        ("Artificial intelligence", "Music instruments list", 0),
        ("Web development frameworks", "Travel destinations", 0),
        ("Cybersecurity threats", "Cooking oil types", 0),
        ("Big data analytics", "Pet care guide", 0),
        ("IoT devices", "Fashion trends", 0),
        ("Neural networks", "Home cleaning tips", 0),
        ("Virtual reality headsets", "Book recommendations", 0),
        ("Robotics engineering", "Dance styles", 0),
        ("5G technology", "Restaurant reviews", 0),
        ("Computer vision", "Interior design ideas", 0),
        ("Cryptocurrency trading", "Fitness equipment", 0),
        ("Data mining algorithms", "Car brands comparison", 0),
        ("Augmented reality apps", "Weather forecasting", 0),
        ("Quantum computing", "Cooking recipes", 0),
        ("Network security", "Gardening tools", 0),
        ("Database management", "Musical instruments", 0),
        ("Operating systems", "Travel packing tips", 0),
        ("Programming languages", "Movie genres", 0),
        ("Software testing", "Coffee shop locations", 0),
        ("Mobile app development", "Exercise routines", 0),
        ("Computer hardware", "Recipe ingredients", 0),
        ("Data structures", "Fashion accessories", 0),
        ("Algorithms", "Home decor ideas", 0),
        ("Web design", "Vacation spots", 0),
        ("Cloud storage", "Cooking methods", 0),
        ("Machine translation", "Music bands", 0),
        ("Speech recognition", "Art techniques", 0),
        ("Predictive analytics", "Shopping malls", 0),
        ("Computer graphics", "Yoga poses", 0),
        ("Information retrieval", "Restaurant menus", 0),
        ("Recommender systems", "Car models", 0),
        ("Data visualization", "Haircut styles", 0),
        ("Text mining", "Gym exercises", 0),
        ("Pattern recognition", "Recipe books", 0),
        ("Expert systems", "Travel agencies", 0),
        ("Fuzzy logic", "Coffee beans types", 0),
        ("Genetic algorithms", "Movie theaters", 0),
        
        # More nuanced examples for better learning - 20 pairs
        ("How to code in Java?", "Java programming tutorial", 1),  # Similar
        ("Java vs Python", "Difference between Java and Python", 1),  # Similar
        ("Java programming", "Making coffee with Java beans", 0),  # Dissimilar (Java double meaning)
        ("Apple iPhone features", "Apple fruit nutrition", 0),  # Dissimilar (Apple double meaning)
        ("What is Amazon AWS?", "Amazon rainforest facts", 0),  # Dissimilar (Amazon double meaning)
        ("Python snake facts", "Python programming language", 0),  # Dissimilar
        ("How to use Git?", "Version control with Git", 1),  # Similar
        ("Git commands tutorial", "Learning Git basics", 1),  # Similar
        ("Git for beginners", "Cooking git fish recipe", 0),  # Dissimilar
        ("Bank account opening", "River bank erosion", 0),  # Dissimilar (bank double meaning)
        ("Cloud storage services", "Cloud formation in sky", 0),  # Dissimilar
        ("Mouse for computer", "Mouse animal facts", 0),  # Dissimilar
        ("Keyboard shortcuts", "Musical keyboard notes", 0),  # Dissimilar
        ("What is Twitter?", "Bird twittering sounds", 0),  # Dissimilar
        ("Facebook social media", "Face book for drawing", 0),  # Dissimilar
        ("Instagram photo sharing", "Instant telegram message", 0),  # Dissimilar
        ("LinkedIn professional network", "Link in chain", 0),  # Dissimilar
        ("Netflix streaming service", "Fishing net fixing", 0),  # Dissimilar
        ("Uber ride service", "Super uber vehicle", 0),  # Dissimilar
        ("Tesla electric cars", "Nikola Tesla inventor", 1),  # Similar (both about Tesla)
        
        # More challenging semantic pairs - 15 pairs
        ("How to become rich?", "Ways to earn money", 1),  # Similar
        ("How to learn Python?", "Ways to learn Python programming?", 1),
        ("Feeling sad today", "I am unhappy", 1),  # Similar
        ("Happy birthday wishes", "Best birthday messages", 1),  # Similar
        ("Global warming effects", "Climate change impact", 1),  # Similar
        ("Healthy food choices", "Nutrition diet plan", 1),  # Similar
        ("Study hard for exams", "Prepare for tests", 1),  # Similar
        ("Save water daily", "Conserve water resources", 1),  # Similar
        ("Learn new skills", "Acquire new abilities", 1),  # Similar
        ("Time is valuable", "Time management important", 1),  # Similar
        ("Exercise daily routine", "Workout regularly", 1),  # Similar
        ("Read books everyday", "Daily reading habit", 1),  # Similar
        ("Sleep early tonight", "Go to bed early", 1),  # Similar
        ("Drink more water", "Stay hydrated always", 1),  # Similar
        ("Eat fresh fruits", "Consume healthy fruits", 1),  # Similar
        ("Walk for 30 minutes", "Take a walk daily", 1),  # Similar
        ("Benefits of exercise", "Advantages of working out", 1),
        ("Healthy diet plan", "Nutritional meal planning", 1),
        ("Mental health awareness", "Importance of mental wellbeing", 1),
        ("Yoga for beginners", "Starting yoga practice", 1),
        ("Sleep improvement tips", "How to sleep better?", 1),
        ("Stress management techniques", "Ways to reduce stress", 1),
        ("Meditation benefits", "Advantages of meditating", 1),
        ("Drink enough water", "Stay hydrated daily", 1),

        # Dissimilar pairs
        ("Healthy heart", "Heart shape drawing", 0),
        ("Blood pressure monitor", "Monitor computer screen", 0),
        ("Vitamin C benefits", "C programming language", 0),
        ("Dental care", "Car engine care", 0),
        ("Eye exercises", "Exercise equipment", 0)
    ]
    
    # Create more variations
    augmented_data = []
    for q1, q2, label in questions:
        augmented_data.append([q1, q2, label])
        # Add some variations
        if label == 1:
            augmented_data.append([q2, q1, label])  # Reverse order
    
    df = pd.DataFrame(augmented_data, columns=['question1', 'question2', 'is_duplicate'])
    return df

def extract_features(df):
    """Extract features from question pairs"""
    print("Extracting features...")
    
    # Preprocess questions
    df['q1_clean'] = df['question1'].apply(preprocess_text)
    df['q2_clean'] = df['question2'].apply(preprocess_text)
    
    # Basic text features
    df['q1_len'] = df['q1_clean'].apply(len)
    df['q2_len'] = df['q2_clean'].apply(len)
    df['len_diff'] = abs(df['q1_len'] - df['q2_len'])
    
    df['q1_word_count'] = df['q1_clean'].apply(lambda x: len(x.split()))
    df['q2_word_count'] = df['q2_clean'].apply(lambda x: len(x.split()))
    df['word_count_diff'] = abs(df['q1_word_count'] - df['q2_word_count'])
    
    # Common words feature
    df['common_words'] = df.apply(
        lambda row: len(set(row['q1_clean'].split()) & set(row['q2_clean'].split())), axis=1
    )
    
    # TF-IDF features

    # TF-IDF (FIT ON ALL QUESTIONS)
    tfidf = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words='english'
    )

    tfidf.fit(pd.concat([df['q1_clean'], df['q2_clean']]))

    # Transform questions
    q1_tfidf = tfidf.transform(df['q1_clean'])
    q2_tfidf = tfidf.transform(df['q2_clean'])

    # Pair-wise cosine similarity
    df['cosine_similarity'] = [
        cosine_similarity(q1_tfidf[i], q2_tfidf[i])[0][0]
        for i in range(len(df))
    ]

    
    # Prepare feature matrix
    feature_cols = ['q1_len', 'q2_len', 'len_diff', 
                    'q1_word_count', 'q2_word_count', 'word_count_diff',
                    'common_words', 'cosine_similarity']
    
    X = df[feature_cols]
    y = df['is_duplicate']
    
    return X, y, df, tfidf

def train_and_save_model():
    """Train the model and save it as .pkl file"""
    print("Training model...")
    
    # Create synthetic data
    df_synth = create_synthetic_data()
    df = pd.read_csv("questions.csv", nrows=30000)

    # keep only required columns
    df_csv = pd.read_csv("questions.csv", nrows=30000)
    df_csv = df_csv[['question1', 'question2', 'is_duplicate']]
    df_csv.dropna(inplace=True)
    df_csv['is_duplicate'] = df_csv['is_duplicate'].astype(int)

    #  Combine both
    df = pd.concat([df_synth, df_csv], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Final training rows:", len(df))
    
    print("CSV Loaded:", df.shape)

    
    # Extract features
    X, y, df_with_features, tfidf = extract_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}")
    
    # Save the model and vectorizer
    joblib.dump(model, 'models/duplicate_model.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    
    # Save feature columns for later use
    feature_cols = list(X.columns)
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    
    # Generate and save visualizations
    generate_visualizations(model, X_test, y_test, y_pred, df_with_features)
    
    print(f"\nModel saved as 'models/duplicate_model.pkl'")
    print(f"TF-IDF vectorizer saved as 'models/tfidf_vectorizer.pkl'")
    
    return accuracy, model, X_test, y_test, y_pred

def generate_visualizations(model, X_test, y_test, y_pred, df):
    """Generate and save visualization plots"""
    print("Generating visualizations...")
    
    # Create models directory if it doesn't exist
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Duplicate', 'Duplicate'],
                yticklabels=['Not Duplicate', 'Duplicate'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    
    # 2. Feature Importance
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    plt.close()
    
    # 3. Similarity Distribution
    plt.figure(figsize=(10, 6))
    
    # Get cosine similarity for duplicate and non-duplicate pairs
    duplicate_sims = df[df['is_duplicate'] == 1]['cosine_similarity']
    non_duplicate_sims = df[df['is_duplicate'] == 0]['cosine_similarity']
    
    plt.hist(duplicate_sims, alpha=0.7, label='Duplicate Pairs', bins=20)
    plt.hist(non_duplicate_sims, alpha=0.7, label='Non-Duplicate Pairs', bins=20)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution: Duplicate vs Non-Duplicate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/similarity_distribution.png')
    plt.close()
    
    print("Visualizations saved in 'models/' directory")

if __name__ == "__main__":
    print("="*50)
    print("Duplicate Questions Detection Model Training")
    print("="*50)
    
    accuracy, model, X_test, y_test, y_pred = train_and_save_model()
    
    # Print classification report
    print("\n" + "="*50)
    print("Classification Report:")
    print("="*50)
    print(classification_report(y_test, y_pred, 
                                target_names=['Not Duplicate', 'Duplicate']))
    
    print("\nTraining completed successfully!")
    print(f"Model files saved in 'models/' directory")