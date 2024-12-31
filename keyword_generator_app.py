import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure the correct resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean and tokenize the input text
def clean_text(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())  # Lowercased and tokenized
    # Remove stopwords (common words like "is", "and", "the", etc.)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

# Function to extract keywords from text using TF-IDF
def extract_keywords(text, num_keywords=10):
    # Clean the input text
    clean_input_text = clean_text(text)
    
    # Create TF-IDF vectorizer to extract terms and calculate their importance
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
    
    # Fit and transform the text to get TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform([clean_input_text])
    
    # Get feature names (terms/keywords)
    keywords = vectorizer.get_feature_names_out()
    
    # Return the list of keywords
    return keywords

# Streamlit user interface
def main():
    # Set the page config
    st.set_page_config(page_title="Keyword Extraction", layout="wide")
    
    # Custom CSS to add margin and styling (removing lines and adjusting margins)
    st.markdown("""
        <style>
            .main {
                padding-left: 40px;
                padding-right: 40px;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 10px 2px;
                cursor: pointer;
                border-radius: 5px;
            }
            .stTextArea>textarea {
                font-size: 16px;
            }
            .footer {
                text-align: center;
                font-size: 12px;
                color: grey;
                padding-top: 20px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Title of the web app with better formatting
    st.title("Text to Keyword Extraction Tool")
    st.markdown("""
        The app will extract the most relevant keywords using TF-IDF.
        You can also adjust how many keywords you want to extract. 
        This is a useful tool for summarizing key ideas from any given text.
    """)
    
    # Input box for the user to input text with a clear description
    input_text = st.text_area("Enter your text here:", height=200, help="Input the text from which you want to extract keywords.")
    
    # Add a description above the slider to guide the user
    st.markdown("""
    ### Adjust the number of keywords to extract
    Select a number between 1 and 30 to extract the top keywords.
    """)

    # Number of keywords to extract (set default to 5) with a maximum of 30
    num_keywords = st.slider("Number of Keywords to Extract", min_value=1, max_value=30, value=5)

    # Button to trigger keyword extraction with a custom style
    extract_button = st.button("Extract Keywords", help="Click to extract keywords from the provided text")
    
    # When the button is pressed, extract and display the keywords
    if extract_button:
        if input_text:
            # Extract keywords
            keywords = extract_keywords(input_text, num_keywords)

            # Display the extracted keywords in a nice format
            st.write(f"**Top {num_keywords} Keywords:**")
            st.markdown('<div style="word-wrap: break-word;">' + ', '.join(keywords) + '</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to extract keywords.")
    
    # Footer Section for more information or credits
    st.markdown("""
        <div class="footer">
            Created with ❤️ by SASA. 
            Feel free to use and modify it as you wish.
        </div>
    """, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()
