import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')  # ← THIS is important!



ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
""


# Load the pre-trained TF-IDF vectorizer
with open(r'C:\Users\kethavath Pragna\OneDrive\Desktop\email-spam-detection\email-spam-detection\vectorizer.pkl', 'rb') as vec_file:
    tfidf = pickle.load(vec_file)
with open(r'C:\Users\kethavath Pragna\OneDrive\Desktop\email-spam-detection\email-spam-detection\model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)



st.title("SMS Fraud Detection ")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess the input message
    transformed_sms = transform_text(input_sms)

    # Debug prints
    print("Transformed Message:", transformed_sms)

    # Vectorize the input message
    vector_input = tfidf.transform([transformed_sms])

    # Debug prints
    print("Vectorized Input:", vector_input)

    # Make predictions
    result = model.predict(vector_input)

    # Debug prints
    print("Prediction Result:", result)

    # Display the result
    if result[0] == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
# import pickle
# import streamlit as st

# # Load the vectorizer and model
# with open(r'C:\Users\sheshu\Desktop\Untitled Folder\email-spam-detection\vectorizer.pkl', 'rb') as vec_file:
#     tfidf = pickle.load(vec_file)
# with open(r'C:\Users\sheshu\Desktop\Untitled Folder\email-spam-detection\model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# # Check if they are loaded correctly
# try:
#     st.write("Testing model loading...")
#     model.predict(tfidf.transform(["test message"]))  # Test prediction to confirm loading
#     st.write("Model loaded successfully and ready for predictions.")
# except Exception as e:
#     st.write("Error loading model:", e)
