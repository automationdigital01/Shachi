import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
import joblib as jb


rf_loaded=jb.load('feedback_model.pkl')
v=jb.load('tfidf_vectorizer.pkl')
sentiment_pipeline=jb.load('sentiment.pkl')



def predict_category(input_text):
    input_text = ''.join([char for char in input_text if char not in string.punctuation])
    input_text = input_text.lower()
    tokens = input_text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemm = PorterStemmer()
    stemm_tokens = [stemm.stem(token) for token in tokens]
    cleaned_text = ' '.join(stemm_tokens)
    input_tfidf = v.transform([cleaned_text])
    y_predict = rf_loaded.predict(input_tfidf)
    category_dict = {
        0: 'team',
        1: 'workload',
        2: 'culture',
        3: 'opportunities',
        4: 'values',
        5: 'management',
        6: 'training',
        7: 'compensation',
        8: 'office'
    }
    predicted_category = category_dict[y_predict[0]]
    return predicted_category

def sentiment_category(text):
    result=sentiment_pipeline(text)[0]
    return result['label']

def app():

    st.set_page_config(
    page_title="Feedback Categorization"
    )


    st.image('logo.png', width=200)
    st.title("Feedback Categorization App")
    st.write("\n\n")


    input_text=st.text_input('Enter your feedback here')
    if st.button('Categorize') and input_text:
        prediction=predict_category(input_text)
        sentiment=sentiment_category(input_text)
        st.success('Thank You for your feedback')
        st.write('The feedback will be saved in category :',prediction.upper())
        st.write('The sentiment of the feedback is : ', sentiment.upper())


    st.write('### OR')


    file=st.file_uploader('Please upload your feedback file here',type=['xlsx'])
    if file is not None:
        st.success('File is uploaded successfully')
        df= pd.read_excel(file)
        with st.spinner('Categorizing feedbacks...'):
            df['feedback_category']=df['Feedback'].apply(predict_category)
            df['Sentiment']=df['Feedback'].apply(sentiment_category)
            df.to_excel('Predicted_feedbacks.xlsx',index=False)
        st.info('Feedback categorization complete and saved as Predicted_feedbacks.xlsx !')
        st.dataframe(df)
if __name__== '__main__':
    app()
