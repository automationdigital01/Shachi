import nltk
nltk.download('stopwords')
nltk.download('punkt')
import sklearn
import streamlit as st
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import PorterStemmer
import joblib as jb
import sklearn
from transformers import pipeline


sentiment_pipeline=pipeline('sentiment-analysis')

rf_loaded=jb.load('model/feedback_category_rf/feedback_model_updated.pkl')
v=jb.load('model/feedback_category_rf/tfidf_vectorizer_updated.pkl')



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


    st.image('model/feedback_category_rf/logo.png', width=200)
    st.title("Feedback Categorization App")
    st.write("\n\n")

    #text input
    input_text=st.text_input('Enter your feedback here')
    if st.button('Categorize') and input_text:
        prediction=predict_category(input_text)
        sentiment=sentiment_category(input_text)
        st.success('Thank You for your feedback')
        st.write('The feedback will be saved in category :',prediction.upper())
        st.write('The sentiment of the feedback is : ', sentiment.upper())


    st.write('### OR')

    #file uploader
    file=st.file_uploader('Please upload your feedback file here',type=['xlsx'])
    if file is not None:
        st.success('File is uploaded successfully')
        df= pd.read_excel(file)
        with st.spinner('Categorizing feedbacks...'):
            df['feedback_category']=df['Feedback'].apply(predict_category)
            df['Sentiment']=df['Feedback'].apply(sentiment_category)
            df.to_excel('Predicted_feedbacks.xlsx',index=False)
            st.toast('Feedback Categorization is completed')
        with open('Predicted_feedbacks.xlsx','rb') as file:
            click=st.download_button('Download Output File',file,file_name='Predicted_feedbacks_output.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        if click:
            st.toast('Predicted_feedbacks_output.xlsx is downloaded')
        st.dataframe(df)


if __name__== '__main__':
    app()
