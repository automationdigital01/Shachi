import time
import streamlit as st
import os
import io
import sys
import PyPDF2
import docx
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import json
import random
import pandas as pd
import string
from langchain.agents import create_pandas_dataframe_agent
import pickle
from blob_functions import *


with open("openai_credentials.json", 'r') as f:
    creds = json.load(f)
openai.api_type = creds['api_type']
openai.api_base = creds['api_base']
openai.api_version = creds['api_version']
openai.api_key = creds['api_key']

os.environ['OPENAI_API_TYPE'] = creds['api_type']
os.environ['OPENAI_API_BASE'] = creds['api_base']
os.environ['OPENAI_API_VERSION'] = creds['api_version']
os.environ['OPENAI_API_KEY'] = creds['api_key']

# davinci_llm = AzureOpenAI(deployment_name='substainability-code-davinci-002')
chat_llm = AzureChatOpenAI(
    deployment_name='sustainability-gpt-35-turbo', temperature=0)
# chain = load_qa_chain(chat_llm, chain_type='refine')

#  Extract Text from PDF for Document Analysis


def text_extract_from_pdf(file_path):
    pdf = PyPDF2.PdfReader(file_path)
    extracted_text = ""
    for page in pdf.pages:
        extracted_text += page.extract_text()
    return extracted_text

#  Extract Text from DOC/DOCX for Document Analysis


def text_extract_from_doc(file_path):
    doc = docx.Document(file_path)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)


def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        st.write("Time taken:", end_time - start_time, "seconds")
        return result
    return wrapper

# Upload the documents pkl file on blob


def upload_documents_to_blob(file):
    # data in 2d list format where we have text and filename
    try:

        filename = file[0]
        text = file[1]
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=3800,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.create_documents([text])
        chunks_len = len(chunks)
        count = (chunks_len)//15
        vectordb = FAISS.from_documents(chunks[:15], embedding=OpenAIEmbeddings(
            deployment="sustainability-text-embedding-ada-002"))
        for i in range(count-1):
            vectordb.add_documents(chunks[15*i:15*(i+1)])
        if count > 2:
            vectordb.add_documents(chunks[count*15:])

        # text_file= f"{filename}.txt"
        filename = filename.split(".")[0]
        knowledge_base_file = f"{filename}.pkl"

        # creating text file with file name
        # with open(text_file,"w") as file:
        #     file.write(text)
        # uploading the file to blob
        # blob_client= container_client.get_blob_client(text_file)
        # with open(text_file,"rb") as data:
        #     blob_client.upload_blob(data, overwrite= True)

        upload_succesfull = upload_blob(knowledge_base_file, vectordb)
        return upload_succesfull
    except Exception as e:
        return str(e)

#  Upload documents for future documents analysis


@calculate_time
@st.cache_resource
def upload_documents(file_list):
    try:
        # documents = []
        file_names = []
        for file in file_list:
            file_name = str(file.name).split('.')[0]
            file_names.append(file_name)
            print(file_name)
            if not check_blob(f"{file_name}.pkl"):
                # print("file not in blob")
                if file.name.endswith(".pdf"):
                    extracted_text = text_extract_from_pdf(file)
                    extracted_text = "The document name is " + \
                        str(file.name)+":" + extracted_text
                    upload_documents_to_blob([str(file.name), extracted_text])

                elif file.name.endswith(".docx") or file.name.endswith(".doc"):
                    extracted_text = text_extract_from_doc(file)
                    extracted_text = "The document name is " + \
                        str(file.name)+":" + extracted_text
                    upload_documents_to_blob([str(file.name), extracted_text])
                elif file.name.endswith(".txt"):
                    with open(file, "rb") as f:
                        extracted_text = f.read()
                    extracted_text = "The document name is " + \
                        str(file.name)+":" + extracted_text
                    upload_documents_to_blob([str(file.name), extracted_text])

        knowledge_base_files = []
        for filename in file_names:
            knowledge_base_file = f"{filename}.pkl"
            downloaded_file_path = download_blob(knowledge_base_file)
            with open(downloaded_file_path, "rb") as f:
                knowledge_base = pickle.load(f)
            knowledge_base_files.append(knowledge_base)

        model = knowledge_base_files[0]
        if len(knowledge_base_files) > 1:
            for i in range(1, len(knowledge_base_files)):
                model.merge_from(knowledge_base_files[i])
        return model
    except Exception as e:
        return str(e)

# After uploading Documents, you can ask questions based on the documents


def document_extract(query, model):
    try:
        # Running FAISS similarity Search algo on the prompt and document embeddings
        # docs = vectordb.similarity_search(prompt)
        prompt = f"""Query: {query}
        Follow the below instructions carefully and provide the result.
        1. If the employee asks for bullet points or milestones, provide the result in bullet points format. Otherwise provide in paragraph format. 
        2. Do Provide in one format only. The default length of answers is 150 words. Unless the employee explicitly mentions length of the output in sentences or words. Use the default length. 
        3. If the employee asks to explain in detailed way, then the result should be in a detailed way.
        4. If the employee asks for the comparison or any other actions where we need to compare two documents, then the result should be in such a way that it analyzes the given documents thoroughly and give the result in a detailed way.
        5. The result should be in such a way that it is relevant and accurate to the query . Do not provide any irrelevant answers.
        6. The headings of the comparison or action points should be in a concise."""
        qa = RetrievalQA.from_chain_type(
            llm=chat_llm, chain_type="stuff", retriever=model.as_retriever())

        with get_openai_callback() as cb:
            response = qa.run(prompt)
            print(cb)
        st.session_state['conversation_history'].append(
            {'user': query, 'bot': response})
        return response
    except Exception as e:
        return str(e)

# Function to generate graphs using CSV file


def csv_plot_creation(query):
    random_word = ''.join(random.choices(string.ascii_lowercase, k=6))

    file_path = f"graph/{random_word}.png"
    prompt2 = f"{query} and add labels if necessary. Don't concat or merge between two different dataframes. Perform only on one dataframe based on the column names. Labels should be seen clearly and labels should not 'overlap'. In axes mention Add Chart Title and Title for both the axes. Save the file in path graph/{random_word}.png"
    prompt = f"{query} and add labels if necessary. Labels should be seen clearly and labels should not 'overlap'. In axes mention Add Chart Title and Title for both the axes. Save the file in path graph/{random_word}.png"
    # prompts = [prompt,prompt2]
    try:
        if graph_file_type:  # Single dataframe
            csv_agent_graph.run(prompt)
        else:  # Multiple dataframe
            csv_agent_graph.run(prompt2)
        # with open(file_path, 'rb') as image_file:
        #     encoded_string = base64.b64encode(image_file.read())
        st.session_state['conversation_history'].append(
            {'user': query, 'bot': file_path})
        return file_path
    except Exception as e:
        return file_path


csv_agent_qna = ""
csv_agent_graph = ""


def csv_file_reader(file_path):
    dfs = []
    file_type = True  # True if CSV else XLSX
    if file_path.name.endswith('.csv'):
        df = pd.read_csv(file_path)
        dfs.append(df)
    elif file_path.name.endswith('.xlsx'):
        # file_type =False
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        for sheet_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            dfs.append(df)
    if len(dfs) > 1:
        file_type = False
    return dfs, file_type

# Function to run simple query on csv File


def csv_qna(query):
    try:
        prompt2 = ("""
                  Don't concat or merge between two different dataframes. Perform only on one dataframe based on the column names.
                  Provide me the data for the given query from the provided dataframe as:-
                  "Data": " "
                  Strictly provide the data in string format only
                  Query:-
                  """
                   + query
                   )
        prompt = ("""
                  Provide me the data for the given query from the provided dataframe as:-
                  "Data": " "
                  Strictly provide the data in string format only
                  Query:-
                  """
                  + query
                  )

        # prompt = f"Extract the data required in Dataframe format to answer this query. Query: {query}"
        if qna_file_type:  # Single dataframe
            result = str(csv_agent_qna.run(prompt))
        else:  # Multiple dataframe
            result = str(csv_agent_qna.run(prompt2))
        # st.write(str(type(result)))
        # return str(result)
        prompt1 = f"""
                    I am having information related to the data ,Using the below extracted data provide the answer in proper format for this query {query}:-
                    Strictly return the answer as string in proper sentence format.
                    Data:-
                """
        message = [
            {"role": "system", "content": prompt1},
            {"role": "user", "content": result}
        ]
        response = openai.ChatCompletion.create(
            engine="sustainability-gpt-35-turbo",
            messages=message)
        res = response.choices[0].message.content
        st.session_state['conversation_history'].append(
            {'user': query, 'bot': res})
        return res
    except Exception as e:
        return str(e)


qna_file_type = True
graph_file_type = True
# Function to read CSV file for qna - Text(Excel) Analysis


def csv_reader_qna(csv_file_path):
    try:
        # df = pd.read_csv(csv_file_path)
        global qna_file_type
        dfs, qna_file_type = csv_file_reader(csv_file_path)
        global csv_agent_qna
        if qna_file_type:  # Single dataframe
            # csv_agent_qna = create_pandas_dataframe_agent(AzureOpenAI(temperature=0, deployment_name='substainability-code-davinci-002'),dfs[0],verbose=True)
            csv_agent_qna = create_pandas_dataframe_agent(
                chat_llm, dfs[0], verbose=False)
        else:  # Multiple dataframe
            # csv_agent_qna = create_pandas_dataframe_agent(AzureOpenAI(temperature=0, deployment_name='substainability-code-davinci-002'),dfs,verbose=False)
            csv_agent_qna = create_pandas_dataframe_agent(
                chat_llm, dfs, verbose=False)

        return "CSV file uploaded successfully"
    except Exception as e:
        return str(e)

# Function to read CSV file for Graph generation


def csv_reader_graph(csv_file_path):
    try:
        global csv_agent_graph
        global graph_file_type
        dfs, graph_file_type = csv_file_reader(csv_file_path)
        if graph_file_type:  # Single dataframe
            csv_agent_graph = create_pandas_dataframe_agent(AzureOpenAI(
                temperature=0, deployment_name='substainability-code-davinci-002'), dfs[0], verbose=True)
        else:  # Multiple dataframe
            csv_agent_graph = create_pandas_dataframe_agent(AzureOpenAI(
                temperature=0, deployment_name='substainability-code-davinci-002'), dfs, verbose=True)

        return "CSV file uploaded successfully"
    except Exception as e:
        return str(e)
    # *********************************************************************


def main():

    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []

    header_container = st.container()

    with header_container:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.write("")
        with col2:
            st.write("")
            # st.markdown(
            #     "<span style='color: #1E90FF'>Smart</span> "   " <span style='color: #92D293'>Sustainability</span>", unsafe_allow_html=True)

            st.markdown("<h1 style='line-height: 1.5; margin-bottom: 0; font-size: 38px;'> <span style='color: #1E90FF'>Smart</span> <span style='color: #92D293'>Sustainability</span></h1>", unsafe_allow_html=True)

        with col3:
            st.image("USER LOGin.png", width=70)
            username = os.getlogin()
            st.markdown(
                f"<span style='color: #1E90FF; margin-left: 10px;'>Welcome {username}!</span>", unsafe_allow_html=True)
            st.markdown(
                "<style>div.stSidebar {display: flex; justify-content: flex-end;}</style>", unsafe_allow_html=True)

    st.markdown("""
        <style>
            [data-testid=stSidebar] {
                background: linear-gradient(to bottom, #1E90FF, #92D293);
            }
        </style>
        """, unsafe_allow_html=True)

    st.sidebar.image("TECHNIP_ENERGIES_LOGO.png", width=100)

    analysis_type = st.sidebar.radio("Select Analysis Type", [
                                     "Multiple Document Analysis", "Text(Excel)Analysis", "Graph Analysis"])

    if analysis_type == ("Multiple Document Analysis"):
        uploaded_files = st.file_uploader("Upload Documents", type=[
                                          "pdf", "doc", "docx"], accept_multiple_files=True)
        if uploaded_files:
            model = upload_documents(uploaded_files)
            st.success("File Uploaded Successfully")
            # query = st.text_input(
            #    "", placeholder="How may I help you?", style={'height': '100px'})

            query = st.text_area(
                "", placeholder="How may I help you?", height=100, max_chars=None, key=None)

            submit_button = st.button("Submit")
            if query and submit_button:
                results = document_extract(query, model)
                st.write("Response:")
                st.write(results)

    elif analysis_type == ("Text(Excel)Analysis"):
        uploaded_file = st.file_uploader(
            "Upload CSV/XLSX", type=["csv", "xlsx"])
        if uploaded_file is not None:
            val = csv_reader_qna(uploaded_file)
            st.success(val)
            query = st.text_area(
                "", placeholder="How may I help you?", height=100, max_chars=None, key=None)
            submit_button = st.button("Submit")
            if query and submit_button:
                results = csv_qna(query)
                st.write("Response:")
                st.write(results)

    elif analysis_type == ("Graph Analysis"):
        uploaded_file = st.file_uploader(
            "Upload CSV/XLSX", type=["csv", "xlsx"])
        if uploaded_file is not None:
            val = csv_reader_graph(uploaded_file)
            st.success(val)
            query = st.text_area(
                "", placeholder="How may I help you?", height=100, max_chars=None, key=None)
            submit_button = st.button("Submit")
            if query and submit_button:
                file_path = csv_plot_creation(query)
                try:
                    st.image(file_path)
                except:
                    st.write(
                        "Please be more specific about the type of graph you want to create.")

    # Display conversation history
    st.write('Conversation History:')
    for item in st.session_state['conversation_history']:
        st.write(f"User: {item['user']}")
        st.write(f"Bot: {item['bot']}")
        st.write('---')


# Run the Streamlit app
if __name__ == "__main__":
    main()
