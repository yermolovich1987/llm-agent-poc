import os

import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader
)

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
import tempfile

# PERSIST_DIRECTORY = 'docs/chroma/'
PERSIST_DIRECTORY = 'docs/chroma_cars/'
RETRIEVED_DOCUMENTS_COUNT = 5
MODEL_NAME = 'gpt-3.5-turbo'

PDF_FOLDER_PATH = 'dataset/pdf_price_cz'


def main():
    # render_main_page()

    pdf_docs = load_pdf_documents()
    print(f"$$$   Loaded pages: {pdf_docs}")


def load_pdf_documents():
    documents = []
    for file in os.listdir(PDF_FOLDER_PATH):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(PDF_FOLDER_PATH, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load_and_split())
    return documents


def render_main_page():
    st.title("Product recommendation demo")
    # embedding = OpenAIEmbeddings()
    print(f"### Started initialization of the embeddings")
    embedding = HuggingFaceEmbeddings()
    print(f"### Started initialization of the Vector DB")
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
    print(f"### Finished initialization of the Vector DB")
    with st.sidebar:
        f"Product vector DB initially contains {vectordb._collection.count()} documents"
    uploaded_file = st.sidebar.file_uploader("Upload additional data from file in PDF format.",
                                             type=["pdf"])
    if uploaded_file:
        # use tempfile because CSVLoader only accepts a file_path
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # loader = construct_loader(tmp_file_path)
        loader = PyPDFLoader(tmp_file_path)

        with st.spinner("Loading file with products..."):
            data = loader.load()
            st.write(data)
            vectordb.add_documents(data)

        with st.sidebar:
            st.success(f"Uploaded additional {len(data)} documents to the vector database")
    # === Start the logic of retrieval chain
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVED_DOCUMENTS_COUNT})
    # create a chatbot chain. Memory is managed externally.
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=MODEL_NAME, temperature=0),
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        verbose=True
    )
    # === Draw main con
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Please provide you preferences about products"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
    # container for the chat history
    response_container = st.container()
    # container for the user's text input
    container = st.container()
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input, chain)

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")


def construct_loader(uploaded_file):
    # use tempfile because loaders only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    if uploaded_file.type == 'application/json':
        loader = JSONLoader(file_path=tmp_file_path, jq_schema='.[]', json_lines=False, text_content=False)
    else:
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ';'})
    return loader


def conversational_chat(query, chain):
    result = chain({"question": query,
                    "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))

    return result["answer"]


if __name__ == "__main__":
    main()
