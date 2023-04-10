import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_iris


def number(wine_df1):
    col1, col2, col3 = st.columns(3)
    with col1:
        col1_1, col1_2 = st.columns(2)
        col1_1.header('최대 : ')
        col1_2.header(max(wine_df1))

    with col2:
        col2_1, col2_2 = st.columns(2)
        col2_1.header('최소 : ')
        col2_2.header(min(wine_df1))

    with col3:
        col3_1, col3_2 = st.columns(2)
        col3_1.header('평균 : ')
        col3_2.header(round(sum(wine_df1)/len(wine_df1), 2))

st.title('Streamlit & CNN vs TL')
tab1, tab2, tab3 = st.tabs(["Streamlit", "CNN vs TL", "Code"])

with tab1:
    st.subheader("Streamlit : 빅데이터와 머신러닝을 간단하게 배포할수 있는 파이썬(Python) 기반의 웹어플리케이션")

    st.subheader(" ")
    
    st.subheader("데이터 관측")
    
    iris = load_iris()
    data = iris['data']
    df_iris = pd.DataFrame(data, columns=iris['feature_names'])
    occupation = st.selectbox('보실 목록을 선택하세요.',list(df_iris.columns))
    if occupation:
        col_1, col_2 = st.columns(2)
        df_iris_1 = df_iris[occupation]

        with col_1:
            start = st.slider('시작 점', 0, len(df_iris_1))

        with col_2:
            end = st.slider('끝 점', start, len(df_iris_1))

        if start < end and end != 0:
            df_iris_1 = df_iris[occupation][start:end+1]
            st.line_chart(df_iris_1)
            number(df_iris_1)
        else:
            df_iris_1 = df_iris[occupation][start:]
            st.line_chart(df_iris_1)
            number(df_iris_1)
    
    st.subheader(" ")
    
    st.subheader("데이터 예측")
    
    text1, text2 = st.columns(2)
    with text1:
        x1 = st.text_input('sepal length (cm)', '')
    with text2:
        x2 = st.text_input('sepal width (cm)', '')
        
    text3, text4 = st.columns(2)   
    with text3:
        x3 = st.text_input('petal length (cm)', '')
    with text4:
        x4 = st.text_input('petal width (cm)', '')
        
    if len(str(x1)) > 0 and len(str(x2)) > 0 and len(str(x3)) > 0 and len(str(x4)) > 0: 
        pred_df= pd.DataFrame([[x1, x2, x3, x4]], columns=iris['feature_names'])
        clf_from_joblib = joblib.load('classification_model.pkl') 
        pred = clf_from_joblib.predict(pred_df)
        if pred[0] == 0:
            st.write('The predicted species is setosa')
        elif pred[0] == 1:
            st.write('The predicted species is versicolor')
        elif pred[0] == 2:
            st.write('The predicted species is virginica')
    
with tab2:
    st.subheader("CNN VS TL")
    
    st.subheader(" ")
    
    cnn, tl = st.columns(2)
    
    with cnn:
        st.write('cnn')
        
    with tl:
        st.write('tl')
    
with tab3:
    st.header("code")