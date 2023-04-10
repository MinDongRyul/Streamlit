import streamlit as st

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
    st.subheader("빅데이터와 머신러닝을 간단하게 배포할수 있는 파이썬(Python) 기반의 웹어플리케이션")

    st.subheader(" ")
    
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