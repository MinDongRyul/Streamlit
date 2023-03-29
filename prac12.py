import streamlit as st
import datetime
from PIL import Image
from sklearn.datasets import load_iris
import pandas as pd

b1 = st.sidebar.button('이미지 분류')
b2 = st.sidebar.button('텍스트 분류')
b3 = st.sidebar.button('Speech2Text')
b4 = st.sidebar.button('시계열 데이터 예측')

if b1:
    st.header('이미지 분류')
    col1, col2, col3, col4, col5 = st.columns([10, 3, 5, 3, 6])
    with col1:
        col1_1, col1_2 = st.columns([2, 5])
        col1_1.write('Class1')
        col1_2.button('rename')

        uploaded_file = st.file_uploader("이미지 샘플추가:")
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            st.write(bytes_data)

            # To convert to a string based IO:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            st.write(stringio)

            # To read file as string:
            string_data = stringio.read()
            st.write(string_data)

        st.write(' ')
        # 새로운 줄 시작
        col2_1, col2_2 = st.columns([2, 5])
        col2_1.write('Class2')
        col2_2.button('rename2')

        uploaded_file2 = st.file_uploader("이미지 샘플추가2:")
        if uploaded_file2 is not None:
            # To read file as bytes:
            bytes_data = uploaded_file2.getvalue()
            st.write(bytes_data)

            # To convert to a string based IO:
            stringio = StringIO(uploaded_file2.getvalue().decode("utf-8"))
            st.write(stringio)

            # To read file as string:
            string_data = stringio.read()
            st.write(string_data)

    with col2:
        for _ in range(6):
            st.write('')
        st.write('  ---→')
        for _ in range(6):
            st.write('')

    with col3:
        st.write('학습')
        st.button('모델 학습시키기')
        with st.expander('고급'):
            st.write('고급 기술')

    with col4:
        for _ in range(6):
            st.write('')
        st.write('  ---→')
        for _ in range(6):
            st.write('')

    with col5:
        col5_1, col5_2 = st.columns([2, 5])
        col5_1.write('미리보기')
        col5_2.button('모델 내보내기')

        st.write('여기서 모델을 미리 확인하려면 먼저 왼쪽에서 모델을 학습시켜야 합니다.')

elif b2:
    
    st.header('텍스트 분류')

    col1, col2 = st.columns(2)

    with col1:
        test = st.text_area('분류할 구문을 넣어주세요','구문 적기')
        st.write('')

    with col2:
        st.write('구문텍스트의 감정 분류')

    st.button('Test')

elif b3:
    st.header('Speech2Text / Text2Speech')

    col1, col2 = st.columns(2)

    with col1:
        st.file_uploader('Upload a mp3 file')

    with col2:
        st.write('Download a Speech File')

    st.text_area('Text to Speech / Speech to Text')

elif b4:
    st.header('시계열 데이터 예측')

    col1, col2 = st.columns(2)

    with col1:

        uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            st.write(bytes_data)
    
    with col2:
        col1_1, col1_2, col1_3 = st.columns(3)
        p_1 = col1_2.date_input('훈련 데이터 지정',datetime.date(1991, 1, 1))
        p_2 = col1_3.date_input('',datetime.date(2009, 12, 31))

        col2_1, col2_2, col2_3 = st.columns(3)
        v_1 = col2_2.date_input('검증 데이터 지정',datetime.date(2010, 1, 1))
        v_2 = col2_3.date_input('',datetime.date(2014, 12, 31))

    iris = load_iris()
    data = iris['data']
    df_iris = pd.DataFrame(data, columns=iris['feature_names'])
    df_iris2 = df_iris['petal length (cm)'] # 임시시
    st.line_chart(df_iris2)