import streamlit as st
import datetime
from PIL import Image
from sklearn.datasets import load_iris
import pandas as pd

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

option = st.sidebar.selectbox('',('이미지 분류', '텍스트 분류', 'Speech2Text', '시계열 데이터 예측'))

if option == '이미지 분류':
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
        st.button('모델 학습')
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

elif option == '텍스트 분류':
    
    st.header('텍스트 분류')

    col1, col2 = st.columns(2)

    with col1:
        test = st.text_area('분류할 구문을 넣어주세요','')
        col3, col4 = st.columns(2)
        
        with col3:
            bt = st.button('test')
            if len(test) == 0 and bt:
                st.write('텍스트를 적어주세요')
            elif len(test) != 0 and bt:
                st.write('분류중입니다.')

    with col2:
        st.write('구문텍스트의 감정 분류')

elif option == 'Speech2Text':
    st.header('Speech2Text')

    col1, col2 = st.columns(2)

    with col1:
        st.file_uploader('Upload a mp3 file')
    
    testfile = '' # 스피치로 받은 파일을 텍스트로 변환후 붙혀주기  
    if len(testfile) == 0:
        st.text_area('Speech to Text')
    else:
        st.text_area('Speech to Text')
        
elif option == '시계열 데이터 예측':
    st.header('시계열 데이터 예측')

    col1, col2 = st.columns(2)

    with col1:

        uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)

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
    
    if uploaded_files:
        data_df = pd.read_csv(uploaded_files)
        occupation = st.selectbox('보실 목록을 선택하세요.',df_iris['feature_names'])
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