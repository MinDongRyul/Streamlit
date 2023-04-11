import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import load_model
from PIL import Image

def number(data_df):
    col1, col2, col3 = st.columns(3)
    with col1:
        col1_1, col1_2 = st.columns(2)
        col1_1.header('최대 : ')
        col1_2.header(max(data_df))

    with col2:
        col2_1, col2_2 = st.columns(2)
        col2_1.header('최소 : ')
        col2_2.header(min(data_df))

    with col3:
        col3_1, col3_2 = st.columns(2)
        col3_1.header('평균 : ')
        col3_2.header(round(sum(data_df)/len(data_df), 2))

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
        data_df = df_iris[occupation]

        with col_1:
            start = st.slider('시작 점', 0, len(data_df))

        with col_2:
            end = st.slider('끝 점', start, len(data_df))

        if start < end and end != 0:
            data_df = df_iris[occupation][start:end+1]
            st.line_chart(data_df)
            number(data_df)
        else:
            df_iris_1 = df_iris[occupation][start:]
            st.line_chart(data_df)
            number(data_df)
    
    st.subheader(" ")
    
    st.subheader("데이터 예측")
    st.write('sepal : 꽃받침, petal : 꽃잎')
    
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
        proba = clf_from_joblib.predict_proba(pred_df)
        if pred[0] == 0:
            st.write('The predicted species with a probability of ' + str(round(proba[0][0] * 100, 2)) + '% is setosa')
        elif pred[0] == 1:
            st.write('The predicted species with a probability of ' + str(round(proba[0][1] * 100, 2)) + '% is versicolor')
        elif pred[0] == 2:
            st.write('The predicted species with a probability of ' + str(round(proba[0][2] * 100, 2)) + '% is virginica')
    
with tab2:
    st.subheader("CNN 과 TL 비교")
    
    st.subheader(" ")
    
    st.caption('import')
    
    import_code = '''
import tensorflow as tf
from keras import layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 모델의 훈련 설정하기
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
import numpy as np'''

    st.code(import_code, 'python')
      
    st.caption('CNN 네트워크 구축')
    cnn_code = '''model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
# Conv2D : 이미지는 2D, 동영상은 3D
# 필터 : 32
# kernel size : 3
# activation : relu함수를 활성화 함수로 사용
# input_shape : 150, 150, 3(너비, 높이, 채널(여기서는 컬러이기에 3, 흑백이면 1))
# (3, 3) -> kernel_size ([3, 3] or 3, (3,3) 가능)
model.add(layers.MaxPooling2D((2,2))) # 2x2로 pooling
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten()) # 1차원 벡터로 나열하기 위함
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))'''
    st.code(cnn_code, 'python')
        
    st.caption('TL 네트워크 구축')
    tl_code = '''base_model = MobileNet(weights='imagenet', include_top = False, input_shape=(150, 150, 3))
# MobileNet : Pre-trained model
# include_top = False : 특징 추출기로만사용 
model = Sequential()
model.add(base_model)
model.add(Flatten())
# user-defined classifier
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))'''
    st.code(tl_code, 'python')
    
    st.caption('같은 부분')
    
    same_code = '''model.compile(optimizer=optimizers.Adam(learning_rate=2e-5), 
              # learning_rate : learnint rate 설정
              loss='binary_crossentropy',
              metrics=['accuracy'])# 최적화, 손실함수, 성능평가지표
              
# ImageDataGenerator를 사용하여 디렉토리에서 이미지 읽기
train_datagen = ImageDataGenerator(rescale=1./255) # -> 픽셀 값(0~255)의 스케일을 [0, 1]로 조정
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, # 타겟 디렉토리
    target_size=(150,150),
    batch_size=40,
    class_mode='binary'
)

val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=40,
    class_mode='binary'
)

# 배치 제너레이터를 사용하여 모델 훈련하기
history = model.fit(
    train_generator, 
    steps_per_epoch=100,
    epochs=30, 
    validation_data= val_generator,
    validation_steps=50)
    
# 각 모델 저장
model.save('./cats_and_dogs_small_[NAME].h5')

# 훈련의 정확도와 손실 그래프 그리기
def history_plot(history_dict):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = np.arange(1, len(acc)+1)
    plt.plot(epochs, acc,'bo', label='train')
    plt.plot(epochs, val_acc, label='val')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(epochs, loss, 'bo', label='train')
    plt.plot(epochs, val_loss, label='val')
    plt.title('Loss')
    plt.legend()
    plt.show()

history_plot(history.history)

# test데이터를 통한 검증
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=40,
        class_mode='binary')

test_loss, test_acc = model.evaluate(test_generator, steps=50)
print('test acc:', test_acc)
    ''' 
    st.code(same_code, 'python')
    
    cnn_2, tl_2 = st.columns(2)
    
    with cnn_2:
        st.caption('CNN')
        cnn_acc = Image.open('cnn_acc.png')
        st.image(cnn_acc, caption='CNN accuracy')
        st.write(' ')
        cnn_loss = Image.open('cnn_loss.png')
        st.image(cnn_loss, caption='CNN loss')
        st.write('총 실행 시간 : 16분 49초')
        st.write('test data acc : 77.4%')
#         st.write(' ')
#         cnn_test_acc = Image.open('cnn_test_acc.png')
#         st.image(cnn_test_acc, caption='CNN test data accuracy')
        
    with tl_2:
        st.caption('TL')
        tl_acc = Image.open('tl_acc.png')
        st.image(tl_acc, caption='TL accuracy')
        st.write(' ')
        tl_loss = Image.open('tl_loss.png')
        st.image(tl_loss, caption='TL loss')
        st.write('총 실행 시간 : 37분 14초')
        st.write('test data acc : 94.5%')
#         st.write(' ')
#         tl_test_acc = Image.open('tl_test_acc.png')
#         st.image(tl_test_acc, caption='TL test data accuracy')
    
with tab3:
    code = '''
    print('hello python')'''
    st.code(code, language='python')