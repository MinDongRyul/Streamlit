#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import datetime
import pandas as pd

from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_option_menu import option_menu
from sklearn.datasets import load_wine

wine = load_wine()
data = wine['data']
wine_df = pd.DataFrame(data, columns=wine['feature_names'])

with st.form('my_form_identifier'):    
    st.form_submit_button('Submit to me')
st.container()
# st.columns(spec)
col1, col2 = st.columns(2)
col1.subheader('Columnisation')
# st.expander('Expander')
with st.expander('Expand'):
    st.write('sqwersasd')
    
def asd(wine_df1):
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
    
occupation = st.selectbox('보실 목록을 선택하세요.',wine['feature_names'])
if occupation:
    col_1, col_2 = st.columns(2)
    wine_df1 = wine_df[occupation]
    
    with col_1:
        start = st.slider('시작 점', 0, len(wine_df1))
        
    with col_2:
        end = st.slider('끝 점', start, len(wine_df1))
    
    if start < end and end != 0:
        wine_df1 = wine_df[occupation][start:end+1]
        st.line_chart(wine_df1)
        asd(wine_df1)
    else:
        wine_df1 = wine_df[occupation][start:]
        st.line_chart(wine_df1)
        asd(wine_df1)
        
    
#     p_1 = st.date_input('훈련 데이터 지정',datetime.date(1991, 1, 1))
#     if p_1:
#         st.write(p_1)

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Settings'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    selected

# 2. horizontal menu
# selected2 = option_menu(None, wine['feature_names'], default_index=0, orientation="horizontal")

# 3. CSS style definitions
# selected3 = option_menu(None, ["Home", "Upload",  "Tasks", 'Settings'], 
#     icons=['house', 'cloud-upload', "list-task", 'gear'], 
#     menu_icon="cast", default_index=0, orientation="horizontal",
#     styles={
#         "container": {"padding": "0!important", "background-color": "#fafafa"},
#         "icon": {"color": "orange", "font-size": "25px"}, 
#         "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
#         "nav-link-selected": {"background-color": "green"},
#     }
# )


test = st.text_area('분류할 구문을 넣어주세요(Ctrl + Enter)','')
col1, col2 = st.columns(2)
with col1:
    bt = st.button('test')
    if len(test) == 0 and bt:
        st.write('텍스트를 적어주세요')
    elif len(test) != 0 and bt:
        st.write('분류중입니다.')

        
from streamlit_cropper import st_cropper
from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header("Cropper Demo")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg','webp'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

if img_file:
    img = Image.open(img_file)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    
    # Manipulate cropped image at will
    st.write("Preview")
    _ = cropped_img.thumbnail((150,150))
    st.image(cropped_img)
    
# nodes = []
# edges = []
# nodes.append( Node(id="Spiderman", 
#                    label="Peter Parker", 
#                    size=25, 
#                    shape="circularImage",
#                    image="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_spiderman.png") 
#             ) # includes **kwargs
# nodes.append( Node(id="Captain_Marvel", 
#                    size=25,
#                    shape="circularImage",
#                    image="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_captainmarvel.png") 
#             )
# edges.append( Edge(source="Captain_Marvel", 
#                    label="friend_of", 
#                    target="Spiderman", 
#                    # **kwargs
#                    ) 
#             ) 

# config = Config(width=750,
#                 height=950,
#                 directed=True, 
#                 physics=True, 
#                 hierarchical=False,
#                 # **kwargs
#                 )

# return_value = agraph(nodes=nodes, 
#                       edges=edges, 
#                       config=config)

uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    st.write(type(uploaded_file))
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    data_df = pd.DataFrame(uploaded_file)
    st.dataframe(data_df)