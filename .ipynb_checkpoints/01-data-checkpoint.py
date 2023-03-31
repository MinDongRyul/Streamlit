{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e36141eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "st.title('데이터프레임 튜토리얼')\n",
    "\n",
    "# DataFrame 생성\n",
    "dataframe = pd.DataFrame({\n",
    "    'first column': [1, 2, 3, 4],\n",
    "    'second column': [10, 20, 30, 40],\n",
    "})\n",
    "\n",
    "# DataFrame\n",
    "# use_container_width 기능은 데이터프레임을 컨테이너 크기에 확장할 때 사용합니다. (True/False)\n",
    "st.dataframe(dataframe, use_container_width=False)\n",
    "\n",
    "\n",
    "# 테이블(static)\n",
    "# DataFrame과는 다르게 interactive 한 UI 를 제공하지 않습니다.\n",
    "st.table(dataframe)\n",
    "\n",
    "\n",
    "# # 메트릭\n",
    "st.metric(label=\"온도\", value=\"10°C\", delta=\"1.2°C\")\n",
    "st.metric(label=\"삼성전자\", value=\"61,000 원\", delta=\"-1,200 원\")\n",
    "\n",
    "# 컬럼으로 영역을 나누어 표기한 경우\n",
    "col1, col2, col3 = st.columns(3)\n",
    "col1.metric(label=\"달러USD\", value=\"1,228 원\", delta=\"-12.00 원\")\n",
    "col2.metric(label=\"일본JPY(100엔)\", value=\"958.63 원\", delta=\"-7.44 원\")\n",
    "col3.metric(label=\"유럽연합EUR\", value=\"1,335.82 원\", delta=\"11.44 원\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
