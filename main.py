# from dotenv import load_dotenv
# load_dotenv()
import streamlit as st
from langchain.llms import openai
from langchain.chat_models import ChatOpenAI

# llm = openai() # ChatGPT의 일반적인 Complete 모드 
chat_model = ChatOpenAI() # ChatGPT의 Chat(채팅) 모드 (챗봇이랑 대화하는 형태)


st.title('인공지능 시인')

content = st.text_input('시의 주제를 제시해주세요')

if st.button('시 작문 요청'):
    with st.spinner('시 작성 중...'):
        result = chat_model.predict(content + "에 대한 시를 써줘")
        st.write(result)

# streamlit cloud에 배포 예정 (내 로컬pc아닌)

    