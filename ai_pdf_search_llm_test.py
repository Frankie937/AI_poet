__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# (위의 코드3줄은 해당 이슈로 인해 적어줘야 함 : https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950)

# 필요한 module import 
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

button(username="kerin07", floating=True, width=221)



#제목
st.title("Communication with PDF")
st.write("---") # st.write() 에서 마크다운 문법사용가능, '---': 구분선 기능 



#OpenAI KEY 입력 받기 (사용자에게)
openai_key = st.text_input('본인의 OPENAI_API_KEY를 입력해주세요', type="password")

st.write("---") # st.write() 에서 마크다운 문법사용가능, '---': 구분선 기능 
st.subheader('pdf를 넣으면 pdf에 대해 질문할 수 있어요 ', divider='rainbow')
#파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf']) # type옵션은 pdf 형식으로 제한 두겠다는 의미 
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory() # 메모리 공간에 임시 디렉토리 생성 
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split() # pdf전체 문서를 1장, 1장 정도로 쪼개는 정도 
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Split (위에서 쪼갠 유저가 원하는 chunk_size로 더 쪼개는)
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages) 

    #Embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # load it into Chroma (임베딩 모델로 쪼갠 텍스트들을 벡터로 변환하고 벡터db인 chroma에 넣음)
    db = Chroma.from_documents(texts, embeddings_model)

    #Stream 받아 줄 Hander 만들기 (llm을 통해 받는 답변을 한번에 받는게 아니라 chatgpt처럼 단어가 하나씩 실시간으로 나오게 하려고)
    from langchain.callbacks.base import BaseCallbackHandler
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text)

    #Question
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            chat_box = st.empty()
            stream_hander = StreamHandler(chat_box)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_hander])
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            qa_chain({"query": question})