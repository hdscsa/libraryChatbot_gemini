import os
import streamlit as st
import nest_asyncio
from PIL import Image

nest_asyncio.apply()

# ==========================================
# 1. 프로젝트 개요에 맞춘 Streamlit UI 구성
# ==========================================
st.header("🌊 해양 생물 실시간 정보 챗봇")

st.markdown("""
### 🐋 프로젝트 개요  
해양 생물의 **이름 또는 사진**을 입력하면 인터넷에서 정보를 수집하여  
해당 생물의 특징·크기·서식지·먹이·위험성·보호등급 등을 제공하는 챗봇입니다.

### 🎯 개발 목표  
- 해양 생물 정보를 빠르게 확인할 수 있는 자동 정보 제공 챗봇 개발  
- 이름 또는 사진만 입력하면 실시간 검색 기반으로 설명 제공  

### 🧩 주요 기능  
- 텍스트/사진으로 입력된 해양 생물 자동 인식  
- 인터넷 검색을 통한 최신 정보 제공  
- 생물 특징·서식지·크기·먹이·위험성·보호등급 정리  
- 유사 생물 추천  
- 최신 연구 내용까지 반영되는 자동 업데이트형 정보 서비스 제공  

### 🎉 기대 효과  
- 누구나 손쉽게 해양 생물 학습 가능  
- 교육·탐구·과학 수업 활용  
- 해양 생물 다양성에 대한 이해 증진  
""")

# ==========================================
# 2. Gemini API 설정
# ==========================================
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("GOOGLE_API_KEY를 설정해주세요!")
    st.stop()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.4
)

search = GoogleSearchAPIWrapper()

# ==========================================
# 3. 사용자 입력 (이름 or 사진)
# ==========================================
st.subheader("🔎 해양 생물 검색")

marine_name = st.text_input("해양 생물 이름을 입력하세요")
uploaded_img = st.file_uploader("또는 해양 생물 사진을 업로드하세요", type=["jpg", "png", "jpeg"])

identified_name = None

# ==========================================
# 4. 이미지 → 해양 생물 이름 인식
# ==========================================
if uploaded_img:
    st.image(uploaded_img, caption="업로드된 이미지", width=300)

    img = Image.open(uploaded_img)

    with st.spinner("🔍 이미지 분석 중..."):
        vision_prompt = """
        이 사진 속 해양 생물의 이름을 한국어로 알려줘.
        가능하면 학명도 함께 제공해줘.
        """
        try:
            analysis = llm.invoke(input=vision_prompt, images=[img])
            identified_name = analysis.content.strip()
            st.success(f"📌 이미지 분석 결과: **{identified_name}**")
        except:
            st.error("이미지 분석 실패")

# ==========================================
# 5. 최종 검색 키워드 결정
# ==========================================
final_query = marine_name or identified_name

if not final_query:
    st.stop()

# ==========================================
# 6. 실시간 웹 검색 기반 정보 수집
# ==========================================
if st.button("🔍 해양 생물 정보 가져오기"):
    with st.spinner("🌐 인터넷에서 정보 수집 중..."):
        search_results = search.run(final_query)

    # ==========================================
    # 7. LLM으로 내용 정리
    # ==========================================
    summarize_prompt = f"""
    다음은 '{final_query}'에 대한 인터넷 검색 결과이다:

    {search_results}

    아래 형식에 맞춰 정리해줘:

    - 🐠 **주요 특징**
    - 📏 **평균 크기**
    - 🌍 **서식지**
    - 🍽️ **먹이 습성**
    - ⚠️ **위험성 여부(사람에게 위험한지)**
    - 🛡️ **보호 등급(멸종위기 여부 포함)**
    - 🔎 **유사한 해양 생물 추천**

    쉬운 한국어로, 학생이 읽기 좋게 정리해줘.
    """

    with st.spinner("📘 정보 정리 중..."):
        result = llm.invoke(summarize_prompt)

    st.subheader(f"🐋 '{final_query}' 정보 요약")
    st.write(result.content)
