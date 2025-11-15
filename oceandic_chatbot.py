import os
import streamlit as st
import nest_asyncio
from PIL import Image

nest_asyncio.apply()

# ==========================================
# 1. 프로젝트 개요
# ==========================================
st.header("🌊 해양 생물 실시간 정보 챗봇")

st.markdown("""
### 🐋 프로젝트 개요  
해양 생물의 **이름 또는 사진**을 입력하면 실시간 검색 기반으로  
특징·크기·서식지 등을 제공하는 챗봇입니다.
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

# LLM 초기화
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.4,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# 🔥 FIX: GoogleSearchAPIWrapper는 반드시 key를 직접 넣어줘야 함
search = GoogleSearchAPIWrapper(
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# ==========================================
# 3. 사용자 입력 (이름 or 사진)
# ==========================================
st.subheader("🔎 해양 생물 검색")

marine_name = st.text_input("해양 생물 이름을 입력하세요")
uploaded_img = st.file_uploader("또는 해양 생물 사진 업로드", type=["jpg", "png", "jpeg"])

identified_name = None

# ==========================================
# 4. 이미지 분석 → 이름 추출
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
            st.success(f"📌 분석 결과: **{identified_name}**")
        except Exception as e:
            st.error(f"이미지 분석 오류: {e}")

# ==========================================
# 5. 최종 검색 키워드 결정
# ==========================================
final_query = marine_name or identified_name

if not final_query:
    st.stop()

# ==========================================
# 6. 실시간 웹 검색
# ==========================================
if st.button("🔍 해양 생물 정보 가져오기"):
    with st.spinner("🌐 인터넷 정보 수집 중..."):
        search_results = search.run(final_query)

    # ==========================================
    # 7. LLM으로 정리
    # ==========================================
    summarize_prompt = f"""
    다음은 '{final_query}'에 대한 인터넷 검색 결과이다:

    {search_results}

    아래 형식에 맞춰 정리해줘:

    - 🐠 주요 특징
    - 📏 평균 크기
    - 🌍 서식지
    - 🍽️ 먹이 습성
    - ⚠️ 위험성 여부
    - 🛡️ 보호 등급
    - 🔎 유사한 해양 생물 추천
    """

    with st.spinner("📘 정보 정리 중..."):
        result = llm.invoke(summarize_prompt)

    st.subheader(f"🐋 '{final_query}' 정보 요약")
    st.write(result.content)
