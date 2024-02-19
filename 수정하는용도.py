import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image #폴더 내 이미지 보여주기

st.sidebar.title("목차")
st.sidebar.markdown("* <데이터 수집~ EDA> 개요, 데이터 소개 & 수집방안, EDA")
st.sidebar.markdown("* <모델링> 모델링 & 튜닝 & 평가(1), 모델링 & 튜닝 & 평가(2), 모델링 & 튜닝 & 평가(3), 모델링 & 튜닝 & 평가(4)")
st.sidebar.markdown("* <활용과 마무리> 린가드, 샬라메,결론, 마무리")



# 타이틀 적용 예시
st.title('린가드 & 티모시 서울 전세집 찾아주기 Home Sweet Home Project')

# Subheader 적용
st.subheader('B07 김치완 정도영')

st.markdown('---')

 # 세로로 나누기
col1, col2 = st.columns(2)

with col1:
 st.header("제시 린가드")
 img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/IMG_3432.PNG")
 st.image(img)

with col2:
 st.header("티모시 샬라메")
 img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/IMG_3441.PNG")
 st.image(img)


# 탭 생성 : 첫번째 탭의 이름은 Tab A 로, Tab B로 표시합니다. 
tab1, tab2, tab3 = st.tabs(['개요', '데이터 소개 & 수집방안', 'EDA'])
tab4, tab5, tab6, tab7 = st.tabs(['모델링 & 튜닝 & 평가(1)', '모델링 & 튜닝 & 평가(2)', '모델링 & 튜닝 & 평가(3)', '모델링 & 튜닝 & 평가(4)'])
tab8, tab9, tab10, tab11 =  st.tabs(['린가드', '샬라메','결론', '마무리'])

with tab1:
  #tab A 를 누르면 표시될 내용
  # 타이틀 적용 예시
  st.title('개요')

  # Header 적용
  st.header('프로젝트의 목표')
  # 마크다운 문법 지원
  st.markdown('1. 서울시의 전세 가격을 예측하는 머신러닝 모델을 개발하고, 이를 **실제 문제 해결**에 적용')
  st.markdown('2. 두명이서 프로젝트를 진행하기에 모든 과정을 함께 진행하고 **최대한 다양하게 파이썬을 이용**')
  st.markdown('3. 활용은 **재미있게**!')
  # 선 그리기
  st.markdown('---')
  
  # Header 적용
  st.header('팀원 역할')
  st.text('김치완: 데이터 수집, 전처리, EDA, 모델링, 대시보드 제작')
  st.text('정도영: 데이터 전처리, 모델링, 평가, 튜닝, 대시보드 제작')
  
  st.markdown('---')
  
  # Header 적용
  st.header('진행순서')
  st.text('1. 개요')
  st.text('2. 데이터 소개 & 수집방안')
  st.text('3. EDA')
  st.text('4. 전세가격 예측 모델링 & 튜닝 & 평가')
  st.text('5. 제시 린가드 전세집 평가해주기')
  st.text('6. 티모시 샬라메 접세집 평가해주기')
  st.text('7. 결론')
  st.text('8. 마무리')
  
with tab2:
  #tab B를 누르면 표시될 내용 
  st.title('데이터 소개 & 수집방안')  
  st.header('1. 서울시 전세 데이터')
  st.text('서울 열린 데이터 광장 - 부동산 전월세가 데이터 활용')
  df1 =pd.read_csv("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/진짜 최종/18-23 서울시 전세자료.csv", encoding='utf-8')
  st.dataframe(df1, use_container_width=True)
  
  st.header('2. 2018~2023 뉴스 키워드')
  st.text('user_agent, konlpy, mecab-python, BeautifulSoup 라이브러리 활용하여 Naver기사 웹 크롤링')
  df2 = pd.read_excel("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/키워드/Keyword1819.xlsx")
  df3 = pd.read_excel("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/키워드/Keyword2021.xlsx")
  df4 = pd.read_excel("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/키워드/Keyword2223.xlsx")
  st.dataframe(df2, use_container_width=True)
  
  st.header('3. 서울시 CCTV')
  st.text('서울 열린 데이터 광장 - 서울시 안심이 CCTV 연계 현황')
  df5 = pd.read_csv("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/서울시 안심이 CCTV 연계 현황(수정).csv", encoding='utf-8')
  st.dataframe(df5, use_container_width=True)
  
  st.header('4. 서울시 관광명소')
  st.text('» 서울시 열린 데이터 광장 - 서울시 관광 명소')
  df6 = pd.read_csv("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/서울시 관광 명소(수정).csv", encoding='utf-8')
  st.dataframe(df6, use_container_width=True)
  
with tab3:
  Seoul_df1 = pd.read_csv("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/DataFrame/서울시_평당가,증감율.csv", encoding='utf-8')
  Seoul_df2 = pd.read_csv("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/DataFrame/서울사구_증감율,평당가.csv", encoding='utf-8')
  std1 = pd.read_csv("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/DataFrame/서울시_표준편차.csv", encoding='utf-8')
  std2 = pd.read_csv("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/DataFrame/서울사구_표준편차.csv", encoding='utf-8')
  Seoul_df3 = pd.read_csv("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/DataFrame/서울시_거래량(완).csv", encoding='utf-8')
  Seoul_df4 = pd.read_csv("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/DataFrame/서울사구_거래량.csv", encoding='utf-8')
  
  #tab B를 누르면 표시될 내용 
  st.title('EDA')
  st.header('1. 평당가 변화')
  
  fig = px.line(data_frame = Seoul_df1, x="접수년도", y="평당가격(만원)")
  fig.update_xaxes(tickvals=Seoul_df1["접수년도"].unique())
  fig.update_layout(title_text="서울시 평당가 변화")
  st.plotly_chart(fig)
  
  
  plt.rc('font', family='NanumGothic') #폰트 설정
  fig = px.line(data_frame = Seoul_df2, x="접수년도", y="평당가격(만원)", color="서울사구")
  fig.update_xaxes(tickvals=Seoul_df2["접수년도"].unique())
  fig.update_layout(title_text="서울사구 평당가 변화")
  st.plotly_chart(fig)

  
  st.header('2. 증감율 변화')
  
  fig = px.bar(data_frame = Seoul_df1, x="접수년도", y="증감율", text_auto='.0d')
  fig.update_xaxes(tickvals=Seoul_df1["접수년도"].unique())
  fig.update_layout(title_text="서울시 평당가 증감율")
  st.plotly_chart(fig)
  
  
  fig = px.bar(data_frame = Seoul_df2, x="접수년도", y="증감율", barmode='group', color = '서울사구', text_auto='.0d')
  fig.update_xaxes(tickvals=Seoul_df2["접수년도"].unique())
  fig.update_layout(title_text="서울사구 평당가 증감율")
  st.plotly_chart(fig)
  

  
  
  st.header('3. 평당가 표준편차')
  
  fig = px.bar(data_frame = std1, x="접수년도", y="평당가격(만원)", text_auto='.0d')
  fig.update_xaxes(tickvals=Seoul_df1["접수년도"].unique())
  fig.update_layout(title_text="서울사 전세금 표준편차")
  st.plotly_chart(fig)
  
  fig = px.bar(data_frame = std2, x="접수년도", y="평당가격(만원)", barmode='group', text_auto='.0d', color="서울사구")
  fig.update_xaxes(tickvals=Seoul_df1["접수년도"].unique()) 
  fig.update_layout(title_text="서울사구 전세금 표준편차")
  st.plotly_chart(fig)
  

  
  st.header('4. 거래량')
  
  fig = px.bar(data_frame = Seoul_df3, x="접수년도", y="거래량", text_auto='.0d', color="건물용도", barmode='group')
  fig.update_layout(title_text="서울시 건물 거래량")
  st.plotly_chart(fig)
  
  fig = px.bar(data_frame = Seoul_df4, x="서울사구", y="거래량", barmode='group', text_auto='.0d', color="건물용도")
  fig.update_layout(title_text="서울사구 건물 거래량")
  st.plotly_chart(fig)
  
  st.header('5. 전세가 변화 흐름')
  col1, col2 = st.columns(2)
  col3, col4 = st.columns(2)
  col5, col6 = st.columns(2)
  
  with col1:
   st.header("2018")
   img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/18년도 지도.png")
   st.image(img)
 
  with col2:
   st.header("2019")
   img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/19년도 지도.png")
   st.image(img)
   
   
  with col3:
   st.header("2020")
   img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/20년도 지도.png")
   st.image(img)

  with col4:
   st.header("2021")
   img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/21년도 지도.png")
   st.image(img)
   
   
  with col5:
   st.header("2022")
   img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/22년도 지도.png")
   st.image(img)
   
  with col6:
   st.header("2023")
   img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/23년도 지도.png")
   st.image(img)
 

  st.header('6. 키워드 분석')
  col7, col8, col9 = st.columns(3)

  with col7:
   st.header("2018-2019")
   img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/18~19년도 키워드.png")
   st.image(img)
  with col8:
   st.header("2020-2021")
   img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/20~21년도 키워드.png")
   st.image(img)
  with col9:
   st.header("2022-2023")
   img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/22~23년도 키워드.png")
   st.image(img) 
  
  st.markdown('---')
  
  
with tab4:
  #tab B를 누르면 표시될 내용 
  st.header('[데이터 전처리]')
  st.subheader('1. 결측치 처리')  
  st.write('> 건축년도: 자치구명 별 건축년도 최빈값 대체')
  
    # 코드 표시
  sample_code = '''
  # 자치구별 최빈값 구하기 (딕셔너리 형태) {'자치구명':'건축년도'의 최빈값}
  year_mode = s_df.groupby('자치구명')['건축년도'].agg(lambda x:x.mode()[0]).to_dict()
  '''
  st.code(sample_code, language="python")

    # 코드 표시
  sample_code = '''
  # 건축년도의 결측값 자치구명별 최빈값으로 대체
  s2_df['건축년도'].fillna(s_df['자치구명'].map(year_mode), inplace=True)
  '''
  st.code(sample_code, language="python")
  
  st.subheader('2. 데이터 필터링')
  st.write('> 건물용도: 데이터 Null값 317행 삭제')
  
    # 코드 표시
  sample_code = '''
  # '건물용도' 결측치 '미정'으로 대체
  s_df['건물용도'] = s_df['건물용도'].fillna('미정')
  '''  
  st.code(sample_code, language="python") 
  
    # 코드 표시
  sample_code = '''
  # '건물용도'가 '미정' 해당 데이터 제외
  s_df =  s_df[s_df['건물용도'] != '미정']
  '''  
  st.code(sample_code, language="python")

  st.markdown('---')
  
  st.header('[예측 모델]')
  st.subheader('1. 다중선형회귀 모델')
  img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/머신러닝/1차 다중선형그래프.png")
  st.image(img)
  st.subheader('▷ MAE: 10380.42   / :blue[RMSE: 17170.01] / :red[R2 SCORE: 0.6]')
  
  st.subheader('2. 랜덤포레스트 모델')
  img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/머신러닝/1차 랜덤포레스트그래프.png")
  st.image(img)
  st.subheader('▷ MAE: 4323.11 / :blue[RMSE: 7867.56] / R2 SCORE: 0.92')
  
  
  
with tab5:
  #tab B를 누르면 표시될 내용 
  st.header('[데이터 전처리]')
  st.subheader('1. 새로운 컬럼 추가')
  st.write('> "법정동명" / "층" / "지번구분" 컬럼')
  
  st.subheader('2. 결측치 처리')
  st.write('> 층: 건물용도 "단독다가구"인 건물에서 층 결측치 발생하여 평균값(1층) 대체')
  
      # 코드 표시
  sample_code = '''
  #단독다가구 1층으로 null값 대치
  s_df['층'] = s_df['층'].fillna(1)
  '''  
  st.code(sample_code, language="python")  
  
  
  st.write('> 지번구분: 주소정보가 "단독다가구"인 경우 "동" 정보까지만 제공 받고 있어 "기타"로 분류') 
  
    # 코드 표시
  sample_code = '''
  #지번구분 null 기타로 대치
  s_df['지번구분'] = s_df['지번구분'].fillna('기타')
  '''  
  st.code(sample_code, language="python")   
  
  
  st.markdown('---')
  
  st.header('[예측 모델]')
  st.subheader('1. 랜덤포레스트 모델')
  img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/머신러닝/2차 랜덤포레스트그래프.png")
  st.image(img)
  st.subheader('▷ MAE: 4178.68 / :blue[RMSE: 7858.8] / R2 SCORE: 0.92')
  
  st.subheader('2. Gradient Boost 모델 ')
  img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/머신러닝/2차 GRADIENT BOOST그래프.png")
  st.image(img)
  st.subheader('▷ MAE: 8052.51 / :blue[RMSE: 13611.29] / :red[R2 SCORE: 0.75]')
  
  
with tab6:
  #tab B를 누르면 표시될 내용 
  st.header('[데이터 전처리]')
  st.subheader('1. 아웃라이어 제거')
  st.write('> 층: 주거환경 부합하지 않은 층수 (EX)지하4층, 601층 등')
  st.write('> 임대면적: 150평 미만')
  st.write('> 건축년도: 1960년 이전 2023년 년도')
  
      # 코드 표시
  sample_code = '''
  #이상치 제거
  s_df = s_df[(s_df['층'] <= 40) & (s_df['층'] >= -1) 
  & (s_df['건축년도'] >= 1960) & (s_df['건축년도'] <= 2023) & (s_df['임대면적'] <= 150)]
  '''  
  st.code(sample_code, language="python")   
  
  st.subheader('2. 하이퍼파라미터 조정')
  st.write('> min_samples_split 기본값 2에서 10으로 조정')
  
      # 코드 표시
  sample_code = '''
  RandomForestRegressor(n_estimators=100, random_state=42, min_samples_split=10)
  '''  
  st.code(sample_code, language="python")    
  
  st.markdown('---')
  
  st.header('[예측 모델]')
  st.subheader('1. 랜덤포레스트 모델')
  img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/머신러닝/3차 랜덤포레스트 그래프.png")
  st.image(img)
  st.subheader('▷ MAE: 3997.88 / :blue[RMSE: 6934.29] / R2 SCORE: 0.92')

  
with tab7:
  #tab B를 누르면 표시될 내용 
  st.header('[데이터 전처리]')
  st.subheader('1. 외부데이터 추가')
  st.write('> 서울시 안심이 CCTV현황, 서울_명소 각각 자치구명별 그룹화하여 새 컬럼 생성')
  
    # 코드 표시
  sample_code = '''
  train_df = train_df.groupby('신주소')['상호명'].nunique().reset_index()
  ''' 
  st.code(sample_code, language="python") 
    # 코드 표시
  sample_code = '''
  train_df = train_df.rename(columns={'신주소': '자치구명'})
  train_df = train_df.rename(columns={'상호명': '명소'})
  ''' 
  st.code(sample_code, language="python") 
    # 코드 표시
  sample_code = '''
  pd.merge(s_df, train_df, how='left', on='자치구명')
  '''  
 
  st.code(sample_code, language="python")    
  
  st.markdown('---')
  
  st.header('[예측 모델]')
  st.subheader('1. 랜덤포레스트 모델')
  img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/머신러닝/4차 랜덤포레스트 그래프.png")
  st.image(img)
  st.subheader('▷ MAE: 3906.52 / :blue[RMSE: 6739.42] / R2 SCORE: 0.92')
  
  import plotly.graph_objects as go

  def main():
     # feature importances
      importances = [0.545, 0.115, 0.034, 0.087, 0.027,
                   0.002, 0.09, 0.001, 0.002, 
                   0.007, 0.008, 0.003, 0.005, 0.005, 0.003]

      variables = ['임대면적sc', '건축년도sc', '층mm', 'cctv 수량', '명소',
                 '건물용도_단독다가구', '건물용도_아파트', '건물용도_연립다세대', '건물용도_오피스텔',
                 '접수년도_2018', '접수년도_2019', '접수년도_2020', '접수년도_2021', '접수년도_2022', '접수년도_2023']

      importances, variables = zip(*sorted(zip(importances, variables), reverse=True))

      data = go.Bar(x=importances[::-1], y=variables[::-1], orientation='h')

      layout = go.Layout(title='Feature Importances',
                       xaxis=dict(title='Importance'),
                       yaxis=dict(title='Variables'))

      fig = go.Figure(data=data, layout=layout)

      st.plotly_chart(fig)

  if __name__ == "__main__":
      main()

  st.subheader('외부데이터 :red[cctv수량/명소] 상대적으로 높은 변수중요도')


with tab8:
  col1, col2= st.columns(2)
  with col1:
    st.header("고객1. 제시 린가드")
    st.subheader('"안녕 BRO! 나 왔다 FC Seoul! 원한다 집! 조건 전세!"')
    
    st.markdown('---')
    
    st.subheader('가져온 매물 정보')
    st.write('> 지역: **:마포구 합정동**')
    st.write('> 지번구분: 대지')
    st.write('> 층/평수/건물용도: **:24층, 42평의 아파트**')
    st.write('> 전세가: **:2억 7천**')
    
    
  with col2:
    img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/IMG_3440.PNG")
    st.image(img)

    
with tab9:
  col1, col2= st.columns(2)
  with col1:
    st.header("고객2. 티모시 샬라메")
    st.subheader('"영화 웡카와 듄2으로 한국활동을 시작하려고 합니다. 한국스러운 좋은 집을 찾고싶네요"')
    
    st.markdown('---')
    
    st.subheader('가져온 매물 정보')
    st.write('> 지역: **:종로구 무악동**')
    st.write('> 지번구분: 대지')
    st.write('> 층/평수/건물용도: **:16층, 34평 아파트**')
    st.write('> 전세가: **:7억**')
    
    
  with col2:
    img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/IMG_3443.PNG")
    st.image(img)
    
with tab10:
  st.title('결론')
  col1, col2= st.columns(2)
  with col1:
    st.header("제시 린가드")
    img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/IMG_3435.PNG")
    st.image(img)
    st.write('> 모델 예측결과: 29946.08980516(만원)')
    st.write('> RMSE기준: 약23,207~ 36,685(만원)사이의 값이 합리적')
    st.write('> 실제 가격은 2억 7천임으로 합리적인 매물 !!')
    
    st.subheader('안녕하세요 FC 서울의 린가드입니다!')
    
  with col2:
    st.header("티모시 샬라메")  
    img = Image.open("C:/Users/USER/Desktop/서울시 전세 분석 프로젝트/스크린샷/IMG_3444.PNG")
    st.image(img)
    st.write('> 모델 예측결과: 44232.39014761(만원）')
    st.write('> RMSE기준: 37,493~50,971(만원)사이의 값이 합리적')
    st.write('> 실제 가격은 7억임으로 불합리한 매물 !!')
    
    st.subheader('Good by Korea..')
    
with tab11:
  #tab B를 누르면 표시될 내용 
  st.title('프로젝트를 마무리하며...')
  st.header('한계점')
  st.write('> :blue[팀원의 이탈]로 인한 역할 배분의 어려움') 
  st.write('> 데이터의 양 증가로 :blue[노트북 성능의 한계]로 모델링과 Grid Search활용이 불가능해지며 한계 봉착')
  st.write('> 전세 매물의 :blue[구체적인 사항(방의 갯수, 인테리어 등)]을 조사하지 못함')
  
  st.header('좋았던 점')
  st.write('> 프로젝트를 진행하며 서로 :red[대학 동문]임을 알게됨') 
  st.write('> SQL, Python으로 모든 과정을 직접 진행하며 스스로 :red[부족한 부분을 파악하고 많은 실력 향상]을 이룸')
  st.write('> :red[새로운 라이브러리들과 머신러닝]을 직접 사용해본 경험')
  st.write('> 가상의 데이터가 아닌 :red[실생활 데이터]를 이용한 경험')
  st.write('> 임정 튜터님 강의에서 배운 :red[전처리의 중요성]을 정말 뼈저리게 경험') 
  
