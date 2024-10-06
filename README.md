# ONLY

## Overview

### 온리: 당신을 위한 단 하나의 챗봇

고객의 감정과 성향을 바탕으로 맞춤형 응답을 제공하는 서비스입니다.

빠른 실행은 2가지 방향으로 가능합니다. <br />

- `streamlit run Agent_ONLY.py text` : 채팅으로 *온리*의 답을 확인하고 싶은 경우 <br />
    - input: 텍스트 / output: 텍스트
- `streamlit run Agent_ONLY.py talk` : 소리로 *온리*의 답을 듣고 싶은 경우 <br />
    - input: 텍스트 / output: 사운드(TTS)



## 확인 사항

1) 파이썬 파일 상단 Upstage API key 입력
2) 아래 환경 설정

    ```
    python3 -m venv venv
    source venv/bin/activate
    pip3 install openai langchain python-dotenv tiktoken streamlit gtts
    pip3 install playsound==1.2.2
    pip3 install -U PyObjC
    ```



## 테스트 제안

<기본 실행>
1. `streamlit run Agent_ONLY.py t000`
    - t000 : 'text' 혹은 'talk' 기입
2. 팝업된 창을 통해 온리와 대화 진행
3. 각 turn마다 파악되는 감정 및 3-turn마다 파악되는 성향은 터미널을 통해 확인 가능
    ex)
    ```
    ### 기존에 고객이 나타낸 emotion 리스트 : ['불만', '실망', '실망']
    ### 현재 파악한 emotion : 기대
    ### 기존에 고객이 나타낸 personality 리스트 : ['예의없음']
    ### 현재 활용하는 고객의 personality : 예의없음
    ```

<상담 내역 리포트 생성을 원하는 경우>
1. `streamlit run Agent_ONLY.py t000`
    - t000 : 'text' 혹은 'talk' 기입
2. 팝업된 창을 통해 온리와 대화 진행
    - 대화 진행 시, 부정적인 감정을 담은 문장 반복 입력 → 모델이 ['불만', '실망', '의심']의 감정으로 판단하게 하기 위함
        ```
        [예시 문장]
        - 아 짜증나 옷이 마음에 안들잖아
        - 당장 환불해
        - 아니 무슨 말이 많아 그냥 하라면 해
        - 아 마음에 안들어 환불하는데 무슨 시간이 필요해
        - 무슨 옷을 이따구로 만들었어 짜증나게
        ```
    - 터미널로 부정적인 감정이 탐지되고 있는지 확인하며 4차례 이상 위의 대화 진행
3. 온리가 `죄송합니다. 제가 처리하기 어려운 문제네요. 더 자세한 상담을 위해 다른 상담원을 연결해 드리겠습니다.` 라고 말한 경우, 터미널에 출력된 상담 내역 리포트 확인 

<현재 세션을 종료하고 새롭게 다시 대화를 시작하고자 하는 경우>
1. `end` 입력
2. 화면에 나타난 새로운 세션 표시 확인 : `Session has been reset!`
3. 새롭게 대화 시작
    - 새로운 내용을 입력했을 때 기존 대화 기록이 사라지는지 확인 → 사라지는 경우 정상 처리된 것
4. 기존과 동일하게 대화 진행