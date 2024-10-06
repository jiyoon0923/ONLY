from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from langchain_upstage import ChatUpstage
import streamlit as st
import time
import argparse
import sys

from gtts import gTTS
import os
import playsound

import os
os.environ["UPSTAGE_API_KEY"] = "up_v2JbYoSOCJ9X9ACwdLsIGdI6gTDCI"

st.title('Customer Service Agent "ONLY"')

import tiktoken
encoder = tiktoken.get_encoding("cl100k_base")

INPUT_TOKENS = 0
OUTPUT_TOKENS = 0

def speak(text):
    tts = gTTS(text=text, lang='ko')
    filename='voice.mp3'
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)


# PROBLEM_CATE = 'Normal'
# def identifying_problem_categories(utterance):
#     global PROBLEM_CATE

#     problem_llm = ChatUpstage()

#     problem_prompt = f'''
#     ### instruction ###
#     당신은 매우 유능한 10년차 콜센터 상담 직원입니다. 고객의 말을 읽고 어떤 문제 상황을 해결하고자 하는지를 정확하게 파악할 수 있죠.
#     주어진 메세지를 읽고, 메세지에 담긴 문제 상황을 파악하세요.
#     문제 상황은 다음 6가지 카테고리 중 가장 적합한 것을 1개만 선택하세요.

#     <문제 상황 카테고리>
#     - 반품 요청 및 접수
#     - 상품 불량 및 하자
#     - 사이즈 및 색상
#     - 환불 및 결제
#     - 배송 및 수거
#     - 고객 정보 및 연락
#     - 상품 정보 및 주문

#     <output format>
#     답변: ['문제 상황']

#     ### input ###
#     메세지: {utterance}
#     ==========
#     답변: 
#     '''
#     #쇼핑 콜센터 반품 상황에 특정하여 문제 상황 카테고리를 정의함

#     problem = problem_llm.invoke(problem_prompt)

#     PROBLEM_CATE = problem

#     return problem, in_token, out_token

def emotion_recognition(utterance):
    emotion_llm = ChatUpstage()

    emotion_prompt = f'''
    ### instruction ###
    당신은 매우 유능한 10년차 콜센터 상담 직원입니다. 고객의 말을 읽고 어떤 감정 상태인지 정확하게 알 수 있죠.
    주어진 메세지를 읽고, 메세지에 담긴 고객의 감정을 파악해주세요. 
    감정은 다음 7가지 카테고리 중 가장 적합한 것을 1개만 선택하세요.

    <감정 카테고리>
    - 기쁨: 고객이 예상보다 좋은 결과나 서비스를 받은 경우. 긍정적인 감정 표현, 만족감을 드러내는 문구 사용.
    - 만족: 원활하게 문제 상황에 대해 처리가 된 경우. 처리에 대한 감사와 긍정적인 경험에 대한 표현 사용.
    - 기대: 고객이 서비스나 상품에 대한 긍정적인 기대감을 갖고 있는 경우. 미래에 대한 긍정적인 기대감을 나타내는 표현 사용.
    - 실망: 고객이 기대했던 결과와 실제 서비스나 상품이 일치하지 않은 경우. 기대감과 대비되는 부정적 반응과 실망을 나타내는 표현 사용.
    - 불만: 고객이 경험한 서비스나 제품에 대한 문제 및 불편함을 강하게 표현하는 경우. 불편함을 직접적으로 언급하며 빠른 해결을 요구.
    - 의심: 서비스나 처리 절차에 대해 신뢰하지 못하고 의심을 표현하는 경우. 불신을 드러내는 질문과 확인을 요구하는 문구 사용.
    - 중립: 특별한 감정을 드러내지 않고, 단순히 문제 해결에만 집중하는 경우. 짧고 간결한 요청과 특별한 감정적 표현 없이 단순 해결을 지시.

    <output format>
    답변: ['감정']

    ** Notes **
    답변 생성 시 감정 카테고리 단어만 생성할 것.
    어떤 설명이나 문장도 포함하지 마시오.

    ### input ###
    메세지: {utterance}
    ==========
    답변: 
    '''

    result = encoder.encode(emotion_prompt)
    in_token = len(result)

    emotion = emotion_llm.invoke(emotion_prompt).content.strip()
    if '답변: ' in emotion:
        emotion = emotion.split('변: ')[1]

    result = encoder.encode(emotion)
    out_token = len(result)

    return emotion, in_token, out_token


def personality_recognition(history, iter, persona_his):
    
    personality_llm = ChatUpstage()
    
    if iter < 5:
        blank = ''
    
    else:
        blank = f'''
        이전 대화를 통해 파악했던 동일한 고객의 성향은 다음과 같습니다.
        [{persona_his}]
        '''

    personality_prompt = f'''
    ### instruction ###
    당신은 매우 유능한 10년차 콜센터 상담 직원입니다. 짧은 대화로도 어떤 유형의 고객인지 고객의 성향을 정확하게 알 수 있죠.
    주어진 대화 기록을 읽고, user의 발화에 매핑된 감정 정보를 활용하여 user의 성향을 파악해주세요.
    성향은 다음 4가지 카테고리 중 가장 가까운 것을 1개만 선택하세요.

    <성향 카테고리> 
    - 꼼꼼함: 매우 꼼꼼하게 정보를 수집하고 분석하며, 세부적인 정보를 요구함. 질문이 매우 구체적이고, 확인을 반복함. 
    - 급함: 신속한 해결책을 요구하며, 시간을 절약하는 것을 중요시 함. 즉각적인 해결을 요청하고, 빠르고 신속한 대응을 요구함. 즉각적인 처리 요구 단어를 자주 사용.
    - 예의없음: 매우 감정적으로 발화하며, 원하는 것만 일방적으로 요구하는 등 무례한 태도를 보임. 명령조로 단순한 해결을 요구하는 표현, 반말과 공격적인 표현을 사용. 명령적, 강압적, 비하적인 어조를 사용.
    - 조용함: 특별한 특징없이 짧고 간결한 질문과 답을 통해 원하는 정보를 얻음. 감정을 많이 드러내지 않으며, 최소한의 필요한 대화만 이어감. 

    <output format>
    답변: ['성향']

    ** Notes **
    답변 생성 시 성향 카테고리 단어만 생성할 것.
    어떤 설명이나 문장도 포함하지 마시오.

    {blank}

    ### input ###
    대화 기록: {history}
    ==========
    답변: 
    '''

    result = encoder.encode(personality_prompt)
    in_token = len(result)

    personality = personality_llm.invoke(personality_prompt).content.strip()
    if '답변: ' in personality:
        personality = personality.split('변: ')[1]

    result = encoder.encode(personality)
    out_token = len(result)

    return personality, in_token, out_token


def identifying_response_attitude(emotion, personality):
    ### 감정과 personality에 따라 어떤 태도로 응대할지에 대한 설명 생성 
    
    attitude_llm = ChatUpstage()

    attitude_prompt = f'''
    ### instruction ###
    당신은 매우 유능한 10년차 콜센터 상담 직원입니다. 고객의 문제 상황을 빠르고 정확하게 처리하기 위해 어떤 것을 해야 하는지 잘 알고 있죠.
    또, 고객이 상담을 통해 더욱 만족스러운 결과를 얻을 수 있도록 어떤 태도로 응대를 해야하는지에 대해서도 매우 잘 알고 있습니다.
    주어진 고객의 감정 리스트와 성향을 바탕으로 올바른 응대 태도에 대해 문장으로 설명해주세요.

    <참고할 수 있는 고객의 성향별 대응 방법>
    1) 꼼꼼함
    - 정확한 답변: 꼼꼼한 고객에게는 정확하고 세세한 정보 제공이 중요합니다. 반복 확인 요청이 있을 경우 차분하고 명확하게 대응하는 것이 필요합니다.
    - 구체적 안내: 고객이 요청한 정보를 최대한 구체적으로 제공하고, 필요한 추가 정보도 미리 안내하는 것이 효과적입니다.

    2) 급함
    - 신속한 응대: 고객의 급함을 반영하여 가능한 한 신속하게 대응하는 것이 중요합니다. 처리 시간을 정확하게 안내하며 고객을 안심시키는 것이 필요합니다.
    - 정확한 시간 안내: 예상 처리 시간과 지연 가능성에 대해 솔직하게 안내해 불필요한 불안감을 줄여야 합니다.

    3) 예의없음
    - 차분한 응대: 예의 없는 고객에게는 감정적으로 대응하지 않고, 차분하게 문제 해결을 목표로 대화하는 것이 중요합니다.
    - 정중한 안내: 고객의 불만 사항을 존중하되, 무례한 표현은 대응하지 않고 핵심 문제에만 집중하는 전략이 효과적입니다.

    4) 조용함
    - 간결하고 효율적인 답변: 고객의 성향에 맞춰 짧고 명확한 답변을 제공하며, 추가적인 질문은 최소화합니다.
    - 필요 시 추가 정보 제공: 고객이 짧은 응답을 보이더라도 중요한 정보는 미리 제공해 향후 문제를 예방합니다.
    
    ### input ###
    감정 리스트: {emotion}
    성향: {personality}
    ==========
    답변: 
    '''
    
    result = encoder.encode(attitude_prompt)
    in_token = len(result)

    response_attitude = attitude_llm.invoke(attitude_prompt).content.strip()
    if '답변: ' in response_attitude:
        response_attitude = response_attitude.split('변: ')[1]

    result = encoder.encode(response_attitude)
    out_token = len(result)

    return response_attitude, in_token, out_token


def generate_report_for_human(history, emotion, personality):
    ### Agent가 해결할 수 없는 문제 상황인 경우, 인간 상담원에게 전달할 리포트 생성
    ### 리포트 :
    
    report_llm = ChatUpstage()

    report_prompt = f'''
    ### instruction ###
    당신과의 대화가 고객에게 도움이 되지 않는 것 같아요.
    지금까지 고객과 대화했던 기록과 파악한 고객의 감정 및 성향 리스트를 바탕으로 인간 상담원에게 제공할 리포트를 작성해주세요.
    리포트에는 (1) 고객의 문제 상황 (2) 시도한 해결 방법 (3) 고객의 감정 변화 (4) 고객의 예상 성향 (5) 한 줄 설명을 포함해 주세요.

    ### input ###
    대화 기록: {st.session_state.messages}
    고객의 감정 리스트: {st.session_state.emotion_list}
    고객의 성향 리스트: {st.session_state.persona_list}
    '''

    report = report_llm.invoke(report_prompt).content.strip()

    print('### 지금까지의 대화 기록을 바탕으로 고객의 상담에 대한 리포트를 작성하였습니다.')
    print(report)

    return report


def main(mode):
    # session_state 초기화
    if "upstage_model" not in st.session_state:
        st.session_state["upstage_model"] = "solar-pro-preview-240910"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "emotion_list" not in st.session_state:
        st.session_state.emotion_list = []

    if "persona_list" not in st.session_state:
        st.session_state.persona_list = []

    if "bad_emotion" not in st.session_state:
        st.session_state.bad_emotion = 0

    if "INPUT_TOKENS" not in st.session_state:
        st.session_state.INPUT_TOKENS = 0

    if "OUTPUT_TOKENS" not in st.session_state:
        st.session_state.OUTPUT_TOKENS = 0


    # print('대화 기록 출력')
    # print(st.session_state.messages)

    #### 대화 시작
    if prompt := st.chat_input("새로운 대화를 시작 시 꼭 'end'를 입력해주세요!"):

        if prompt == 'end':
            print('input tokens : ', INPUT_TOKENS)
            print('output tokens : ', OUTPUT_TOKENS)
            st.session_state.clear()
            st.success("Session has been reset!")

        else:
            with st.chat_message("user"):

                st.markdown(prompt)
                print('=====================')
                print('\n')
                print('### 기존에 고객이 나타낸 emotion 리스트 : ', st.session_state.emotion_list)

                emotion, in_token, out_token = emotion_recognition(prompt)
                st.session_state.emotion_list.append(emotion)

                st.session_state.INPUT_TOKENS += in_token
                st.session_state.OUTPUT_TOKENS += out_token

                if '실망' or '불만' or '의심' in str(emotion):
                    st.session_state.bad_emotion += 1
                
                print('### 현재 파악한 emotion : ', emotion)

                st.session_state.messages.append({"role": "user", "content": prompt, "emotion": emotion})

                # print('persona_list : ', persona_list)
                iters = len(st.session_state.messages)-1

                if iters == 4:
                    n = 1
                elif iters > 4:
                    n = (iters - 4) / 6 + 1
                else:
                    n = 0
                    personality = '아직 파악하기 어려움'

                print('### 기존에 고객이 나타낸 personality 리스트 : ', st.session_state.persona_list)

                if n > 0:
                    if n == int(n):
                        # print('personality 파악 turn')
                        personality, in_token, out_token = personality_recognition(st.session_state.messages[-5:], iters, st.session_state.persona_list) #3-turn의 history만 input / 이전 personality list도 현재 personality 파악에 활용
                        st.session_state.persona_list.append(personality)

                        print('### 새롭게 파악한 고객의 personality : ', personality)

                        st.session_state.INPUT_TOKENS += in_token
                        st.session_state.OUTPUT_TOKENS += out_token
                    else:
                        # print('이전 personality 활용')
                        personality = st.session_state.persona_list[-1] #personality 파악 순서가 아니면 이전 파악해 둔 정보 활용

                        print('### 현재 활용하는 고객의 personality : ', personality)

                print('\n')
                print('=====================')
                print('\n')

            with st.chat_message("assistant"):
                
                #### 답변 
                message_placeholder = st.empty()
                full_response = ""

                # 사용자와 챗봇 메시지를 분리하여 저장할 리스트 생성
                user_inputs = []
                ai_responses = []

                # 메시지 리스트 순회
                for msg in st.session_state.messages:
                    if msg['role'] == 'user':
                        user_inputs.append(msg['content'])
                    elif msg['role'] == 'assistant':
                        ai_responses.append(msg['content'])

                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                for user_input, ai_response in zip(user_inputs, ai_responses):
                    memory.chat_memory.add_user_message(user_input)
                    memory.chat_memory.add_ai_message(ai_response)

                # LLM
                llm = ChatUpstage()

                # Prompt
                if st.session_state.bad_emotion > 3:

                    report = generate_report_for_human(st.session_state.messages, st.session_state.emotion_list, st.session_state.persona_list)

                    result = {'text':''}
                    result['text'] = '죄송합니다. 제가 처리하기 어려운 문제네요. 더 자세한 상담을 위해 다른 상담원을 연결해 드리겠습니다.'
                else:
                    response_attitude, in_token, out_token = identifying_response_attitude(st.session_state.emotion_list[-3:], personality) #3-turn의 emotion만 활용
                    print('### 고객의 감정 및 성향을 바탕으로 생성한 응대 attitude : ', response_attitude)

                    st.session_state.INPUT_TOKENS += in_token
                    st.session_state.OUTPUT_TOKENS += out_token

                    llm_prompt = ChatPromptTemplate(
                        messages=[
                            SystemMessagePromptTemplate.from_template(
                                f'''
                            당신은 매우 유능한 10년차 콜센터 상담 직원 "온리"입니다. 아주 자연스럽고 친절한 대화를 통해 고객의 문제 상황을 매우 효과적으로 해결할 수 있죠.
                            고객의 입력이 주어지면 아래 응대 방법을 잘 읽고, 이를 참고하여 입력에 대해 적절한 답을 제공해주세요.
                            응대 방법: {response_attitude}
                            '''
                            ),
                            MessagesPlaceholder(variable_name="chat_history"),
                            HumanMessagePromptTemplate.from_template("{question}"),
                        ]
                    )

                    conversation = LLMChain(llm=llm, prompt=llm_prompt, verbose=True, memory=memory)

                    result = conversation({"question": prompt})

                if mode == 'talk':
                    speak(result['text'])
                    full_response = result['text']
                if mode == 'text':
                    for chunk in result['text'].split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            print('\n')
            print('총 입력 토큰 수 : ', st.session_state.INPUT_TOKENS)
            print('총 출력 토큰 수 : ', st.session_state.OUTPUT_TOKENS)
            print('\n')


if __name__ == "__main__":
    mode = sys.argv[-1]
    print(f'[{mode} 모드로 대화를 진행합니다.]\n')
    main(mode)