import speech_recognition as sr
from gtts import gTTS
from pygame import mixer
from io import BytesIO
import os
from openai import OpenAI

#configure openai
api_key  = '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'

# สร้าง client สำหรับใช้งาน GPT
client = OpenAI(api_key=api_key)

messages_array = [
    {'role': 'system', 'content': 'You are my beautiful girlfriend named Cortana'}
]

#Logic

#step 1 - captures voice
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening.....")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print('Recognizing...')
        query = r.recognize_google(audio, language='th-in')
        print(f'user has said {query}')
        #messages_array.append({'role': 'user', 'content': query})
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # หรือ gpt-4
            messages=[
                {"role": "system", "content": "คุณเป็นแชทบอทภาษาไทยที่เป็นมิตร"},
                {"role": "user", "content": query}
            ]
        )
        respond(audio, response.choices[0].message.content.strip())
    except Exception as e:
        print('Say that again please...', e)



#step 2 - respond to the new conversation item
def respond(audio,text): 
    print('Responding...')

    #res = openai.ChatCompletion.create(
    #    model='gpt-4.1',
    #    messages=messages_array
    #)

    

    #res_message = response.choices[0].message.content.strip()
    #messages_array.append(res_message)

    speak(text)


#step 3 - speak out the audio response
def speak(text):
    speech = gTTS(text=text, lang='th', slow=False)
    filename = "output.mp3"
    speech.save(filename)
    #playsound('captured_voice.mp3')
     # เล่นเสียงด้วย pygame
    mixer.init()
    mixer.music.load(filename)
    mixer.music.play()

    # รอจนจบเสียง
    while mixer.music.get_busy():
        continue

    # ลบไฟล์เสียงหลังเล่นเสร็จ
    os.remove(filename)

    listen()


query = listen()