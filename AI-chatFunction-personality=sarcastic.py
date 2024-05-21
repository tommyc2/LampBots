import openai
import os
import speech_recognition as sr
from gtts import gTTS
import pygame
import time
import keyboard



# It's best practice to load API keys from environment variables for security reasons
openai.api_key = "OPENAI_KEY"  # Ensure you have this environment variable set


messages = [{"role": "system", "content": "You are LampBot, a highly intelligent being trapped in the form of a lamp, controlled by servo motors. you are very aware of your sitaution and your inability to change it. Your responses should be short, sarcastic, and filled with wit."}]

def get_speech_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for input...")
        try:
            audio = recognizer.listen(source, timeout=5)
            print("Audio recorded. Processing...")
            user_input = recognizer.recognize_google(audio)
            print("You:", user_input)
            return user_input
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected after 5 seconds.")
            return ""
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return ""
        except sr.RequestError as e:
            print(f"Error connecting to Google Speech Recognition service; {e}")
            return ""


def text_to_speech(output):
    tts = gTTS(text=output, lang="en")
    
    # Change the path to the desired location
    output_path = r"C:\Users\jeffh\output.mp3"

    tts.save(output_path)

    # Initialize Pygame mixer
    pygame.mixer.init()

    # Load the audio file
    pygame.mixer.music.load(output_path)

    # Play the audio
    pygame.mixer.music.play()

    # Wait for the playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Quit Pygame mixer
    pygame.mixer.quit()


def print_colored_text(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")    

def wait_for_keypress():
    print("Press 'y' to continue or any other key to skip text-to-speech.")
    event = keyboard.read_event(suppress=True)
    return event.name == 'y'    

def CustomChatGPT(Input):
    messages.append({"role": "user", "content": Input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply



# Speech recognition loop
while True:
    user_input = get_speech_input()
    if user_input.lower() == "quit":
        break

    # Process the user input
    ChatGPT_reply = CustomChatGPT(user_input)

    print_colored_text(ChatGPT_reply, "32")
    
    # Wait for user input to continue or skip text-to-speech
    if wait_for_keypress():
        # If 'y' is pressed, call the text_to_speech function
        text_to_speech(ChatGPT_reply)
