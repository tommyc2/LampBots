import openai # Importing OpenAI python library

##############################################################################################
#
# Steps to get this script up and running:
#
# 1. Generate API key from OpenAI website (requirements money lodged in account, very small amount e.g. €1 or €2)
# 2. Install opencv-python using pip or install the libraries in requirements.txt
# 3. Make sure Python is version 3.10 or equivalent
#
#
#################################################################################################


openai.api_key = "api_key_goes_here"

messages = [{"role": "system", "content": "You give short and concise responses"}]

def CustomChatGPT():
    prompt = input("Enter your prompt (e.g question): ")
    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    if(ChatGPT_reply):
        messages.append({"role": "assistant", "content": ChatGPT_reply})
        print("----------------------------------","\n")
        print("Response: ",ChatGPT_reply,"\n")
        print("----------------------------------","\n")
    else:
        print("Failed to generate a response, please try again")

while True:
    CustomChatGPT()

