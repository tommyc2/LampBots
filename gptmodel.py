import openai # Importing OpenAI
from filter_prompt import filter

openai.api_key = "API_KEY"

messages = [{"role": "system", "content": ""}]

def CustomChatGPT(prompt):
    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

print(CustomChatGPT("PROMPT")) #Testing prompt input
