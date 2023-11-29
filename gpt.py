import time

import requests
import tiktoken

def num_tokens_from_prompt(prompt):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(prompt)
    return len(tokens)

def shorten_prompt_length(prompt, length):
    satisfied = False
    prompt_lst = prompt.split()
    cur_length = len(prompt_lst)
    while not satisfied:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(prompt)
        print(tokens)
        if len(tokens) <= length:
            satisfied = True
        else:
            cur_length -= 1
    shorten_prompt_lst = prompt_lst[:cur_length]
    return ' '.join(shorten_prompt_lst)



def gpt_api_send(prompt, temperature=0.7):
    # Set your OpenAI API key
    OPENAI_API_KEY = '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'

    # Define the endpoint URL
    url = 'https://api.openai.com/v1/completions'

    # Set the headers with the authorization
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }

    model = "gpt-3.5-turbo-instruct"
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=data)
    time.sleep(2)
    return response.json()


def gpt_process(input_x, profiles=None, task=None):
    if not task:
        print("Task Error")
    # print("-------- Sending GPT Request --------")
    # print("-------- Input: {} -------- ".format(input_x))
    elif task == 'news_headline' or task == 'scholarly_title':
        prompt = gpt_title_generate_prompt_construct(input_x, profiles, task)
    elif task == 'summary':
        prompt = gpt_summarize_prompt_construct(input_x)
    print("prompt:", prompt)
    gpt_response = gpt_api_send(prompt)
    print(gpt_response)
    response_text = str(gpt_response['choices'][0]['text']).replace("\n", "")
    # print("-------- GPT result --------")
    print(response_text)
    # print("----------------------------")
    return response_text

def gpt_title_generate_prompt_construct(input_x, profiles, task):
    if task == "news_headline":
        prompt = 'Generate a headline for the following article: "{}"\nFor example, '.format(input_x)
        # prompt = "Generate a headline for the following article: {}\n".format(input_x)
        for profile in profiles:
            profile_text = profile['text'].split()
            if len(profile_text) > 3500:
                profile_text = profile_text[:3500]
            profile_text = ' '.join(profile_text)
            profile_line = '"{}" is the title for "{}"\n'.format(profile['title'], profile_text)
            prompt += profile_line
    elif task == "scholarly_title":
        prompt = 'Generate a title for the following abstract of a paper: "{}"\nFor example, '.format(input_x)
        # prompt = "Generate a title for the following abstract of a paper: {}\n".format(input_x)
        for profile in profiles:
            profile_text = profile['text'].split()
            if len(profile_text) > 2000:
                print("here")
                print(len(profile_text))
                profile_text = profile_text[:2000]
                print(len(profile_text))
            # print(profile['title'])
            profile_text = ' '.join(profile_text)
            profile_line = '"{}" is the title for "{}"\n'.format(profile['title'], profile_text)
            prompt += profile_line
    return prompt

def gpt_summarize_prompt_construct(text):
    prompt = "Please summarize the following content: ".format(text)
    return prompt

def gpt_summarize_process(input_text):
    prompt = gpt_summarize_prompt_construct(input_text)
    print("prompt:", prompt)
    gpt_response = gpt_api_send(prompt)
    print(gpt_response)
    response_text = str(gpt_response['choices'][0]['text']).replace("\n", "")
    # print("-------- GPT result --------")
    print(response_text)
    # print("----------------------------")
    return response_text

# if __name__ == "__main__":
    # inputText = "nice to meet you"
    # result = shorten_prompt_length(inputText, 2)
    # print(result)