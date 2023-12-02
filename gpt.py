import time

import requests

def gpt_api_send(prompt, temperature=0.7, maxToken=None):
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
    if maxToken:
        data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            'max_tokens': maxToken
        }
    else:
        data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
        }
    response = requests.post(url, headers=headers, json=data)
    # time.sleep(2)
    return response.json()


def gpt_process(input_x, profiles=None, task=None):
    if not task:
        print("Task Error")
    # print("-------- Sending GPT Request --------")
    # print("-------- Input: {} -------- ".format(input_x))
    elif task == 'news_headline' or task == 'scholarly_title' or task == 'news_headline_summary' or task == 'scholarly_title_summary':
        prompt = gpt_title_generate_prompt_construct(input_x, profiles, task)
        print("prompt:", prompt)
        gpt_response = gpt_api_send(prompt, maxToken=50)
    elif task == 'summary':
        prompt = gpt_summarize_prompt_construct(input_x)
        print("prompt:", prompt)
        gpt_response = gpt_api_send(prompt, maxToken=300)
    else:
        print("Task Undefined")
    print(gpt_response)
    response_text = str(gpt_response['choices'][0]['text']).replace("\n", "")
    # print("-------- GPT result --------")
    print(response_text)
    # print("----------------------------")
    return response_text

def gpt_title_generate_prompt_construct(input_x, profiles, task):
    if task == "news_headline" or task == "news_headline_summary":
        # prompt = 'Generate a headline for the following article: "{}"\nFor example, '.format(input_x)
        prompt = '{}\nFor example, '.format(input_x)
        # prompt = "Generate a headline for the following article: {}\n".format(input_x)
        for profile in profiles:
            profile_text = profile['text'].split()
            if len(profile_text) > 3500:
                profile_text = profile_text[:3500]
            profile_text = ' '.join(profile_text)
            profile_line = '"{}" is the title for "{}"\n'.format(profile['title'], profile_text)
            prompt += profile_line
    elif task == "scholarly_title" or task == "scholarly_title_summary":
        # prompt = '{}\nFor example, '.format(input_x)
        prompt = 'For example, '
        # prompt = 'Based on the example, generate a title for the following abstract of a paper: "{}"\nFor example, '.format(input_x)
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
        prompt = prompt + '\n{}'.format(input_x)
    else:
        print("Task Undefined")
    return prompt

def gpt_summarize_prompt_construct(text):
    tokens = text.split()
    if len(tokens) > 2000:
        tokens = tokens[:2000]
    content_text = ' '.join(tokens)
    prompt = "Please summarize the following content in about 100 words: {}".format(content_text)
    return prompt

# def gpt_summarize_process(input_text):
#     prompt = gpt_summarize_prompt_construct(input_text)
#     print("prompt:", prompt)
#     gpt_response = gpt_api_send(prompt)
#     print(gpt_response)
#     response_text = str(gpt_response['choices'][0]['text']).replace("\n", "")
#     # print("-------- GPT result --------")
#     print(response_text)
#     # print("----------------------------")
#     return response_text

# if __name__ == "__main__":
    # inputText = "nice to meet you"
    # result = shorten_prompt_length(inputText, 2)
    # print(result)