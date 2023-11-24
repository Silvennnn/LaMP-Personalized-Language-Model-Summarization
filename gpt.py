import requests


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
    return response.json()


def gpt_process(input_x, profiles, task=None):
    if not task:
        print("Task Error")
    print("-------- Sending GPT Request --------")
    print("-------- Input: {} -------- ".format(input_x))
    prompt = gpt_prompt_construct(input_x, profiles, task)
    gpt_response = gpt_api_send(prompt)
    response_text = str(gpt_response['choices'][0]['text']).replace("\n", "")
    print("-------- GPT result --------")
    print(response_text)
    print("----------------------------")
    return response_text


def gpt_prompt_construct(input_x, profiles, task):
    if task == "news_headline":
        prompt = "Generate a headline for the following article: {}\n".format(input_x)
        for profile in profiles:
            profile_line = '"{}" is the title for "{}"\n'.format(profile['title'], profile['text'])
            prompt += profile_line
    elif task == "scholarly_title":
        prompt = "Generate a title for the following abstract of a paper: {}\n".format(input_x)
        for profile in profiles:
            profile_line = '"{}" is the title for "{}"\n'.format(profile['title'], profile['text'])
            prompt += profile_line
    return prompt

if __name__ == "__main__":
    print("hello")