import json
import math
import os.path

import gpt


def news_headline_input_json_data_processor(inputFilePath):
    input_dict = {}
    inputFile = open(inputFilePath, 'r')
    inputData = json.load(inputFile)

    for _, element in enumerate(inputData):
        queryId = element['id']
        queryInput = element["input"]
        queryProfile = element["profile"] # List of profile [{"text", "title", "id"}]
        queryProfile_dict = {}
        for profile in element["profile"]:
            profile_id = profile["id"]
            profile_title = profile["title"]
            profile_text = profile["text"]
            queryProfile_dict[profile_id] = {}
            queryProfile_dict[profile_id]["title"] = profile_title
            queryProfile_dict[profile_id]["text"] = profile_text


        input_dict[queryId] = {
            "queryInput": queryInput, # Query Input
            "queryProfile": queryProfile, # List of profile [{"text", "title", "id"} ... ]
            "queryProfileDict": queryProfile_dict # Dict of profile
        }
    return input_dict

def news_headline_output_json_data_processor(outputFilePath):
    output_dict = {}
    outputFile = open(outputFilePath, 'r')
    outputData = json.load(outputFile)["golds"] # List of query output [{"id", "output"} ... ]

    for _, element in enumerate(outputData):
        queryId = element['id']
        queryOutput = element["output"]
        output_dict[queryId] = queryOutput
    return output_dict

def news_headline_query_generate(input):
    query = "Generate a headline for the following article: {}".format(input)
    return query


def scholarly_title_input_json_data_processor(inputFilePath):
    input_dict = {}
    inputFile = open(inputFilePath, 'r')
    inputData = json.load(inputFile)

    for _, element in enumerate(inputData): # List of query [{"id", "input", "profile"}]
        queryId = element['id']
        queryInput = element["input"]
        queryProfile = element["profile"] # List of profile [{"title", "abstract", "id"}]
        queryProfile_dict = {}
        for profile in element["profile"]:
            profile_id = profile["id"]
            profile_title = profile["title"]
            profile_text = profile["abstract"]
            queryProfile_dict[profile_id] = {}
            queryProfile_dict[profile_id]["title"] = profile_title
            queryProfile_dict[profile_id]["text"] = profile_text

        input_dict[queryId] = {
            "queryInput": queryInput,  # Query Input
            "queryProfile": queryProfile,  # List of profile [{"title", "abstract", "id"}]
            "queryProfileDict": queryProfile_dict  # Dict of profile
        }
    return input_dict


def scholarly_title_output_json_data_processor(outputFilePath):
    output_dict = {}
    outputFile = open(outputFilePath, 'r')
    outputData = json.load(outputFile)["golds"]  # List of query output [{"id", "output"} ... ]

    for _, element in enumerate(outputData):
        queryId = element['id']
        queryOutput = element["output"]
        output_dict[queryId] = queryOutput
    return output_dict


def scholarly_title_query_generate(input):
    query = "Generate a title for the following abstract of a paper: {}".format(input)
    return query

def BM25_index_prepare(profiles):
    """
    :param profiles: list of profile [{"text", "title", "id"} ... ]
    :return: index: dict {
                        "term": {
                            "profileID": count
                        }
                    },
            total_term_count: number of terms in profiles list,
            avg_profile_length: average length of profile's text,
            profile_length_dict: dict {
                                            "profileID": length of text
                                        }
    """
    index = {}
    total_term_count = 0
    profile_length_dict = {}
    for profile in profiles:
        profile_id = profile["id"]
        title = profile["title"]
        text = profile["text"]
        terms_lst = text.split()
        profile_length_dict[profile_id] = len(terms_lst)
        total_term_count += len(terms_lst)
        for term in terms_lst:
            if term not in index: # if the term not exist in the index
                index[term] = {}
                index[term][profile_id] = 1
                index[term]["total"] = 1
            else: # if the term already exist
                if profile_id not in index[term]: # if the doc_id not in index[term] --> initialize to 1
                    index[term][profile_id] = 1
                    index[term]["total"] = 1
                else:
                    index[term][profile_id] += 1
                    index[term]["total"] += 1
    avg_profile_length = total_term_count / len(profile)
    return index, total_term_count, avg_profile_length, profile_length_dict


def BM25(query, profiles_lst, profiles_dict, k1=1.2, b=0.75):
    """
    :param profiles_dict: dict of profiles {profileID: {"text":str, "title": str}}
    :param b: int
    :param query: string
    :param profiles_lst: list of profiles
    :param k1: int
    :return: result: list of sorted tuples [(profileID, bm25 score)]
    """
    bm25_score_dict = {}
    index, total_term_count, avg_profile_length, profile_length_dict = BM25_index_prepare(profiles_lst)
    for query_term in query.split():
        for profile in profiles_lst:
            profile_id = profile["id"]
            if query_term in index:
                df_t = index[query_term]["total"]
                if profile_id in index[query_term]: # if the profile contain the term
                    tf_t_d = index[query_term][profile_id]
                else: # if not exist
                    tf_t_d = 0
            else: # if not exist
                df_t = 0
                tf_t_d = 0
            d_length = profile_length_dict[profile_id]
            N = total_term_count

            first_part = ((k1 + 1) * tf_t_d)/(k1 * (1 - b + b * (d_length / avg_profile_length)) + tf_t_d)
            second_part = math.log((N - df_t + 0.5) / (df_t + 0.5))
            cur_score = first_part * second_part
            # print(query_term, profile_id, cur_score)
            if profile_id not in bm25_score_dict:
                bm25_score_dict[profile_id] = cur_score
            else:
                bm25_score_dict[profile_id] += cur_score
    bm25_score_lst = list(bm25_score_dict.items())
    bm25_score_lst.sort(key= lambda x: x[1], reverse=True)
    return bm25_score_lst


def main(task_name="news_headline", k=1):
    print('----------- data loading ------------')
    if task_name == "news_headline":
        data_dir = "Data/news_headline"
        train_input_path = os.path.join(data_dir, "dev_questions.json")
        train_output_path = os.path.join(data_dir, "dev_outputs.json")
        train_input_data_dict = news_headline_input_json_data_processor(train_input_path)
        train_output_data_dict = news_headline_output_json_data_processor(train_output_path)
    elif task_name == "scholarly_title":
        data_dir = "Data/scholarly_title"
        train_input_path = os.path.join(data_dir, "dev_questions.json")
        train_output_path = os.path.join(data_dir, "dev_outputs.json")
        train_input_data_dict = scholarly_title_input_json_data_processor(train_input_path)
        train_output_data_dict = scholarly_title_output_json_data_processor(train_output_path)
    else:
        print("Task Not Exist")

    assert len(train_input_data_dict.keys()) == len(train_output_data_dict.keys())
    print('Train_input_data: {}'.format(len(train_input_data_dict.keys())))
    print('Train_output_data: {}'.format(len(train_output_data_dict.keys())))
    print('----------- data loading completed ------------')

    query_310 = train_input_data_dict["310"]
    print(query_310.keys())
    query_310_input = news_headline_query_generate(query_310["queryInput"])
    query_310_queryProfileList = query_310["queryProfile"]
    query_310_queryProfileDict = query_310["queryProfileDict"]
    query_310_BM25_result = BM25(query_310_input, query_310_queryProfileList, query_310_queryProfileDict)
    print(query_310_BM25_result)
    query_310_selected_profile = []
    for i in range(k):
        profile_id = query_310_BM25_result[i][0]
        query_310_selected_profile.append(query_310_queryProfileDict[profile_id])
    query_310_gpt_result = gpt.gpt_process(query_310_input, query_310_selected_profile, task=task_name)




if __name__ == "__main__":
    task_name = "news_headline" # news_headline or scholarly_title
    print("************ Task: {} Begin ************ ".format(task_name))
    main(task_name)
    print("************ Task: {} End ************".format(task_name))