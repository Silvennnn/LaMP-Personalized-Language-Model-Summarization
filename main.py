import json
import math
import os.path
import time

import evaluation
import gpt
from transformers import T5Tokenizer, T5ForConditionalGeneration

def summary_file_loader(summaryFilePath):
    summaryFile = open(summaryFilePath, 'r')
    summary_dict = {} # key = queryID, value = list of [{"text", "title"}]
    for line in summaryFile.readlines():
        elements = line.split('\t')
        queryId = elements[0]
        profileId = elements[1]
        summaryTitle = elements[2]
        summaryText = elements[3]
        dict_item = {"profileId": profileId, "text": summaryText, "title": summaryTitle}
        if queryId not in summary_dict:
            summary_dict[queryId] = [dict_item]
        else:
            summary_dict[queryId].append(dict_item)
    return summary_dict
def news_headline_input_json_data_processor(inputFilePath):
    input_dict = {}
    inputFile = open(inputFilePath, 'r')
    inputData = json.load(inputFile)

    for index, element in enumerate(inputData):
        # if index == 1: # TODO: Remove
        #     break
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

    for index, element in enumerate(outputData):
        # if index == 1000: # TODO: Remove
        #     break
        queryId = element['id']
        queryOutput = element["output"]
        output_dict[queryId] = queryOutput
    return output_dict

# def query_task_generator(inputText, task):
#     if task == 'news_headline':
#         return news_headline_query_generate(inputText)
#     elif task == 'scholarly_title':
#         return scholarly_title_query_generate(inputText)
#     else:
#         print("Task Error")
#         return None

def news_headline_query_generate(inputText):
    query = "Generate a headline for the following article: {}".format(inputText)
    return query


def scholarly_title_input_json_data_processor(inputFilePath):
    input_dict = {}
    inputFile = open(inputFilePath, 'r')
    inputData = json.load(inputFile)

    for index, element in enumerate(inputData): # List of query [{"id", "input", "profile"}]
        # if index == 300:  # TODO: Remove
        #     break
        queryId = element['id']
        queryInput = element["input"]
        # print(queryInput)
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

    for index, element in enumerate(outputData):
        # if index == 300:  # TODO: Remove
        #     break
        queryId = element['id']
        queryOutput = element["output"]
        output_dict[queryId] = queryOutput
    return output_dict


def scholarly_title_query_generate(inputText):
    query = "Generate a title for the following abstract of a paper: {}".format(inputText)
    return query

# def BM25_index_prepare_V1(profiles, task):
#     """
#     :param profiles: list of profile [{"text", "title", "id"} ... ]
#     :return: index: dict {
#                         "term": {
#                             "profileID": count
#                         }
#                     },
#             total_term_count: number of terms in profiles list,
#             avg_profile_length: average length of profile's text,
#             profile_length_dict: dict {
#                                             "profileID": length of text
#                                         }
#     """
#     index = {}
#     total_term_count = 0
#     profile_length_dict = {}
#     for profile in profiles:
#         profile_id = profile["id"]
#         if task == "news_headline":
#             title = profile["title"]
#             text = profile["text"]
#         elif task == "scholarly_title":
#             title = profile["title"]
#             text = profile["abstract"]
#         terms_lst = text.split()
#         profile_length_dict[profile_id] = len(terms_lst)
#         total_term_count += len(terms_lst)
#         for term in terms_lst:
#             if term not in index: # if the term not exist in the index
#                 index[term] = {}
#                 index[term][profile_id] = 1
#                 index[term]["total"] = 1
#             else: # if the term already exist
#                 if profile_id not in index[term]: # if the doc_id not in index[term] --> initialize to 1
#                     index[term][profile_id] = 1
#                     index[term]["total"] = 1
#                 else:
#                     index[term][profile_id] += 1
#                     index[term]["total"] += 1
#     avg_profile_length = total_term_count / len(profile)
#     return index, total_term_count, avg_profile_length, profile_length_dict

# def BM25_V1(query, profiles_lst, profiles_dict, k1=1.2, b=0.75, task=None):
#     """
#     :param profiles_dict: dict of profiles {profileID: {"text":str, "title": str}}
#     :param b: int
#     :param query: string
#     :param profiles_lst: list of profiles
#     :param k1: int
#     :return: result: list of sorted tuples [(profileID, bm25 score)]
#     """
#     bm25_score_dict = {}
#     index, total_term_count, avg_profile_length, profile_length_dict = BM25_index_prepare(profiles_lst, task)
#     for query_term in query.split():
#         for profile in profiles_lst:
#             profile_id = profile["id"]
#             if query_term in index:
#                 df_t = index[query_term]["total"]
#                 if profile_id in index[query_term]: # if the profile contain the term
#                     tf_t_d = index[query_term][profile_id]
#                 else: # if not exist
#                     tf_t_d = 0
#             else: # if not exist
#                 df_t = 0
#                 tf_t_d = 0
#             d_length = profile_length_dict[profile_id]
#             N = total_term_count
#
#             first_part = ((k1 + 1) * tf_t_d)/(k1 * (1 - b + b * (d_length / avg_profile_length)) + tf_t_d)
#             second_part = math.log((N - df_t + 0.5) / (df_t + 0.5))
#             cur_score = first_part * second_part
#             # print(query_term, profile_id, cur_score)
#             if profile_id not in bm25_score_dict:
#                 bm25_score_dict[profile_id] = cur_score
#             else:
#                 bm25_score_dict[profile_id] += cur_score
#     bm25_score_lst = list(bm25_score_dict.items())
#     bm25_score_lst.sort(key= lambda x: x[1], reverse=True)
#     return bm25_score_lst

def BM25_preprocess_V2(input_data_dict):
    term_document_frequency_index = {}
    collection_term_count = 0
    total_profile_count = 0
    for query_id, query_elements in input_data_dict.items():
        query_ProfileList = query_elements["queryProfile"] # List of profile [{"title", "abstract", "id"}]
        for profile in query_ProfileList:
            total_profile_count += 1
            abstract = profile["abstract"].split()
            unique_term_in_profile = []

            # count unique term in this document
            for term in abstract:
                collection_term_count += 1
                if term not in unique_term_in_profile:
                    unique_term_in_profile.append(term)

            # add document frequency of term
            for term in unique_term_in_profile:
                if term not in term_document_frequency_index:
                    term_document_frequency_index[term] = 1
                else:
                    term_document_frequency_index[term] += 1

    avg_document_length = collection_term_count / total_profile_count
    return term_document_frequency_index, collection_term_count, avg_document_length

def BM25_index_build_V2(profiles, task):
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
    profile_length_dict = {}
    for profile in profiles:
        profile_id = profile["id"]
        if task == "news_headline":
            title = profile["title"]
            text = profile["text"]
        elif task == "scholarly_title":
            title = profile["title"]
            text = profile["abstract"]
        terms_lst = text.split()
        profile_length_dict[profile_id] = len(terms_lst)
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
    return index, profile_length_dict
def BM25_V2(query, profiles_lst, profiles_dict, avg_doc_length, document_frequency_dict, collection_term_count ,task_name, k1=1.2, b=0.75, ):
    """
    :param task_name:
    :param collection_term_count:
    :param document_frequency_dict:
    :param avg_doc_length:
    :param profiles_dict: dict of profiles {profileID: {"text":str, "title": str}}
    :param b: int
    :param query: string
    :param profiles_lst: list of profiles
    :param k1: int
    :return: result: list of sorted tuples [(profileID, bm25 score)]
    """
    bm25_score_dict = {}
    index, profile_length_dict = BM25_index_build_V2(profiles_lst, task)
    for query_term in query.split():
        for profile in profiles_lst:
            profile_id = profile["id"]

            # term frequency in current profile
            if query_term in index:
                if profile_id in index[query_term]: # if the profile contain the term
                    tf_t_d = index[query_term][profile_id]
                else: # if not exist
                    tf_t_d = 0
            else: # if not exist
                tf_t_d = 0

            # document frequency of query_term
            if query_term in document_frequency_dict:
                df_t = document_frequency_dict[query_term]
            else:
                df_t = 0

            d_length = profile_length_dict[profile_id]
            N = collection_term_count

            first_part = ((k1 + 1) * tf_t_d)/(k1 * (1 - b + b * (d_length / avg_doc_length)) + tf_t_d)
            second_part = math.log((N - df_t + 0.5) / (df_t + 0.5))
            cur_score = first_part * second_part
            # print(query_term, profile_id, cur_score)
            if profile_id not in bm25_score_dict:
                bm25_score_dict[profile_id] = cur_score
            else:
                bm25_score_dict[profile_id] += cur_score
    bm25_score_lst = list(bm25_score_dict.items())
    bm25_score_lst.sort(key= lambda x: x[1], reverse=True)
    # print(bm25_score_lst)
    return bm25_score_lst

def summarization_generate(taskName, outputPath, method='GPT', bm25_top_k=1):
    """
    :param taskName: news_headline or scholarly_title
    :param outputPath: output
    :param method:
    :param bm25_top_k:
    :return:
    """
    if method == 'flan_t5_base':
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")
    else:
        model = None

    print('----------- data loading ------------')
    if taskName == "news_headline":
        data_dir = "Data/news_headline"
        train_input_path = os.path.join(data_dir, "dev_questions.json")
        train_output_path = os.path.join(data_dir, "dev_outputs.json")
        train_input_data_dict = news_headline_input_json_data_processor(train_input_path)
        train_output_data_dict = news_headline_output_json_data_processor(train_output_path)
        lib_path =  'Data/lib/news_headline_{}_lib'.format(method)
    elif taskName == "scholarly_title":
        data_dir = "Data/scholarly_title"
        train_input_path = os.path.join(data_dir, "dev_questions.json")
        train_output_path = os.path.join(data_dir, "dev_outputs.json")
        train_input_data_dict = scholarly_title_input_json_data_processor(train_input_path)
        train_output_data_dict = scholarly_title_output_json_data_processor(train_output_path)
        lib_path = 'Data/lib/scholarly_title_{}_lib'.format(method)
    else:
        print("Task Not Exist")

    assert len(train_input_data_dict.keys()) == len(train_output_data_dict.keys())
    print('Train_input_data: {}'.format(len(train_input_data_dict.keys())))
    print('Train_output_data: {}'.format(len(train_output_data_dict.keys())))
    print('----------- data loading completed ------------')

    # Read Library (If the summary already exist, we don't generate again)
    lib = {}
    with open(lib_path, 'r') as lib_f:
        lines = lib_f.readlines()
        for line in lines:
            elements = line.split('\t')
            queryID = elements[0]
            profileID = elements[1]
            summary = elements[2]
            if queryID not in lib:
                lib[queryID] = {}
                lib[queryID][profileID] = summary
            else:
                lib[queryID][profileID] = summary

    # Summary Generate
    outputFile = open(outputPath, 'a')
    libFile = open(lib_path, 'a')

    # BM25 Preprocess
    term_document_frequency_dict, collection_term_count, avg_document_length = BM25_preprocess_V2(train_input_data_dict)

    for query_id, query_elements in train_input_data_dict.items():
        print("------------ Current Process: {} -----------".format(query_id))
        query_input = query_elements["queryInput"]
        query_ProfileList = query_elements["queryProfile"]
        query_ProfileDict = query_elements["queryProfileDict"]
        query_BM25_result = BM25_V2(query_input, query_ProfileList, query_ProfileDict, avg_document_length, term_document_frequency_dict, collection_term_count, taskName)
        for i in range(bm25_top_k):
            profile_id= query_BM25_result[i][0]
            exist = False
            if query_id in lib:
                if profile_id in lib[query_id]:
                    summary = lib[query_id][profile_id]
                    print("Summary Exist")
                    exist = True

            if not exist:
                input_text = query_ProfileDict[profile_id]['text']
                if method == 'GPT':
                    summary = gpt.gpt_process(input_text, task="summary")
                elif method == 'flan_t5_base':
                    input_text = gpt.gpt_summarize_prompt_construct(input_text)
                    print('Input: ', input_text)
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                    outputs = model.generate(input_ids)
                    summary = tokenizer.decode(outputs[0])
                    print('Output: ', summary)

            title = query_ProfileDict[profile_id]['title']
            line = "{}\t{}\t{}\t{}\n".format(query_id, profile_id, title, summary.replace('\n', ''))
            outputFile.write(line)
            if not exist:
                libFile.write(line)

    print("******************* Summary Complete ********************")




def main(task_name="news_headline", k=1, outputPath=None, language_model="gpt"):
    if language_model == 'flan_t5_base':
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")
    else:
        model = None

    if task_name == 'news_headline' or task_name == 'scholarly_title':
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



        outputFile = open(outputPath, 'r')
        lines = outputFile.readlines()
        lib = {}
        for line in lines:
            elements = line.split('\t')
            # print(elements)
            queryID = elements[0]
            result = elements[1]
            if result:
                lib[queryID] = result
        outputFile.close()
        outputFile = open(outputPath, 'a')
        gpt_result_dict = {}

        print("------------ BM25 Preprocessing Begin -----------")
        # BM25 Preprocess
        term_document_frequency_dict, collection_term_count, avg_document_length = BM25_preprocess_V2(train_input_data_dict)
        print("collection total term count: {}".format(collection_term_count))
        print("average document length: {}".format(avg_document_length))
        print("------------ BM25 Preprocessing End -----------")

        for query_id, query_elements in train_input_data_dict.items():
            print("------------ Current Process: {} -----------".format(query_id))
            if query_id in lib:
                print("{} already exist".format(query_id))
                gpt_result_dict[query_id] = lib[query_id]
                continue
            query_input = query_elements['queryInput']
            query_ProfileList = query_elements["queryProfile"]
            query_ProfileDict = query_elements["queryProfileDict"]
            query_BM25_result = BM25_V2(query_input, query_ProfileList, query_ProfileDict, avg_document_length, term_document_frequency_dict, collection_term_count, task_name) # TODO: replace to dynamic
            # query_BM25_result = BM25_V1(query_input, query_ProfileList, query_ProfileDict, task=task_name)
            # print(query_BM25_result)

            # print(query_elements["queryInput"])
            query_selected_profile = []
            for i in range(k):
                profile_id = query_BM25_result[i][0]
                query_selected_profile.append(query_ProfileDict[profile_id])
            query_gpt_result = gpt.gpt_process(query_input, profiles=query_selected_profile, task=task_name).replace('"', '')
            # time.sleep(2)
            gpt_result_dict[query_id] = query_gpt_result
            line = str(query_id) + '\t' + query_gpt_result + '\n'
            outputFile.write(line)

        print("------------ Begin Evaluation -----------")
        result = evaluation.evaluation(train_output_data_dict, gpt_result_dict)
        print("ROUGE_1: {}".format(result["rouge-1"]))
        print("ROUGE_L: {}".format(result["rouge-L"]))
        print("------------ End Evaluation -----------")

    else: # Summary Task
        print('----------- data loading ------------')
        if task_name == "news_headline_summary":
            summary_data_dir = "Data/summary"
            train_data_dir = "Data/news_headline"

            summary_input_path = os.path.join(summary_data_dir, "news_headline_summary_k={}".format(k))
            summary_input_data_dict = summary_file_loader(summary_input_path)

            train_input_path = os.path.join(train_data_dir, "dev_questions.json")
            train_output_path = os.path.join(train_data_dir, "dev_outputs.json")

            train_input_data_dict = scholarly_title_input_json_data_processor(train_input_path)
            train_output_data_dict = news_headline_output_json_data_processor(train_output_path)
        elif task_name == "scholarly_title_summary":
            summary_data_dir = "Data/summary"
            train_data_dir = "Data/scholarly_title"

            summary_input_path = os.path.join(summary_data_dir, "scholarly_title_summary_k={}".format(k))
            summary_input_data_dict = summary_file_loader(summary_input_path)

            train_input_path = os.path.join(train_data_dir, "dev_questions.json")
            train_output_path = os.path.join(train_data_dir, "dev_outputs.json")

            train_input_data_dict = scholarly_title_input_json_data_processor(train_input_path)
            train_output_data_dict = scholarly_title_output_json_data_processor(train_output_path)
        else:
            print("Task Not Exist")

        outputFile = open(outputPath, 'r')
        lines = outputFile.readlines()
        lib = {}
        for line in lines:
            elements = line.split('\t')
            # print(elements)
            queryID = elements[0]
            result = elements[1]
            if result:
                lib[queryID] = result
        outputFile.close()
        outputFile = open(outputPath, 'a')
        result_dict = {}

        for query_id, summary_profiles_lst in summary_input_data_dict.items():
            print("------------ Current Process: {} -----------".format(query_id))
            if query_id in lib:
                print("{} already exist".format(query_id))
                result_dict[query_id] = lib[query_id]
                continue
            query_input = train_input_data_dict[query_id]["queryInput"]

            if model == 'flan_t5_base':
                input_text = gpt.gpt_title_generate_prompt_construct(query_input, profiles=summary_profiles_lst, task=task_name)
                print('Input: ', input_text)
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                outputs = model.generate(input_ids)
                query_result = tokenizer.decode(outputs[0])
                print('Output: ', query_result)
            else:
                query_result = gpt.gpt_process(query_input, profiles=summary_profiles_lst, task=task_name).replace(
                    '"', '')
            result_dict[query_id] = query_result
            line = str(query_id) + '\t' + query_result + '\n'
            outputFile.write(line)

        print("------------ Begin Evaluation -----------")
        result = evaluation.evaluation(train_output_data_dict, result_dict)
        print("ROUGE_1: {}".format(result["rouge-1"]))
        print("ROUGE_L: {}".format(result["rouge-L"]))
        print("------------ End Evaluation -----------")







if __name__ == "__main__":
    # task = "scholarly_title_summary" # news_headline or scholarly_title or news_headline_summary or scholarly_title_summary
    # LLM = 'flan_t5_base' # model = 'gpt' or 'flan_t5_base'
    #  # ************************* Title Prediction Task ****************************
    # output_path = 'GPT_output'
    # print("************ Task: {} Begin ************ ".format(task))
    # main(task_name=task, k=2, outputPath=output_path, language_model=LLM)
    # print("************ Task: {} End ************".format(task))


    # ************************* Summary Generate Task ****************************
    LLM = 'GPT' # model = 'GPT' or 'flan_t5_base'
    task = "scholarly_title"
    output_path = 'Data/summary/{}_summary_k=1'.format(task)
    summarization_generate(task, output_path, method=LLM, bm25_top_k=1)


