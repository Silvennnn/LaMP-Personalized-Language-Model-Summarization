import json
import os.path


def news_headline_input_json_data_processor(inputFilePath):
    input_dict = {}
    inputFile = open(inputFilePath, 'r')
    inputData = json.load(inputFile)

    for _, element in enumerate(inputData):
        queryId = element['id']
        queryInput = element["input"]
        queryProfile = element["profile"] # List of profile [{"text", "title", "id"}]
        input_dict[queryId] = {
            "queryInput": queryInput, # Query Input
            "queryProfile": queryProfile # List of profile [{"text", "title", "id"} ... ]
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


def scholarly_title_input_json_data_processor(inputFilePath):
    input_dict = {}
    inputFile = open(inputFilePath, 'r')
    inputData = json.load(inputFile)

    for _, element in enumerate(inputData): # List of query [{"id", "input", "profile"}]
        queryId = element['id']
        queryInput = element["input"]
        queryProfile = element["profile"] # List of profile [{"title", "abstract", "id"}]
        input_dict[queryId] = {
            "queryInput": queryInput,  # Query Input
            "queryProfile": queryProfile  # List of profile [{"title", "abstract", "id"}]
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

def main(task_name="news_headline"):
    print('----------- data loading ------------')
    if task_name == "news_headline":
        data_dir = "Data/news_headline"
        train_input_path = os.path.join(data_dir, "train_questions.json")
        train_output_path = os.path.join(data_dir, "train_outputs.json")
        train_input_data_dict = news_headline_input_json_data_processor(train_input_path)
        train_output_data_dict = news_headline_output_json_data_processor(train_output_path)
    elif task_name == "scholarly_title":
        data_dir = "Data/scholarly_title"
        train_input_path = os.path.join(data_dir, "train_questions.json")
        train_output_path = os.path.join(data_dir, "train_outputs.json")
        train_input_data_dict = scholarly_title_input_json_data_processor(train_input_path)
        train_output_data_dict = scholarly_title_output_json_data_processor(train_output_path)
    else:
        print("Task Not Exist")

    assert len(train_input_data_dict.keys()) == len(train_output_data_dict.keys())
    print('Train_input_data: {}'.format(len(train_input_data_dict.keys())))
    print('Train_output_data: {}'.format(len(train_output_data_dict.keys())))
    print('----------- data loading completed ------------')

if __name__ == "__main__":
    task_name = "news_headline" # news_headline or scholarly_title
    print("************ Task: {} ************ Begin".format(task_name))
    main(task_name)
    print("************ Task: {} End ************".format(task_name))