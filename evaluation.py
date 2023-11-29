import evaluate

# def evaluation(expected_dict, result_dict):
#     ROUGE_1 = 0
#     ROUGE_L = 0
#     count = 0
#     for queryID, output_title in result_dict.items():
#         count += 1
#         expected_output = expected_dict[queryID]
#         rouge = Rouge()
#         # scores = rouge.get_scores(hypothesis, reference)
#         scores = rouge.get_scores(output_title, expected_output)[0]
#         cur_rouge_1 = scores["rouge-1"]
#         cur_rouge_1_f1 = cur_rouge_1["f"]
#         cur_rouge_l = scores["rouge-l"]
#         cur_rouge_l_f1 = cur_rouge_l["f"]
#
#         ROUGE_1 +=  cur_rouge_1_f1
#         ROUGE_L += cur_rouge_l_f1
#         print(cur_rouge_1_f1)
#
#     return ROUGE_1 / count, ROUGE_L / count


def evaluation(expected_dict, result_dict):
    rouge = evaluate.load('rouge')
    preds_lst = []
    labels_lst = []
    for queryID, preds_title in result_dict.items():
        # count += 1
        expected_title = expected_dict[queryID]
        preds_lst.append(preds_title.strip())
        labels_lst.append(expected_title.strip())

    result_rouge = rouge.compute(predictions=preds_lst, references=labels_lst)
    result = {"rouge-1" : result_rouge["rouge1"], "rouge-L" : result_rouge["rougeL"]}
    return result
