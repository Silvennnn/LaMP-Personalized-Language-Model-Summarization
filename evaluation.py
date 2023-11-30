import evaluate


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
