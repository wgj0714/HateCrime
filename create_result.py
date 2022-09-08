import pickle as pkl
import pandas as pd

with open("./Data/hate/data.pkl", "rb") as f:
    train, test, dev, vocab, embedding = pkl.load(f)

# with open("./Data/hate/targets.pkl", "rb") as f:
#     data = pkl.load(f)
#     print(data)
    # #train, test, dev, vocab, embedding = pkl.load(f)
    # #print(vocab, len(vocab))
    # print(train[0]) 
    # #print(len(test), len(train))
    # print(train[0]['article'][10])   # 1번째 기사에서 10번째 문장 가져오기
    # #print(train[0]['article'][12])

with open("./Data/hate/predict.pkl","rb") as f:
    pred_data = pkl.load(f)
vocab_dic = {key:val for key,val in enumerate(vocab)}

predict_list = []

for p in pred_data:
    best_index = p['best_sent']
    k_holder = []
    for bi in best_index:
        best_sentence = p['article'][bi]
        sent = ''
        for idx in best_sentence:
            sent += ' ' + vocab_dic[idx]
        k_holder.append(sent)
    predict_list.append(k_holder)

print(len(predict_list))
target_names = ["race", "nationality", "gender", "religion", "sexual orientation", "ideology", "political identiﬁcation", "mental/physical health"]
action_names = ["assault", "arson", "vandalism", "hate demonstration"]
target_list = pkl.load(open("Data/hate/targets.pkl", "rb"))
action_list = pkl.load(open("Data/hate/actions.pkl", "rb")) 

print(target_list)
print("-"*10)
print(action_list)
print(predict_list[0])

# new_dic = {"preds": predict_list, "target": target_list, "action": action_list}

new_dic = {'pred_1':[], 'pred_2':[], 'target':[], 'action':[]}

for row_idx, item in enumerate(predict_list):
    pred_1 = item[0]
    pred_2 = item[1]
    target = target_names[target_list[row_idx]]
    action = action_names[action_list[row_idx]]
    new_dic['pred_1'].append(pred_1)
    new_dic['pred_2'].append(pred_2)
    new_dic['target'].append(target)
    new_dic['action'].append(action)

print(len(new_dic))

df = pd.DataFrame(new_dic)

print(df.head())

df.to_csv("prediction_result_w_names.csv", encoding="utf8", index=False)
