import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import math

class Metric():
    '''评价指标 precision recall f1'''
    # def __init__(self, args, stop_words, tokenized, ids, predictions, goldens, sen_lengths, tokens_ranges, ignore_index=-1, logging=print):
        # metric = Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
        # print([i.sum() for i in predictions], [i.sum() for i in goldens])
    def __init__(self, args, stop_words, tokenized, ids, predictions, goldens, sen_lengths, 
                 tokens_ranges, intensities, predicted_intensities, ignore_index=-1, logging=print):    
        _g = np.array(goldens)
        _g[_g==-1] = 0
        
        print('sum_pred: ', np.array(predictions).sum(), ' sum_gt: ', _g.sum())
        self.args = args
        self.predictions = predictions
        self.goldens = goldens
        # self.bert_lengths = bert_lengths
        self.sen_lengths = sen_lengths
        self.tokens_ranges = tokens_ranges
        self.ignore_index = ignore_index
        self.data_num = len(self.predictions)
        self.ids = ids
        self.stop_words = stop_words
        self.tokenized = tokenized
        self.logging = logging
        
        self.intensities = intensities  # 真實值
        self.predicted_intensities = predicted_intensities  # 預測值
        self.epochcount = 0
        
        # print(np.array(all_intensities).shape) # (907, 1, 2)
        # print(np.array(all_intensity_logits).shape) # (907, 80, 80, 2)

        

    def get_spans(self, tags, length, token_range, type):
        spans = []
        start = -1
        for i in range(length):
            l, r = token_range[i]
            if tags[l][l] == self.ignore_index:
                continue
            elif tags[l][l] == type:
                if start == -1:
                    start = i
            elif tags[l][l] != type:
                if start != -1:
                    spans.append([start, i - 1])
                    start = -1
        if start != -1:
            spans.append([start, length - 1])
        return spans

    def find_triplet_golden(self, tag):
        triplets = []
        # print(f"調試 tag 矩陣：\n{tag}")

        for row in range(1, tag.shape[0]-1):
            for col in range(1, tag.shape[1]-1):
                if row==col:
                    pass
                elif tag[row][col] in self.args.sentiment2id.values():
                    # print(f"匹配成功：row={row}, col={col}, value={tag[row][col]}")

                    sentiment = int(tag[row][col])
                    al, pl = row, col
                    ar = al
                    pr = pl
                    while tag[ar+1][pr] == 1:
                        ar += 1
                    while tag[ar][pr+1] == 1:
                        pr += 1

                    triplets.append([al, ar, pl, pr, sentiment])
                    # print(f"匹配 triplet: {[al, ar, pl, pr, sentiment]}")

                # print(triplets)
                # [[1, 3, 6, 6, 5]]
                # [[1, 3, 6, 6, 5], [9, 11, 13, 13, 5]]
                # [[1, 3, 6, 6, 5], [9, 11, 13, 13, 5], [16, 16, 15, 15, 5]]
                
        return triplets
    # 調試：對應 golden_tuple 與 find_triplet 結果
    def compare_triplets(self, golden_tuple, tokenized, tokens_ranges):
        for i in range(len(golden_tuple)):
            print(f"golden_tuple:{golden_tuple[i]}")
            al, ar, pl, pr, sentiment = golden_tuple[i]
            print(f"golden_tuple[i]:{golden_tuple[i]}")
            aspect_range = tokens_ranges[i][al:ar ]
            opinion_range = tokens_ranges[i][pl:pr ]
            print(f"對應檢查:")
            print(f"Aspect words: {tokenized[al:ar]}, Expected range: {aspect_range}")
            print(f"Opinion words: {tokenized[pl:pr]}, Expected range: {opinion_range}")

    # def find_triplet_golden(self, tag):
    #     triplets = []
    #     for row in range(tag.shape[0]):
    #         for col in range(tag.shape[1]):
    #             if tag[row][col] in self.args.sentiment2id.values():
    #                 # print(f"匹配成功：row={row}, col={col}, value={tag[row][col]}")

    #                 sentiment = tag[row][col]
    #                 triplets.append([row, row, col, col, sentiment])
    #     return triplets

    
    def find_triplet(self, id , tag, ws, tokenized ,intensity_pred_matrix):
        triplets = []
        # print(f"調試：tag 矩陣\n{tag}")
        # print(f"調試：tokens_ranges={ws}")
        # print(f"調試：tokenized sentence={tokenized}")

        for row in range(1, tag.shape[0]-1):
            for col in range(1, tag.shape[1]-1):
                if row==col:
                    pass
                elif tag[row][col] in self.args.sentiment2id.values():
                    sentiment = int(tag[row][col])
                    al, pl = row, col
                    ar = al
                    pr = pl
                    while tag[ar+1][pr] == 1:
                        ar += 1
                    while tag[ar][pr+1] == 1:
                        pr += 1
                    
                    '''filting the illegal preds'''
                    #目標： 確保 (al, ar) 和 (pl, pr) 對應的索引落在 ws (word spans) 的範圍內
                    condition1 = al in np.array(ws)[:, 0] and ar in np.array(ws)[:, 1] and pl in np.array(ws)[:, 0] and pr in np.array(ws)[:, 1]
                    
                    #目標： 確保 aspect 和 opinion 的範圍沒有交疊。
                    condition2 = True
                    for ii in range(al, ar+1):
                        for jj in range(pl, pr+1):
                            if ii == jj:
                                condition2 = False
                                
                    #目標： 確保 aspect 的 tokens 不包含停用詞（stop_words）
                    # condition3 = True
                    # for tk in tokenized[al: ar+1]:
                    #     # print(tk)
                    #     if tk in self.stop_words:
                    #         condition3 = False
                    #         break
                    
                    #目標： 確保 opinion 的 tokens 不包含停用詞。
                    # condition4 = True
                    # for tk in tokenized[pl: pr+1]:
                    #     # print(tk)
                    #     if tk in self.stop_words:
                    #         condition4 = False
                    #         break

                    # conditions = condition1 and condition2 and condition3 and condition4                        
                    conditions = condition1 and condition2

                    if conditions:
                        # print(f"Triplet found: al={al}, ar={ar}, pl={pl}, pr={pr}, sentiment={sentiment}")
                        # print(f"Aspect range: {tokenized[al:ar+1]}, Opinion range: {tokenized[pl:pr+1]}")
                        sub_matrix_0 = intensity_pred_matrix[al:ar+1, pl:pr+1 , 0]
                        sub_matrix_1 = intensity_pred_matrix[al:ar+1, pl:pr+1 , 1]
                        pred_v = int(round(sub_matrix_0.mean().item() ))
                        pred_a = int(round(sub_matrix_1.mean().item() ))
                        triplets.append([ id , al, ar, pl, pr, sentiment,pred_v ,pred_a])
                        # print(f"pred_v: {pred_v}, pred_a: {pred_a}")

                        # triplets.append([al, ar, pl, pr, sentiment])

                # print(triplets)
                # [[1, 3, 6, 6, 5]]
                # [[1, 3, 6, 6, 5], [9, 11, 13, 13, 5]]
                # [[1, 3, 6, 6, 5], [9, 11, 13, 13, 5], [16, 16, 15, 15, 5]]
        if not triplets:
            return triplets

        # self.compare_triplets(triplets, tokenized, ws)        
        return triplets
    
    # def get_sets(self):
    #     assert len(self.predictions) == len(self.goldens)
    #     golden_set = set()
    #     predicted_set = set()
    #     for i in range(self.data_num):
    #         # golden_aspect_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
    #         # golden_opinion_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
    #         id = self.ids[i]
    #         golden_tuples = self.find_triplet_golden(np.array(self.goldens[i]))
    #         # golden_tuples: triplets.append([al, ar, pl, pr, sentiment])
    #         for golden_tuple in golden_tuples:
    #             golden_set.add(id + '-' + '-'.join(map(str, golden_tuple)))  # 从前到后把得到的三元组纳入总集合
    #             # golden_set: ('0-{al}-{ar}-{pl}-{pr}-{sentiment}', '1-{al}-{ar}-{pl}-{pr}-{sentiment}', '2-{al}-{ar}-{pl}-{pr}-{sentiment}')

    #         # predicted_aspect_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
    #         # predicted_opinion_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
    #         # if self.args.task == 'pair':
    #         #     predicted_tuples = self.find_pair(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
    #         # elif self.args.task == 'triplet':
                
    #         tag = np.array(self.predictions[i])

    #         tag[0][:] = -1
    #         tag[-1][:] = -1
    #         tag[:, 0] = -1
    #         tag[:, -1] = -1

    #         predicted_triplets = self.find_triplet(tag, self.tokens_ranges[i], self.tokenized[i])  # , predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i]
    #         for pair in predicted_triplets:
    #             predicted_set.add(id + '-' + '-'.join(map(str, pair)))
    #     return predicted_set, golden_set
    def predicted_triplets_filter(self, triplets, tokenized_sentence, min_span_length=2):
        """
        過濾同一個 ID 下的多個 triplets。
        
        :param triplets: triplets 列表，每個 triplet 是列表 [id ,al, ar, pl, pr, sentiment, v, a]
        :param min_span_length: 最小 span 長度，預設為 2
        :return: 過濾後的 triplets 列表
        """
        filtered = []
        occupied_aspect = set()
        occupied_opinion = set()
        # 按照綜合 span 長度（aspect_len + opinion_len）降序排序
        
        triplets_sorted = sorted(
            triplets,
            key=lambda x: (x[2] - x[1] + 1) + (x[4] - x[3] + 1),
            reverse=True
        )
        for triplet in triplets_sorted:
            id , al, ar, pl, pr, sentiment, v, a = triplet
            
            aspect_len = ar - al + 1
            opinion_len = pr - pl + 1
            # print(f"triplet: {triplet}")
            # print(f"aspect_len: {aspect_len}")
            # print(f"opinion_len: {opinion_len}")
            # 1. 過濾 aspect opinion 都是單個字 
            if aspect_len < min_span_length and opinion_len < min_span_length:
                continue

            # 最長的被優先考慮  aspect 或 op 有數字不行

            # 2. 避免重疊的 triplets
            # 檢查 aspect 是否與已存在的 aspect 重疊
            if any(pos in occupied_aspect for pos in range(al, ar + 1)):
                continue
            # 檢查 opinion 是否與已存在的 opinion 重疊
            if any(pos in occupied_opinion for pos in range(pl, pr + 1)):
                continue

            # 如果沒有重疊，則保留 triplet 並標記占用的位置
            filtered.append(triplet)
            occupied_aspect.update(range(al, ar + 1))
            occupied_opinion.update(range(pl, pr + 1))
        
        return filtered

    #改動起點
    def get_sets(self):
        assert len(self.predictions) == len(self.goldens)
        self.epochcount =  self.epochcount +1

        p_predicted_set = []
        p_pred_temp_set = set()
        golden_set = set()
        predicted_set = set()
        # 使用字典將 triplets 按 ID 分組
        p_golden_dict = defaultdict(list)
        p_predicted_dict = defaultdict(list)

        print(f"self.data_num: {self.data_num}, len(self.intensities): {len(self.intensities)}")

        # 針對每句 token做
        for i in range(self.data_num):
            id_  = self.ids[i]
            tokenized_sentence = self.tokenized[i]
            golden_tuples = self.find_triplet_golden(np.array(self.goldens[i]))

            # 處理 golden triplets
            for triplet_index, golden_tuple in enumerate(golden_tuples):
                if triplet_index < len(self.intensities[i]):
                    intensity_values = self.intensities[i][triplet_index]  # 已經是列表
                    
                else:
                    intensity_values = [0.0, 0.0]  # 如果沒有 intensity，使用預設值
                intensity_values = [math.floor(x + 0.5) for x in intensity_values]
                unique_str = f"{id_}-{'-'.join(map(str, golden_tuple))}-{'-'.join(map(str, intensity_values))}"
                golden_set.add(unique_str)


            # 獲取 predicted triplets，並新增 Intensity
            tag = np.array(self.predictions[i])
            tag[0, :] = -1
            tag[-1, :] = -1
            tag[:, 0] = -1
            tag[:, -1] = -1

            predicted_triplets = self.find_triplet(id_, tag, self.tokens_ranges[i], self.tokenized[i], np.array(self.predicted_intensities[i]))
            
            for pair in predicted_triplets:
                p_pred_temp_set.add('-'.join(map(str, pair)))
            
            # 將 predicted_triplets 按 ID 分組
            for triplet in predicted_triplets:
                p_predicted_dict[id_].append(triplet)

        if self.epochcount >150:
            print(f"Go filter ! epoch{self.epochcount}")
        
            # 遍歷 predicted_dict，應用過濾規則並更新 predicted_set 和 p_predicted_set
            for id_, triplets in p_predicted_dict.items():
                if len(triplets) > 1:
                    print(f"id_: {id_}, triplets: {triplets}")
                    print(f"number of triplets: {len(triplets)}")

                    print("Go filter!")
                    # 只對有多個 triplets 的 ID 進行過濾
                    filtered_triplets = self.predicted_triplets_filter(triplets, self.tokenized[self.ids.index(id_)])
                    print(f"filtered_triplets: {filtered_triplets}")

                else:
                    # 如果只有一個 triplet，直接保留
                    # print(f"triplets in else: {triplets}")
                    filtered_triplets = triplets
                    # print(f"filtered_triplets: {filtered_triplets}")

                for triplet in filtered_triplets:
                    # print(f"filtered_triplets's item: {triplet}")
                    # triplet has [id, al, ar, pl, pr, sentiment, v, a]
                    # 創建唯一標識符
                    unique_str = f"{triplet[0]}-{triplet[1]}-{triplet[2]}-{triplet[3]}-{triplet[4]}-{triplet[5]}-{triplet[6]}-{triplet[7]}"
                    predicted_set.add(unique_str)
                    # 添加到 p_predicted_set
                    p_predicted_set.append({
                        'id': triplet[0],
                        'aspect_indices': (triplet[1], triplet[2]),
                        'opinion_indices': (triplet[3], triplet[4]),
                        'sentiment': triplet[5],
                        'intensity': [triplet[6], triplet[7]]
                    })
        else:
            predicted_set =  p_pred_temp_set

        print(f"Total Golden Triplets after filtering: {len(golden_set)}")
        print(f"Total Predicted Triplets before filtering: {len(p_pred_temp_set)}")
        print(f"Total Predicted Triplets after filtering: {len(predicted_set)}")

        return p_predicted_set, predicted_set, golden_set 


    def extract_and_print_triplets(self, predicted_set):
        sentiment_map = {2: 'negative', 3: 'neutral', 4: 'positive'}
        
        for triplet_info in predicted_set:
            id = triplet_info['id']
            aspect_indices = triplet_info['aspect_indices']
            opinion_indices = triplet_info['opinion_indices']
            sentiment = triplet_info['sentiment']
            intensity = triplet_info['intensity']
            
            # Get the tokenized sentence
            tokenized_sentence = self.tokenized[self.ids.index(id)]
            # Extract aspect and opinion words
            aspect_words = tokenized_sentence[aspect_indices[0]:aspect_indices[1]+1]
            opinion_words = tokenized_sentence[opinion_indices[0]:opinion_indices[1]+1]
            
            # Convert intensity values to strings
            intensity_str = '#'.join([f"{val:.2f}" for val in intensity])
            
            # Format and print the triplet
            aspect = ''.join(aspect_words)
            opinion = ''.join(opinion_words)
            sentiment_label = sentiment_map.get(sentiment, 'unknown')
            
            # print(f"id:{id}, tokenized_sentence{tokenized_sentence}")
            # print(f"aspect_indices{aspect_indices}")
            # print(f"opinion_indices{opinion_indices}")
            # print(f"sentiment{sentiment}")
            # print(f"intensity{intensity}")
            
            # print(f"{id}: ({aspect}, {opinion}, {intensity_str})")


    def score_triplets(self, predicted_set, golden_set):
        predicted_set = set(['-'.join(i.split('-')[0: 6]) for i in predicted_set])
        golden_set = set(['-'.join(i.split('-')[0: 6]) for i in golden_set])
       
        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # self.logging('Triplet\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(precision, recall, f1))
        return precision, recall, f1

    def score_pairs(self, predicted_set, golden_set):
        predicted_set = set(['-'.join(i.split('-')[0: 5]) for i in predicted_set])
        golden_set = set(['-'.join(i.split('-')[0: 5]) for i in golden_set])
        
        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # self.logging('Pair\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(precision, recall, f1))
        return precision, recall, f1
    
    def score_aspect(self, predicted_set, golden_set):
        predicted_set = set(['-'.join(i.split('-')[0: 3]) for i in predicted_set])
        golden_set = set(['-'.join(i.split('-')[0: 3]) for i in golden_set])

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # self.logging('Aspect\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(precision, recall, f1))
        return precision, recall, f1

    def score_opinion(self, predicted_set, golden_set):
        predicted_set = set([i.split('-')[0] + '-' + ('-'.join(i.split('-')[3: 5])) for i in predicted_set])
        golden_set = set([i.split('-')[0] + '-' + ('-'.join(i.split('-')[3: 5])) for i in golden_set])

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # self.logging('Opinion\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(precision, recall, f1))
        return precision, recall, f1

    def score_triplets_intensity(self, predicted_set, golden_set):
        # 对 predicted_set 进行处理：取 0:5 和 6:8 的组合
        predicted_set = set([('-'.join(i.split('-')[0:5])) + '-' + ('-'.join(i.split('-')[6:8])) for i in predicted_set])

        # 对 golden_set 进行处理：取 0:5 和 6:8 的组合
        golden_set = set([('-'.join(i.split('-')[0:5])) + '-' + ('-'.join(i.split('-')[6:8])) for i in golden_set])

        # 获取 predicted_set 的前 5 个元素的 ID
        predicted_matched = sorted(list(predicted_set))[:5]
        predicted_ids = [i.split('-')[0] for i in predicted_matched]

        # 根据 predicted_ids 从 golden_set 中找到匹配的内容
        golden_matched = [item for item in golden_set if item.split('-')[0] in predicted_ids]

        # 打印结果
        print("Predict Matched (前 5 个):", predicted_matched)
        print("Golden Matched (匹配的内容):", golden_matched)

        # 确定正确匹配的数量
        correct_num = len(golden_set & predicted_set)

        # Precision: 正确预测的占比
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0

        # Recall: 真实标签中正确预测的占比
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0

        # F1 Score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1


