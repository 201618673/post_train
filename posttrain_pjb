from tqdm import tqdm
import sys
import os
sys.path.append(os.getcwd())
import collections
import random
import argparse
import json
from transformers import BertTokenizer, BertForMaskedLM


class PostTrainInstance(object):

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next):

        #학습 인스턴스를 구성하는 토큰 들의 vocab 인덱스
        self.tokens = tokens

        #BERT의 각 segment를 구분하기 위한 인덱스(0 or 1)
        self.segment_ids = segment_ids

        #NSP 문제의 정답(True: negative, False: positive)
        self.is_random_next = is_random_next

        #MLM 문제에서 Masking한 토큰의 위치
        self.masked_lm_positions = masked_lm_positions

        #MLM 문제의 정답(masked_lm_postion 에 대응하는 [MASK] 토큰의 정답)
        self.masked_lm_labels = masked_lm_labels

# 마스킹 된 토큰의 index, label
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

class CreateBertPosttrainingData(object):

    def __init__(self, args):
        self.args = args
        self.bert_tokenizer_init(args.special_tok)

    # 토크나이징 결과에 스페셜 토큰 [EOT] 추가
    def add_special_tokens(self, tokens, special_tok="[EOT]"):
        tokens = tokens + [special_tok]
        return tokens

    def bert_tokenizer_init(self, special_tok):
    
        # 토크나이저 설정
        self.bert_tokenizer = BertTokenizer(vocab_file='/home/parkjb/repository/kNN/BERT-ResSel/data/ubuntu_corpus_v1/vocab.txt')
        # 토크나이저 vocab에 [EOT] 추가
        self.bert_tokenizer.add_tokens([special_tok])

    
    ## 입력 파일들을 문서 단위로 만들고 Tokenize
    def create_training_instances(self, input_file, max_seq_length, dupe_factor, 
                                  short_seq_prob, masked_lm_prob,
                                  max_predictions_per_seq, rng, special_tok=None):
        ## dupe_factor : 중복 요인

        all_documents = [[]]

        with open(input_file, "r", encoding="utf=8") as fr_handle:
            for line in tqdm(fr_handle):
                line = line.strip()

            tokens = self.bert_tokenizer.tokenize(line)

            #special token일 경우 tokenize
            if special_tok:
                tokens = self.add_special_tokens(tokens, special_tok)

            #일반 token일 경우 tokenize
            if tokens:
                all_documents[-1].append(tokens)
        # create_trainiing_instance 함수는 위와 같이 간단하게 각 입력 파일들을 읽어서 Tokenize 과정. 
        # 텍스트 파일의 한 줄에는 하나의 문장이, 다른 문서들 사이에는 빈 라인이 존재해야 함.

        # documents 중 빈 list 삭제
        all_documents = [x for x in all_documents if x]

        # 데이터 shuffle
        rng.shuffle(all_documents)

        # tokenizer vocab : dict -> list
        vocab_words = list(self.bert_tokenizer.vocab.keys())

        # 인스턴스를 구성하는 keys
        self.feature_keys = ["input_ids", "attention_mask", "token_type_ids",
                            "masked_lm_positions", "masked_lm_ids", "next_sentence_labels"]

        datas = []

        for d in range(dupe_factor):
            rng.shuffle(all_documents)

            data = {}
            self.all_doc_feat_dict = dict()
            for feat_key in self.feature_keys:
                self.all_doc_feat_dict[feat_key] = []

            for document_index in tqdm(range(len(all_documents))):
                # 문서들을 이용해서 입력 인스턴스 만들기
                instances = self.create_instances_from_document(
                all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng)

                # 계산한 입력 인스턴스를 input으로 생성
                (input_ids, 
                attention_mask,
                token_type_ids,
                masked_lm_positions,
                masked_lm_ids,
                next_sentence_labels) = self.instance_to_example_feature(instances, self.args.max_seq_length,
                                                                        self.args.max_predictions_per_seq)

                data["input_ids"] = input_ids
                data["attention_mask"] = attention_mask
                data["token_type_ids"] = token_type_ids
                data["masked_lm_positions"] = masked_lm_positions
                data["masked_lm_ids"] = masked_lm_ids
                data["next_sentence_labels"] = next_sentence_labels

                datas.append(data)

            # print("Current Dupe Factor : %d" % (d + 1))
        
            with open(self.args.output_file, "w") as jsn:
                json.dump(datas, jsn, indent=4)


    ## 문서들을 이용해서 입력 인스턴스 만들기 ([CLS] + token_a + [SEP] + token_b + [SEP]) 
    ##                                      (0 + ...0... + 0 + ...1... + 1)
    def create_instances_from_document(self, all_documents, document_index, max_seq_length, short_seq_prob,
                                        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
        
        document = all_documents[document_index]

        # Traget Sequence Length 정하기, -([CLS], [SEP], [EOT])
        max_num_tokens = max_seq_length - 3

        target_seq_length = max_num_tokens
        if rng.random() < short_seq_prob:
            target_seq_length = rng.randint(2, max_num_tokens)

        instances = []
        current_chunk = []
        current_length = 0
        i = 0

        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                # Segment A, Segment B를 구성할 tokens_a, tokens_b를 구하기
                if current_chunk:
                    a_end = 1
                    # `a_end` : `current_chunk` 에서 `A` 로 이동한 segments의 수
                    ## a_end = 1 : 첫번째 문장
                    if len(current_chunk) >= 2:
                        a_end = rng.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
    
                    is_random_next = False

                    if len(current_chunk) == 1 or rng.random() < 0.5:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        for _ in range(10):
                            random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                        random_document = all_documents[random_document_index]
                        random_start = rng.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
    
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    # token_a + token_b 길이 조절     
                    self.truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    tokens = []
                    segment_ids = []

                    ## token [CLS] 의 segment_ids = 0
                    tokens.append("[CLS]")
                    segment_ids.append(0)

                    ## token_a 의 segment_ids = 0
                    for token in tokens_a:
                        tokens.append(token)
                        segment_ids.append(0)

                    ## 중간 token [SEP] 의 segment_ids = 0
                    tokens.append("[SEP]")
                    segment_ids.append(0)

                    ## token_b 의 segment_ids = 1
                    for token in tokens_b:
                        tokens.append(token)
                        segment_ids.append(1)

                    ## 마지막 token [SEP] 의 segment_ids = 1
                    tokens.append("[SEP]")
                    segment_ids.append(1)
                    
                    # 마스킹 된 토큰의 위치, pred, label
                    (tokens, masked_lm_positions, masked_lm_labels) = self.create_masked_lm_predictions(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)

                    ## 인스턴스 선언
                    instance = PostTrainInstance(
                        tokens=tokens,
                        segment_ids=segment_ids,
                        is_random_next=is_random_next,
                        masked_lm_positions=masked_lm_positions,
                        masked_lm_labels=masked_lm_labels)
                    instances.append(instance)

                current_chunk = []
                current_length = 0
            i += 1

        return instances

    # 마스킹 된 토큰의 pred, label
    def create_masked_lm_predictions(self, tokens, masked_lm_prob,  max_predictions_per_seq, vocab_words, rng):

        cand_indexes = []
        for i, token in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":  # [CLS], [SEP]는 마스킹 후보에서 제외
                continue
            
            ## do_whole_word_mask 적용하냐 마냐
            if (self.args.do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
                cand_indexes[-1].append(i)
            else :
                cand_indexes.append(i)

        rng.shuffle(cand_indexes)

        output_tokens = list(tokens)

        # 마스킹할 토큰 개수 -> 토큰 개수 * n
        num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms = []  # 마스킹 된 토큰 객체 (인데스 + 라벨)
        covered_indexes = set()  # 마스킹할 토큰의 인덱스
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:  # 목표치 채웠으면 끝내고
                break
            if index in covered_indexes:  # 이미 마스킹된 index 면 건너뛰기
                continue
            covered_indexes.add(index)  # 해당 Index 마스킹 할 것이다.

            # 0.8의 확률로 마스킹
            if rng.random() < 0.8:
                masked_token = "[MASK]"  
            else:
                # 0.1의 확률로 그대로
                if rng.random() < 0.5:
                    masked_token = tokens[index]  
                # 0.1의 확률로 랜덤 토큰
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]  

            output_tokens[index] = masked_token  # 마스킹된 토큰으로 교체
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))  # 마스킹한 토큰의 인덱스와 라벨 저장

        masked_lms = sorted(masked_lms, key=lambda x: x.index)  # Index를 기준으로 정렬

        masked_lm_positions = []
        masked_lm_labels = []

        for p in masked_lms:
            masked_lm_positions.append(p.index)  # 마스킹한 토큰의 인덱스 저장
            masked_lm_labels.append(p.label)  # 마스킹한 토큰의 라벨 저장

        # (전체 토큰들 - 마스킹 반영, 마스킹 된 토큰의 위치, 마스킹된 토큰의 라벨)
        return (output_tokens, masked_lm_positions, masked_lm_labels)
    
    ## seq A + seq B 가 Target Sequence Length 보다 길면 자르는? 과정
    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens, rng):
       
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break
                
            # 둘 중 긴 것 선택
            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            assert len(trunc_tokens) >= 1


            if rng.random() < 0.5: # 0.5의 확률
                del trunc_tokens[0] # 맨 앞 토큰 제거
            else:
                trunc_tokens.pop() # 맨 뒤 토큰 제거

    # model에 input할 데이터 생성
    def instance_to_example_feature(self, instances, max_seq_length, max_predictions_per_seq):

        for instance in instances:
            input_ids = self.bert_tokenizer.convert_tokens_to_ids(instance.tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = list(instance.segment_ids)

            assert len(input_ids) <= max_seq_length

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            masked_lm_positions = list(instance.masked_lm_positions)
            masked_lm_ids = self.bert_tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
            masked_lm_weights = [1.0] * len(masked_lm_ids)

            while len(masked_lm_positions) < max_predictions_per_seq:
                masked_lm_positions.append(0)
                masked_lm_ids.append(0)
                masked_lm_weights.append(0.0)

            next_sentence_label = 1 if instance.is_random_next else 0

            self.all_doc_feat_dict["input_ids"].append(input_ids)
            self.all_doc_feat_dict["attention_mask"].append(input_mask)
            self.all_doc_feat_dict["token_type_ids"].append(segment_ids)
            self.all_doc_feat_dict["masked_lm_positions"].append(masked_lm_positions)
            self.all_doc_feat_dict["masked_lm_ids"].append(masked_lm_ids)
            self.all_doc_feat_dict["next_sentence_labels"].append([next_sentence_label])

        return (self.all_doc_feat_dict["input_ids"], 
                self.all_doc_feat_dict["attention_mask"],
                self.all_doc_feat_dict["token_type_ids"],
                self.all_doc_feat_dict["masked_lm_positions"],
                self.all_doc_feat_dict["masked_lm_ids"],
                self.all_doc_feat_dict["next_sentence_labels"])


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser(description="Bert / Create Pretraining Data")
  arg_parser.add_argument("--input_file", dest="input_file", type=str,
                          default="/home/parkjb/repository/kNN/BERT-ResSel/data/ubuntu_corpus_v1/ubuntu_post_training.txt",
                          help="Input raw text file (or comma-separated list of files).")
  arg_parser.add_argument("--output_file", dest="output_file", type=str,
                          default="./data/ubuntu_corpus_v1/ubuntu_post_training.json",
                          help="Output example pkl.")
  arg_parser.add_argument("--do_lower_case", dest="do_lower_case", type=bool, default=True,
                          help="Whether to lower case the input text. Should be True for uncased.")
  arg_parser.add_argument("--do_whole_word_mask", dest="do_whole_word_mask", type=bool, default=True,
                          help="Whether to use whole word masking rather than per-WordPiece masking.")
  arg_parser.add_argument("--max_seq_length", dest="max_seq_length", type=int, default=512,
                          help="Maximum sequence length.")
  arg_parser.add_argument("--max_predictions_per_seq", dest="max_predictions_per_seq", type=int, default=70,
                          help="Maximum number of masked LM predictions per sequence.")
  arg_parser.add_argument("--random_seed", dest="random_seed", type=int, default=12345,
                          help="Random seed for data generation.")
  arg_parser.add_argument("--dupe_factor", dest="dupe_factor", type=int, default=1,
                          help="Number of times to duplicate the input data (with different masks).")
  arg_parser.add_argument("--masked_lm_prob", dest="masked_lm_prob", type=float, default=0.15,
                          help="Masked LM probability.")
  arg_parser.add_argument("--short_seq_prob", dest="short_seq_prob", type=float, default=0.1,
                          help="Probability of creating sequences which are shorter than the maximum length.")
  arg_parser.add_argument("--special_tok", dest="special_tok", type=str, default="[EOT]",
                          help="Special Token.")
  args = arg_parser.parse_args()


  post_train_data = CreateBertPosttrainingData(args)

  rng = random.Random(args.random_seed)

  post_train_data.create_training_instances(
    args.input_file, args.max_seq_length, args.dupe_factor,
    args.short_seq_prob, args.masked_lm_prob, args.max_predictions_per_seq, rng, args.special_tok)
