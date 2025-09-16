# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
from torch.utils.data import Dataset

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

aspect_cate_list = ['location general',
                    'food prices',
                    'food quality',
                    'food general',
                    'ambience general',
                    'service general',
                    'restaurant prices',
                    'drinks prices',
                    'restaurant miscellaneous',
                    'drinks quality',
                    'drinks style_options',
                    'restaurant general',
                    'food style_options']


def read_line_examples_from_file(data_path, silence):
    """
    Read data from file, each line is: sent####overall####labels
    Return List[List[word]], List[Tuple], List[overall_score]
    """
    sents, labels, overalls = [], [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line_num, line in enumerate(fp):
            line = line.strip()
            if line != '':
                try:
                    parts = line.split('####')
                    if len(parts) != 3:
                        print(f"Warning: Line {line_num+1} has {len(parts)} parts instead of 3")
                        continue
                    
                    sent, overall, quad_str = parts
                    
                    # 处理overall_score格式问题
                    overall_clean = overall.strip().replace('#', '')
                    try:
                        overall_score = int(overall_clean)
                    except ValueError:
                        print(f"Warning: Line {line_num+1} has invalid overall score: {overall}")
                        continue
                    
                    # 处理null值问题
                    quad_str_clean = quad_str.replace('null', 'None')
                    
                    sents.append(sent.split())
                    overalls.append(overall_score)
                    labels.append(eval(quad_str_clean))
                except Exception as e:
                    print(f"Error parsing line {line_num+1}: {e}")
                    print(f"Line content: {line[:100]}...")
                    continue
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels, overalls


def get_para_aste_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri_sentences = []
        for tri in label:
            # a is an aspect term
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx+1])

            # b is an opinion term
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx+1])

            # c is the sentiment polarity
            c = senttag2opinion[tri[2]]           # 'POS' -> 'good'

            one_tri = f"It is {c} because {a} is {b}"
            all_tri_sentences.append(one_tri)
        targets.append(' [SSEP] '.join(all_tri_sentences))
    return targets


def get_para_tasd_targets(sents, labels):

    targets = []
    for label in labels:
        all_tri_sentences = []
        for triplet in label:
            at, ac, sp = triplet

            man_ot = sentword2opinion[sp]   # 'positive' -> 'great'

            if at == 'NULL':
                at = 'it'
            one_tri = f"{ac} is {man_ot} because {at} is {man_ot}"
            all_tri_sentences.append(one_tri)

        target = ' [SSEP] '.join(all_tri_sentences)
        targets.append(target)
    return targets


def get_para_asqp_targets(sents, labels, overalls=None):
    """
    Obtain the target sentence under the paraphrase paradigm (now for six-tuple)
    """
    targets = []
    for i, label in enumerate(labels):
        all_quad_sentences = []
        for quad in label:
            if len(quad) == 6:
                at, ac, sp, ot, score, reason = quad
                # 如果有overall_score，添加到每个六元组后面
                if overalls is not None and i < len(overalls):
                     one_quad_sentence = f"{ac}|{overalls[i]}|{at}|{ot}|{sp}|{score}|{reason}"
                else:
                    one_quad_sentence = f"{ac}||{at}|{ot}|{sp}|{score}|{reason}"
                
            else:
                at, ac, sp, ot = quad
                if overalls is not None and i < len(overalls):
                    one_quad_sentence = f"{ac}|{overalls[i]}|{at}|{ot}|{sp}|||"
                else:
                    one_quad_sentence = f"{ac}||{at}|{ot}|{sp}|||"
            all_quad_sentences.append(one_quad_sentence)
        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)
    return targets


def get_transformed_io(data_path, data_dir, include_overall=False, use_instruction=False, instruction_template=""):
    """
    The main function to transform input & target according to the task
    """
    sents, labels, overalls = read_line_examples_from_file(data_path, silence=False)
    
    # 根据参数决定输入格式
    if use_instruction:
        inputs = [f"{instruction_template}{' '.join(sent)}" for sent in sents]
    else:
        inputs = [s.copy() for s in sents]  # 原有逻辑
    
    task = 'asqp'
    if task == 'aste':
        targets = get_para_aste_targets(sents, labels)
    elif task == 'tasd':
        targets = get_para_tasd_targets(sents, labels)
    elif task == 'asqp':
        if include_overall:
            targets = get_para_asqp_targets(sents, labels, overalls)
        else:
            targets = get_para_asqp_targets(sents, labels)
    else:
        raise NotImplementedError
    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128, include_overall=False, use_instruction=False, instruction_template="", data_path=None):
        if data_path is not None:
            self.data_path = data_path
        else:
            self.data_path = f'data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.include_overall = include_overall
        self.use_instruction = use_instruction
        self.instruction_template = instruction_template

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):

        inputs, targets = get_transformed_io(self.data_path, self.data_dir, self.include_overall, self.use_instruction, self.instruction_template)

        # 确保inputs和targets长度一致
        min_len = min(len(inputs), len(targets))
        inputs = inputs[:min_len]
        targets = targets[:min_len]

        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
