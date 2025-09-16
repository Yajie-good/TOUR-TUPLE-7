import re
import numpy as np
from data_utils import aspect_cate_list

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}


def extract_spans_para(task, seq, seq_type):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    if task == 'aste':
        for s in sents:
            # It is bad because editing is problem.
            try:
                c, ab = s.split(' because ')
                c = opinion2word.get(c[6:], 'nope')    # 'good' -> 'positive'
                a, b = ab.split(' is ')
            except ValueError:
                a, b, c = '', '', ''
            quads.append((a, b, c))
    elif task == 'tasd':
        for s in sents:
            # food quality is bad because pizza is bad.
            try:
                ac_sp, at_sp = s.split(' because ')
                ac, sp = ac_sp.split(' is ')
                at, sp2 = at_sp.split(' is ')
                sp = opinion2word.get(sp, 'nope')
                sp2 = opinion2word.get(sp2, 'nope')
                if sp != sp2:
                    print(f'Sentiment polairty of AC({sp}) and AT({sp2}) is inconsistent!')
                if at.lower() == 'it':
                    at = 'NULL'
            except ValueError:
                ac, at, sp = '', '', ''
            quads.append((ac, at, sp))
    elif task == 'asqp':
        for s in sents:
            # 新顺序：ac|overall|at|ot|sp|score|reason
            parts = s.split('|')
            if len(parts) >= 7:
                ac, overall, at, ot, sp, score, reason = [p.strip() for p in parts[:7]]
                quad = (ac, overall, at, ot, sp, score, reason)
                quads.append(quad)
            elif len(parts) >= 6:
                ac, overall, at, ot, sp, score = [p.strip() for p in parts[:6]]
                quad = (ac, overall, at, ot, sp, score, "")
                quads.append(quad)
            elif len(parts) >= 4:
                ac, overall, at, ot = [p.strip() for p in parts[:4]]
                quad = (ac, overall, at, ot, "", "", "")
                quads.append(quad)
            else:
                quads.append(("", "", "", "", "", "", ""))
    else:
        raise NotImplementedError
    return quads


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    n_tp, n_gold, n_pred = 0, 0, 0
    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])
        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1
    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}
    return scores


def compute_scores(pred_seqs, gold_seqs, sents):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)
    all_labels, all_preds = [], []
    gold_overall, pred_overall = [], []

    for i in range(num_samples):
        gold_list = extract_spans_para('asqp', gold_seqs[i], 'gold')
        pred_list = extract_spans_para('asqp', pred_seqs[i], 'pred')
        all_labels.append(gold_list)
        all_preds.append(pred_list)

        
        # gold
        if gold_list and gold_list[0][1].isdigit():
            gold_overall.append(float(gold_list[0][1]))
        else:
            gold_overall.append(np.nan)
        # pred
        if pred_list and pred_list[0][1].replace('.', '', 1).isdigit():
            pred_overall.append(float(pred_list[0][1]))
        else:
            pred_overall.append(np.nan)

    print("\nResults:")
    scores = compute_f1_scores(all_preds, all_labels)
    print(scores)

    # MAE
    gold_overall = np.array(gold_overall)
    pred_overall = np.array(pred_overall)
    mask = ~np.isnan(gold_overall) & ~np.isnan(pred_overall)
    if np.sum(mask) > 0:
        mae = np.mean(np.abs(gold_overall[mask] - pred_overall[mask]))
    else:
        mae = None
    print(f"MAE (overall score): {mae}")

    scores['mae'] = mae
    return scores, all_labels, all_preds
