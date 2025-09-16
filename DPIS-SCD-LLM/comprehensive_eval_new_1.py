import argparse
import json
import os
import re
from typing import List, Tuple, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error
from rouge_score import rouge_scorer
import sacrebleu
import numpy as np


COMBO_METRICS = {
    'triplet_ac_at_sp': (0, 2, 4),  # (aspect_category, aspect_term, sentiment)
    'triplet_at_ot_sp': (2, 3, 4),  # (aspect_term, opinion_term, sentiment)
    'pair_at_sp': (2, 4),           # (aspect_term, sentiment)
    'pair_ac_sp': (0, 4),           # (aspect_category, sentiment)
    'pair_at_ot': (2, 3),           # (aspect_term, opinion_term)
}

def extract_tuples_from_text(text: str):
    """
     (aspect_category = ..., overall_score = ..., ...) # (aspect_category = ..., ...) 
     List[Tuple]
    """
    if not text or text.strip() == '':
        return []
    tuples = []
   
    parts = [p.strip() for p in text.strip().split('#')]
    for part in parts:

        match = re.match(r'^\(?(.+?)\)?$', part)
        if not match:
            continue
        content = match.group(1)
  
        fields = [f.strip() for f in content.split(',')]
        values = []
        for f in fields:
            if '=' in f:
                k, v = f.split('=', 1)
                values.append(v.strip())
        if len(values) >= 4:  
            tuples.append(tuple(values))
    return tuples

def calculate_f1_score(pred_tuples: List[List[Tuple]], gold_tuples: List[List[Tuple]], num_fields: int = 4) -> Dict[str, float]:
    total_pred, total_gold, total_correct = 0, 0, 0
    for pred_list, gold_list in zip(pred_tuples, gold_tuples):
        pred_set = set(tuple(t[:num_fields]) for t in pred_list if len(t) >= num_fields)
        gold_set = set(tuple(t[:num_fields]) for t in gold_list if len(t) >= num_fields)
        total_pred += len(pred_set)
        total_gold += len(gold_set)
        total_correct += len(pred_set & gold_set)
    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}

def calculate_combo_f1(pred_tuples: List[List[Tuple]], gold_tuples: List[List[Tuple]], fields: Tuple[int]) -> Dict[str, float]:
    total_pred, total_gold, total_correct = 0, 0, 0
    for pred_list, gold_list in zip(pred_tuples, gold_tuples):
        pred_set = set(tuple(t[i] for i in fields) for t in pred_list if all(len(t) > max(fields) for i in fields))
        gold_set = set(tuple(t[i] for i in fields) for t in gold_list if all(len(t) > max(fields) for i in fields))
        total_pred += len(pred_set)
        total_gold += len(gold_set)
        total_correct += len(pred_set & gold_set)
    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}

def calculate_field_level_metrics(pred_tuples: List[List[Tuple]], gold_tuples: List[List[Tuple]]) -> Dict[str, Dict[str, float]]:

    fields = ['aspect_category', 'aspect_term', 'opinion_term', 'sentiment']
    field_indices = [0, 2, 3, 4]
    field_metrics = {f: {'correct': 0, 'pred': 0, 'gold': 0} for f in fields}
    aspect_scores_pred, aspect_scores_gold = [], []
    reasons_pred, reasons_gold = [], []
    for pred_list, gold_list in zip(pred_tuples, gold_tuples):
   
        pred_dict = { (t[0], t[2], t[3]) : t for t in pred_list if len(t) >= 5 }
        gold_dict = { (t[0], t[2], t[3]) : t for t in gold_list if len(t) >= 5 }
        for key in set(pred_dict.keys()) | set(gold_dict.keys()):
            pred = pred_dict.get(key)
            gold = gold_dict.get(key)
            for i, f in enumerate(fields):
                idx = field_indices[i]
                if pred: field_metrics[f]['pred'] += 1
                if gold: field_metrics[f]['gold'] += 1
                if pred and gold and pred[idx] == gold[idx]:
                    field_metrics[f]['correct'] += 1
            
            def _to_float_safe(x: str):
                import re
                m = re.search(r'-?\d+(?:\.\d+)?', str(x))  
                return float(m.group(0)) if m else None

            if pred and gold and len(pred) > 5 and len(gold) > 5:
                p = _to_float_safe(pred[5])
                g = _to_float_safe(gold[5])
                if p is not None and g is not None:
                    aspect_scores_pred.append(p)
                    aspect_scores_gold.append(g)

            if pred and gold and len(pred) > 6 and len(gold) > 6:
                reasons_pred.append(pred[6])
                reasons_gold.append(gold[6])
    results = {}
    for f in fields:
        m = field_metrics[f]
        p = m['correct'] / m['pred'] if m['pred'] > 0 else 0
        r = m['correct'] / m['gold'] if m['gold'] > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results[f] = {'precision': p, 'recall': r, 'f1': f1}
    if aspect_scores_pred:
        results['aspect_score'] = {
            'mae': mean_absolute_error(aspect_scores_gold, aspect_scores_pred),
            'mse': mean_squared_error(aspect_scores_gold, aspect_scores_pred),
            'rmse': np.sqrt(mean_squared_error(aspect_scores_gold, aspect_scores_pred)),
            'correlation': np.corrcoef(aspect_scores_gold, aspect_scores_pred)[0, 1] if len(aspect_scores_gold) > 1 else 0
        }
    if reasons_pred:
        bleu = sacrebleu.corpus_bleu(reasons_pred, [reasons_gold])
        bleu_score = bleu.score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1, rouge2, rougel = [], [], []
        for p, g in zip(reasons_pred, reasons_gold):
            scores = scorer.score(g, p)
            rouge1.append(scores['rouge1'].fmeasure)
            rouge2.append(scores['rouge2'].fmeasure)
            rougel.append(scores['rougeL'].fmeasure)
        results['reason'] = {
            'bleu': bleu_score,
            'rouge1': np.mean(rouge1),
            'rouge2': np.mean(rouge2),
            'rougeL': np.mean(rougel)
        }
    return results

def calculate_overall_score_metrics(pred_tuples: List[List[Tuple]], gold_tuples: List[List[Tuple]]) -> Dict[str, float]:
    pred_overalls, gold_overalls = [], []
    for pred_list, gold_list in zip(pred_tuples, gold_tuples):
        pred_overall = next((float(t[1]) for t in pred_list if len(t) >= 7 and t[1].replace('.', '', 1).isdigit()), None)
        gold_overall = next((float(t[1]) for t in gold_list if len(t) >= 7 and t[1].replace('.', '', 1).isdigit()), None)
        if pred_overall is not None and gold_overall is not None:
            pred_overalls.append(pred_overall)
            gold_overalls.append(gold_overall)
    if not pred_overalls:
        return {'mae': 0, 'mse': 0, 'rmse': 0, 'correlation': 0}
    return {
        'mae': mean_absolute_error(gold_overalls, pred_overalls),
        'mse': mean_squared_error(gold_overalls, pred_overalls),
        'rmse': np.sqrt(mean_squared_error(gold_overalls, pred_overalls)),
        'correlation': np.corrcoef(gold_overalls, pred_overalls)[0, 1] if len(gold_overalls) > 1 else 0
    }

def comprehensive_evaluation(predictions: List[str], targets: List[str], sentences: List[str] = None) -> Dict[str, Any]:
    pred_tuples = [extract_tuples_from_text(pred) for pred in predictions]
    gold_tuples = [extract_tuples_from_text(tar) for tar in targets]
    min_len = min(len(pred_tuples), len(gold_tuples))
    pred_tuples, gold_tuples = pred_tuples[:min_len], gold_tuples[:min_len]
    f1_4 = calculate_f1_score(pred_tuples, gold_tuples, num_fields=4)
    f1_6 = calculate_f1_score(pred_tuples, gold_tuples, num_fields=6)
    f1_7 = calculate_f1_score(pred_tuples, gold_tuples, num_fields=7)
    field_metrics = calculate_field_level_metrics(pred_tuples, gold_tuples)
    overall_metrics = calculate_overall_score_metrics(pred_tuples, gold_tuples)
    total_pred_tuples = sum(len(t) for t in pred_tuples)
    total_gold_tuples = sum(len(t) for t in gold_tuples)
    combo_metrics = {}
    for name, idxs in COMBO_METRICS.items():
        combo_metrics[name] = calculate_combo_f1(pred_tuples, gold_tuples, idxs)
    results = {
        'summary': {
            'total_samples': min_len,
            'total_pred_tuples': total_pred_tuples,
            'total_gold_tuples': total_gold_tuples
        },
        'tuple_level_f1': {
            '4_tuple': f1_4,
            '6_tuple': f1_6,
            '7_tuple': f1_7
        },
        'field_level_metrics': field_metrics,
        'overall_score_metrics': overall_metrics,
        'absa_subtasks': {
            'Aspect Term Extraction (ATE)': field_metrics.get('aspect_term', {}),
            'Aspect Category Detection (ACD)': field_metrics.get('aspect_category', {}),
            'Opinion Term Extraction (OTE)': field_metrics.get('opinion_term', {}),
            'Sentiment Classification (SC)': field_metrics.get('sentiment', {}),
            'Aspect Score Regression (ASR)': field_metrics.get('aspect_score', {}),
            'Reason Generation (RG)': field_metrics.get('reason', {}),
            'Overall Score Regression (OSR)': overall_metrics
        },
        'combo_metrics': combo_metrics
    }
    return results

def save_comprehensive_results(results: Dict[str, Any], output_path: str, model_name: str = "Unknown", dataset_name: str = "Unknown"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Comprehensive ABSA Evaluation Report\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write("=" * 80 + "\n\n")
        summary = results['summary']
        f.write("1. 统计摘要\n" + "-" * 40 + "\n")
        f.write(f"总样本数: {summary['total_samples']}\n预测元组总数: {summary['total_pred_tuples']}\n黄金标准元组总数: {summary['total_gold_tuples']}\n\n")
        f.write("2. 元组级别F1分数\n" + "-" * 40 + "\n")
        f.write("说明:\n- 4元组: aspect_category|overall_score|aspect_term|opinion_term (新顺序)\n- 6元组: aspect_category|overall_score|aspect_term|opinion_term|sentiment|aspect_score\n- 7元组: aspect_category|overall_score|aspect_term|opinion_term|sentiment|aspect_score|reason (DIM7完整评估)\n\n")
        tuple_f1 = results['tuple_level_f1']
        for tuple_type, metrics in tuple_f1.items():
            f.write(f"{tuple_type}:\n  Precision: {metrics['precision']:.4f}\n  Recall: {metrics['recall']:.4f}\n  F1: {metrics['f1']:.4f}\n\n")
        f.write("3. 字段级别指标\n" + "-" * 40 + "\n")
        field_metrics = results['field_level_metrics']
        for field in ['aspect_category', 'aspect_term', 'opinion_term', 'sentiment']:
            if field in field_metrics:
                metrics = field_metrics[field]
                f.write(f"{field}:\n  Precision: {metrics['precision']:.4f}\n  Recall: {metrics['recall']:.4f}\n  F1: {metrics['f1']:.4f}\n\n")
        if 'aspect_score' in field_metrics:
            metrics = field_metrics['aspect_score']
            f.write("aspect_score (回归指标):\n  MAE: {mae:.4f}\n  MSE: {mse:.4f}\n  RMSE: {rmse:.4f}\n  Correlation: {cor:.4f}\n\n".format(mae=metrics['mae'], mse=metrics['mse'], rmse=metrics['rmse'], cor=metrics['correlation']))
        if 'reason' in field_metrics:
            metrics = field_metrics['reason']
            f.write("reason (生成指标):\n  BLEU: {bleu:.4f}\n  ROUGE-1: {r1:.4f}\n  ROUGE-2: {r2:.4f}\n  ROUGE-L: {rl:.4f}\n\n".format(bleu=metrics['bleu'], r1=metrics['rouge1'], r2=metrics['rouge2'], rl=metrics['rougeL']))
        f.write("4. 整体评分回归指标\n" + "-" * 40 + "\n")
        overall_metrics = results['overall_score_metrics']
        f.write(f"MAE: {overall_metrics['mae']:.4f}\nMSE: {overall_metrics['mse']:.4f}\nRMSE: {overall_metrics['rmse']:.4f}\nCorrelation: {overall_metrics['correlation']:.4f}\n\n")
        f.write("5. ABSA子任务指标总结\n" + "-" * 40 + "\n说明: 这是ABSA各个子任务的标准化指标\n\n")
        subtasks = results['absa_subtasks']
        for task_name, metrics in subtasks.items():
            f.write(f"{task_name}:\n")
            if 'f1' in metrics:
                f.write(f"  Precision: {metrics['precision']:.4f}\n  Recall: {metrics['recall']:.4f}\n  F1: {metrics['f1']:.4f}\n")
            elif 'mae' in metrics:
                f.write(f"  MAE: {metrics['mae']:.4f}\n  MSE: {metrics['mse']:.4f}\n  RMSE: {metrics['rmse']:.4f}\n  Correlation: {metrics['correlation']:.4f}\n")
            elif 'bleu' in metrics:
                f.write(f"  BLEU: {metrics['bleu']:.4f}\n  ROUGE-1: {metrics['rouge1']:.4f}\n  ROUGE-2: {metrics['rouge2']:.4f}\n  ROUGE-L: {metrics['rougeL']:.4f}\n")
            f.write("\n")
        f.write("7. 组合子任务指标\n" + "-" * 40 + "\n")
        f.write("常见三元组/二元组F1:\n")
        for name, metrics in results.get('combo_metrics', {}).items():
            f.write(f"{name}: Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}\n")
        f.write("\n")
        f.write("8. 主要指标总结 (论文表格格式)\n" + "-" * 40 + "\n主要评估指标:\n")
        tuple_f1 = results['tuple_level_f1']
        f.write(f"  7元组F1: {tuple_f1['7_tuple']['f1']:.4f}\n  6元组F1: {tuple_f1['6_tuple']['f1']:.4f}\n  4元组F1: {tuple_f1['4_tuple']['f1']:.4f}\n  整体评分MAE: {overall_metrics['mae']:.4f}\n  原因生成BLEU: {field_metrics.get('reason', {}).get('bleu', 0):.4f}\n")
    print(f"综合评估结果已保存到: {output_path}")
    return output_path

def read_jsonl_predictions(jsonl_path: str) -> (List[str], List[str]):
    predictions, labels = [], []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pred = obj.get('predict') or obj.get('prediction')
            label = obj.get('label') or obj.get('target')
            if pred is not None and label is not None:
                predictions.append(pred.strip())
                labels.append(label.strip())
    return predictions, labels

def main():
    parser = argparse.ArgumentParser(description='Comprehensive ABSA Evaluation for jsonl predictions')
    parser.add_argument('--model', type=str, required=True, help='Model name (qwen or llama)')
    args = parser.parse_args()


    if args.model.lower() == 'qwen':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_qwen2.5_predict_without_instruction_train/generated_predictions.jsonl'
        out_base = 'sim7_qwen2.5'
    elif args.model.lower() == 'llama':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_llama3.1_predict_without_instruction_train/generated_predictions.jsonl'
        out_base = 'sim7_llama3.1'
    elif args.model.lower() == 'llama_without_promt':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_llama3.1_predict_without_instruction_without_test_instruction_train/generated_predictions.jsonl'
        out_base = 'sim7_llama3.1_without_test_promt'    
    elif args.model.lower() == 'llama_instruct':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_llama3.1_predict_with_instruction_train/generated_predictions.jsonl'
        out_base = 'sim7_llama3.1_instruct'    
    elif args.model.lower() == 'llama_instruct_without_promt':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_llama3.1_predict_with_instruction__without_test_instruction_train/generated_predictions.jsonl'
        out_base = 'sim7_llama3.1_instruct_without_test_promt'        
    elif args.model.lower() == 'llama_instruct_1':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_llama3.1_predict_with_instruction_1_train/generated_predictions.jsonl'
        out_base = 'sim7_llama3.1_instruct_1'  
    elif args.model.lower() == 'llama_instruct_trick1':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_llama3.1_predict_with_instruction_trick1_train/generated_predictions.jsonl'
        out_base = 'sim7_llama3.1_instruct_trick1' 
    elif args.model.lower() == 'llama_instruct_trick2':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_llama3.1_predict_with_instruction_softpromt_A_train/generated_predictions.jsonl'
        out_base = 'sim7_llama3.1_instruct_trick2' 
    elif args.model.lower() == 'llama_instruct_trick3':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_llama3.1_predict_with_instruction_train_pass2_tuple/generated_predictions.jsonl'
        out_base = 'sim7_llama3.1_instruct_trick3'
    elif args.model.lower() == 'llama_instruct_trick4':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_llama3.1_predict_with_instruction_trick4/generated_predictions.jsonl'
        out_base = 'sim7_llama3.1_instruct_trick4'
    elif args.model.lower() == 'llama_instruct_0shot':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_llama31_icl0/generated_predictions.jsonl'
        out_base = 'sim7_llama31_icl0'
    elif args.model.lower() == 'llama_instruct_5shot':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_llama31_icl5/generated_predictions.jsonl'
        out_base = 'sim7_llama31_icl5'     
    elif args.model.lower() == 'llama_instruct_10shot':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_llama31_icl10/generated_predictions.jsonl'
        out_base = 'sim7_llama31_icl10'  
    elif args.model.lower() == 'qwen_instruct_0shot':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_qwen2.5_icl0/generated_predictions.jsonl'
        out_base = 'sim7_qwen2.5_icl0'
    elif args.model.lower() == 'qwen_instruct_5shot':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_qwen2.5_icl5/generated_predictions.jsonl'
        out_base = 'sim7_qwen2.5_icl5'     
    elif args.model.lower() == 'qwen_instruct_10shot':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_qwen2.5_icl10/generated_predictions.jsonl'
        out_base = 'sim7_qwen2.5_icl10'
    elif args.model.lower() == 'qwen_without_promt':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_qwen2.5_predict_without_instruction_without_test_instruction_train/generated_predictions.jsonl'
        out_base = 'sim7_qwen2.5_without_promt'
    elif args.model.lower() == 'qwen_with_promt':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_qwen2.5_predict_without_instruction_train/generated_predictions.jsonl'
        out_base = 'sim7_qwen2.5_with_promt'
    elif args.model.lower() == 'qwen_instruct_with_promt':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_qwen2.5_instruct_predict_with_instruction_train/generated_predictions.jsonl'
        out_base = 'sim7_qwen2.5_instruct_with_promt'
    elif args.model.lower() == 'qwen_instruct_without_promt':
        input_path = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/predict/sim7_qwen2.5_insturct_predict_without_instruction_train/generated_predictions.jsonl'
        out_base = 'sim7_qwen2.5_instruct_without_promt'



    else:
        raise ValueError('Unknown model name! 只支持qwen或llama')

    predictions, labels = read_jsonl_predictions(input_path)
    print(f"Loaded {len(predictions)} predictions and {len(labels)} labels from {input_path}.")


    result_dir = '/newdata_yjwang/ABSA_code/ABSA-QUAD-New/llama_factory/result_log'
    os.makedirs(result_dir, exist_ok=True)
    output_file = f"{out_base}_eval.txt"
    output_path = os.path.join(result_dir, output_file)

    results = comprehensive_evaluation(predictions, labels)
    save_comprehensive_results(results, output_path, model_name=args.model, dataset_name=out_base)

    print(f"评估结果已保存到: {output_path}")

if __name__ == '__main__':
    main()