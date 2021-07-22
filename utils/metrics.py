from typing import Dict, List, Optional

from rouge import Rouge
import torch


class PredictionStatistic:
    def __init__(self, mask_pad: bool, pad_idx: Optional[int] = None, skip_tokens: List[int] = None):
        if mask_pad and pad_idx is None:
            raise ValueError("You should specify pad id token for masking after it")
        self._mask_pad = mask_pad
        self._pad_idx = pad_idx
        self._skip_tokens = [] if skip_tokens is None else skip_tokens
        self._true_positive = self._false_positive = self._false_negative = 0
        self._rouge = Rouge()

    @staticmethod
    def _calculate_metric(true_positive: int, false_positive: int, false_negative: int) -> Dict[str, float]:
        precision, recall, f1 = 0.0, 0.0, 0.0
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        metrics_dict = {"precision": precision, "recall": recall, "f1": f1}
        return metrics_dict

    def get_metric(self) -> Dict[str, float]:
        return self._calculate_metric(self._true_positive, self._false_positive, self._false_negative)

    def _mask_tensor_after_pad(self, target: torch.Tensor) -> torch.Tensor:
        assert self._pad_idx is not None
        mask = target != self._pad_idx
        return mask

    def update_statistic(self, original: torch.Tensor, prediction: torch.Tensor) -> Dict[str, float]:
        """Calculate subtoken statistic for ground truth and predicted batches of labels.
        :param original: [true seq length; batch_size] ground truth labels
        :param prediction: [pred seq length; batch_size] predicted labels
        :return: dict with metrics for current batch
        """
        batch_size = original.shape[1]
        if prediction.shape[1] != batch_size:
            raise ValueError(f"Wrong batch size for prediction (expected: {batch_size}, actual: {prediction.shape[1]})")
        if self._mask_pad:
            mask = self._mask_tensor_after_pad(original)
            prediction[mask] = self._pad_idx

        true_positive = false_positive = false_negative = 0

        hyps = []
        refs = []

        for batch_idx in range(batch_size):
            gt_seq = [st for st in original[:, batch_idx] if st not in self._skip_tokens]
            pred_seq = [st for st in prediction[:, batch_idx] if st not in self._skip_tokens]

            refs.append(' '.join(list(map(lambda t: str(int(t.item())), gt_seq))))
            pred_str = ' '.join(list(map(lambda t: str(int(t.item())), pred_seq)))
            if len(pred_str) == 0:
                pred_str = '-'
            hyps.append(pred_str)

            if len(gt_seq) == len(pred_seq) and all([g == p for g, p in zip(gt_seq, pred_seq)]):
                true_positive += len(gt_seq)
                continue

            for pred_subtoken in pred_seq:
                if pred_subtoken in gt_seq:
                    true_positive += 1
                else:
                    false_positive += 1
            for gt_subtoken in gt_seq:
                if gt_subtoken not in pred_seq:
                    false_negative += 1

        self._true_positive += true_positive
        self._false_positive += false_positive
        self._false_negative += false_negative

        one_token_metrics = self._calculate_metric(true_positive, false_positive, false_negative)

        rouge_metrics = self._rouge.get_scores(hyps, refs, avg=True)
        ans = {}
        for m, d in rouge_metrics.items():
            for k, v in d.items():
                ans[f"{m}_{k}"] = v
        for k, v in one_token_metrics.items():
            ans[k] = v

        return ans

    @staticmethod
    def create_from_list(statistics: List["PredictionStatistic"]) -> "PredictionStatistic":
        if len(statistics) == 0:
            raise ValueError("Empty list of statistics passed")
        statistic = statistics[0]
        for other_statistic in statistics[1:]:
            statistic._true_positive += other_statistic._true_positive
            statistic._false_positive += other_statistic._false_positive
            statistic._false_negative += other_statistic._false_negative
        return statistic