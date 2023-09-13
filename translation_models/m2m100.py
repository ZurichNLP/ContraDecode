from typing import List, Union, Tuple, Set, Optional

import torch
import sys
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, LogitsProcessorList, LogitsProcessor, \
    ForcedBOSTokenLogitsProcessor
from transformers.file_utils import PaddingStrategy

from translation_models import TranslationModel
from translation_models.utils import batch
import torch.nn.functional as F


def zero_out_max(x):
    max_num, max_index = x.max(dim=0)
    x[max_index] = 0
    return x


class EnsembleLogitsProcessor(LogitsProcessor):

    def __init__(self, num_beams: int, source_weights: List[float] = None, preserve_bos_token: bool = False):
        self.num_beams = num_beams
        self.source_weights = source_weights
        self.preserve_bos_token = preserve_bos_token

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if self.preserve_bos_token and cur_len <= 1:
            return scores

        scores = F.softmax(scores, dim=-1)

        batch_size = int(input_ids.size(0) / self.num_beams)
        if self.source_weights is not None:
            assert len(self.source_weights) == batch_size
            source_weights = torch.Tensor(self.source_weights).to(scores.device)
        else:
            source_weights = 1/(batch_size-1) * torch.ones((batch_size,), device=scores.device)
        for i in range(self.num_beams):
            beam_indices = self.num_beams * torch.arange(batch_size, device=scores.device, dtype=torch.long) + i
            cands = scores[beam_indices]
            mean_scores = torch.log((source_weights.unsqueeze(-1).expand(-1, scores.size(-1)) * cands).sum(dim=0))
            for j in beam_indices:
                scores[j] = mean_scores

        if torch.isnan(scores).any():
            scores = torch.nan_to_num(scores, nan=float('-inf'))

        return scores


class BatchedForcedBOSTokenLogitsProcessor(ForcedBOSTokenLogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the first generated token.
    This subclass allows for different forced bos tokens per row in the batch.
    """

    def __init__(self, bos_token_ids: List[int]):
        self.bos_token_ids = bos_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len != 1:
            return scores

        batch_size = len(self.bos_token_ids)
        num_beams = int(input_ids.size(0) / batch_size)
        # Disable all tokens by default
        scores.fill_(-float("inf"))
        for row_idx in range(batch_size):
            # Set the bos token for all beams corresponding to this input row
            row_indices = torch.arange(num_beams, device=scores.device) + row_idx * num_beams
            scores[row_indices, self.bos_token_ids[row_idx]] = 0
        return scores


class M2M100Model(TranslationModel):
    """
    Loads one of the models described in: Fan, Angela, et al. "Beyond english-centric multilingual machine
    translation." Journal of Machine Learning Research 22.107 (2021): 1-48.

    Uses the implementation of the Hugging Face Transformers library
    (https://huggingface.co/docs/transformers/model_doc/m2m_100).
    """

    def __init__(self,
                 model_name_or_path: str = "facebook/m2m100_418M",
                 device=None,
                 ):
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name_or_path)
        if device is not None:
            self.model = self.model.to(device)
        self.model.config.max_length = max(self.model.config.max_length, self.model.config.max_position_embeddings - 4)

    def __str__(self):
        return self.model_name_or_path

    @property
    def supported_languages(self) -> Set[str]:
        return {'af', 'am', 'ar', 'ast', 'az', 'ba', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'ceb', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'fa', 'ff', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'ilo', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'lb', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'ne', 'nl', 'no', 'ns', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'th', 'tl', 'tn', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'zu'}

    @property
    def requires_src_lang(self) -> bool:
        return True

    def _set_src_lang(self, src_lang: str):
        assert src_lang in self.supported_languages
        self.src_lang = src_lang
        self.tokenizer.src_lang = src_lang

    def _set_tgt_lang(self, tgt_lang: str):
        assert tgt_lang in self.supported_languages
        self.tgt_lang = tgt_lang
        self.tokenizer.tgt_lang = tgt_lang

    @torch.no_grad()
    def _translate(self,
                   source_sentences: List[str],
                   return_score: bool = False,
                   batch_size: int = 8,
                   num_beams: int = 5,
                   **kwargs,
                   ) -> Union[List[str], List[Tuple[str, float]]]:
        padding_strategy = PaddingStrategy.LONGEST if batch_size > 1 else PaddingStrategy.DO_NOT_PAD
        translations = []
        for src_sentences in tqdm(list(batch(source_sentences, batch_size)), disable=len(source_sentences) / batch_size < 10):
            inputs = self.tokenizer._batch_encode_plus(src_sentences, return_tensors="pt",
                                                       padding_strategy=padding_strategy)
            inputs = inputs.to(self.model.device)
            model_output = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang),
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=return_score,
                **kwargs,
            )
            batch_translations = self.tokenizer.batch_decode(model_output.sequences, skip_special_tokens=True)
            if return_score:
                # Does not match our score method output for some reason; need to investigate further
                # scores = (2 ** model_output.sequences_scores).tolist()
                scores = [None for _ in batch_translations]
                assert len(batch_translations) == len(scores)
                batch_translations = list(zip(batch_translations, scores))
            translations += batch_translations
        return translations

    def _translate_multi_source(self,
                                multi_source_sentences: List[str],
                                src_langs: List[str],
                                tgt_langs: List[str],
                                src_weights: Optional[List[float]] = None,
                                num_beams: int = 1,
                                **kwargs,
                                ) -> str:
        assert len(multi_source_sentences) == len(src_langs) == len(tgt_langs)
        #src_weights = [1, -0.1]
        if src_weights is not None:
            assert len(src_weights) == len(multi_source_sentences)

        inputs = self.tokenizer._batch_encode_plus(multi_source_sentences, return_tensors="pt",
                                                   padding_strategy=PaddingStrategy.LONGEST)
        # Set individual src language token per row
        for i, src_lang in enumerate(src_langs):
            inputs["input_ids"][i][0] = self.tokenizer.get_lang_id(src_lang)
        inputs = inputs.to(self.model.device)
        logits_processor = LogitsProcessorList([
            BatchedForcedBOSTokenLogitsProcessor([self.tokenizer.get_lang_id(tgt_lang) for tgt_lang in tgt_langs]),
            EnsembleLogitsProcessor(num_beams=num_beams, source_weights=src_weights, preserve_bos_token=True),
        ])
        model_output = self.model.generate(
            **inputs,
            num_beams=num_beams,
            return_dict_in_generate=True,
            logits_processor=logits_processor,
            **kwargs,
        )
        translations = self.tokenizer.batch_decode(model_output.sequences, skip_special_tokens=True)
        return translations[0]
