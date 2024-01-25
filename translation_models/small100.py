from typing import List, Union, Tuple, Set, Optional
import torch
from tqdm import tqdm
from transformers import LogitsProcessorList, LogitsProcessor
from transformers.file_utils import PaddingStrategy
from translation_models import TranslationModel
from translation_models.utils import batch
import torch.nn.functional as F
from translation_models.m2m100 import zero_out_max, EnsembleLogitsProcessor
from transformers import M2M100ForConditionalGeneration
from translation_models.tokenization_small100 import SMALL100Tokenizer


class SMaLL100Model(TranslationModel):
    """
    Uses the implementation of the Hugging Face Transformers library
    (https://huggingface.co/alirezamsh/small100).
    """

    def __init__(self,
                 model_name_or_path: str = "alirezamsh/small100",
                 device=None,
                 ):
        self.tokenizer = SMALL100Tokenizer.from_pretrained(model_name_or_path)
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
    def ranked_languages(self) -> List[str]:
        return ["en", "es", "fr", "de", "pt", "ru", "nl", "sv", "pl", "tr", "id", "zh", "vi", "ar", "el", "cz", "ja", "hu", "fi", "ko", "he", "fa", "lt", "hi"]

    @property
    def requires_src_lang(self) -> bool:
        return False

    def _set_src_lang(self, src_lang: str):
        assert src_lang in self.supported_languages
        self.src_lang = src_lang
        #self.tokenizer.src_lang = src_lang

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
        assert len(multi_source_sentences) == len(src_langs)
        #src_weights = [0.5,0.25,0.25]

        inputs = self.tokenizer._batch_encode_plus(multi_source_sentences, return_tensors="pt",
                                                   padding_strategy=PaddingStrategy.LONGEST)
        # Set individual src language token per row
        for i, src_lang in enumerate(src_langs):
            inputs["input_ids"][i][0] = self.tokenizer.get_lang_id(tgt_langs[i])
        inputs = inputs.to(self.model.device)
        logits_processor = LogitsProcessorList([EnsembleLogitsProcessor(num_beams=num_beams, source_weights=src_weights)])
        model_output = self.model.generate(
            **inputs,
            num_beams=num_beams,
            return_dict_in_generate=True,
            logits_processor=logits_processor,
            **kwargs,
        )
        translations = self.tokenizer.batch_decode(model_output.sequences, skip_special_tokens=True)
        return translations[0]
