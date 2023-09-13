import json
import logging
import os
import warnings

from pathlib import Path
from typing import List, Union, Tuple, Set, Optional

from tqdm import tqdm


class TranslationModel:

    def __str__(self):
        raise NotImplementedError

    def translate(self,
                  tgt_lang: str,
                  source_sentences: Union[str, List[str]],
                  src_lang: str = None,
                  return_score: bool = False,
                  batch_size: int = 8,
                  num_beams: int = 5,
                  **kwargs,
                  ) -> Union[str, List[str], Tuple[str, float], List[Tuple[str, float]]]:
        """
        :param tgt_lang: Language code of the target language
        :param source_sentences: A sentence or list of sentences
        :param src_lang: Language code of the source language (not needed for some multilingual models)
        :param return score: If true, return a tuple where the second element is sequence-level score of the translation
        :param batch_size
        :param kwargs
        :return: A sentence or list of sentences
        """
        if isinstance(source_sentences, str):
            source_sentences_list = [source_sentences]
        elif isinstance(source_sentences, list):
            source_sentences_list = source_sentences
        else:
            raise ValueError

        self._set_tgt_lang(tgt_lang)
        if self.requires_src_lang:
            if src_lang is None:
                warnings.warn(f"NMT model {self} requires the src language. Assuming 'en'; override with `src_lang`")
                src_lang = "en"
            self._set_src_lang(src_lang)
        translations_list = self._translate(source_sentences_list, return_score, batch_size, num_beams=num_beams, **kwargs)
        assert len(translations_list) == len(source_sentences_list)

        if isinstance(source_sentences, str):
            translations = translations_list[0]
        else:
            translations = translations_list
        return translations

    def _translate_multi_source(self,
                                multi_source_sentences: List[str],
                                src_langs: List[str],
                                tgt_langs: List[str],
                                src_weights: Optional[List[float]] = None,
                                **kwargs,
                                ) -> str:
        raise NotImplementedError


    def translate_multi_source(self,
                               multi_source_sentences: List[str],
                               tgt_langs: List[str],
                               src_langs: Optional[List[str]] = None,
                               src_weights: Optional[List[float]] = None,
                               num_beams: int = 5,
                               **kwargs,
                               ) -> str:
        translation = None

        if translation is None:
            self._set_tgt_lang(tgt_langs[0])
            if self.requires_src_lang:
                assert src_langs is not None
            translation = self._translate_multi_source(multi_source_sentences, src_langs, tgt_langs, src_weights=src_weights, num_beams=num_beams, **kwargs)

        return translation


def load_translation_model(name: str, **kwargs) -> TranslationModel:
    """
    Convenience function to load a :class: TranslationModel using a shorthand name of the model
    """
    if name == "m2m100_418M":
        from translation_models.m2m100 import M2M100Model
        translation_model = M2M100Model(model_name_or_path="facebook/m2m100_418M", **kwargs)
    elif name == "m2m100_1.2B":
        from translation_models.m2m100 import M2M100Model
        translation_model = M2M100Model(model_name_or_path="facebook/m2m100_1.2B", **kwargs)
    elif name == "small100":
        from translation_models.small100 import SMaLL100Model
        translation_model = SMaLL100Model(model_name_or_path="alirezamsh/small100", **kwargs)
    elif name == "llama-2-7b-chat":
        from translation_models.llama import LLaMaTranslationModel
        translation_model = LLaMaTranslationModel(model_name_or_path="meta-llama/Llama-2-7b-chat-hf", **kwargs)
    elif name == "llama-2-13b-chat":
        from translation_models.llama import LLaMaTranslationModel
        translation_model = LLaMaTranslationModel(model_name_or_path="meta-llama/Llama-2-13b-chat-hf", **kwargs)
    elif name == "llama-2-70b-chat":
        from translation_models.llama import LLaMaTranslationModel
        translation_model = LLaMaTranslationModel(model_name_or_path="meta-llama/Llama-2-70b-chat-hf", **kwargs)
    else:
        raise NotImplementedError
    return translation_model
