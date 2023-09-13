import logging
from unittest import TestCase

from translation_models import load_translation_model
from translation_models.llama import LLaMaTranslationModel


# logging.basicConfig(level=logging.INFO)


class LLaMaTranslationModelTestCase(TestCase):

    def setUp(self) -> None:
        self.llama: LLaMaTranslationModel = load_translation_model("llama-2-7b-chat")
        # self.llama.one_shot = True
        self.assertEqual(self.llama._lang_code_to_name("en"), "English")
        self.assertEqual(self.llama._lang_code_to_name("de"), "German")

    def test_translate(self):
        source_sentences = [
            "Hello, how are you?",
            "An inquiry was established to investigate.",
            "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new "
                "diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using "
                "standard inkjet printers for possibly about one U.S. cent each.",
        ]

        for tgt_lang in [
            "de",
            "fr",
            "ru",
        ]:
            translations = self.llama.translate(
                src_lang="en",
                tgt_lang=tgt_lang,
                source_sentences=source_sentences,
                num_beams=1,
            )
            for translation in translations:
                print(translation)

    def test_translate_multi_source(self):
        translation = self.llama.translate_multi_source(
            multi_source_sentences=[
                "I like apples",
                "I like apples",
            ],
            src_langs=2 * ["en"],
            tgt_langs=[
                "de",
                "fr",
            ],
            src_weights=[0.8, 0.2],  # Upweight German target ID
            num_beams=1,
        )
        print(translation)

        translation = self.llama.translate_multi_source(
            multi_source_sentences=[
                "I like apples",
                "I like apples",
            ],
            src_langs=2 * ["en"],
            tgt_langs=[
                "de",
                "fr",
            ],
            src_weights=[0.2, 0.8],  # Upweight French target ID
            num_beams=1,
        )
        print(translation)
