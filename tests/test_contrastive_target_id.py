from unittest import TestCase

from translation_models import load_translation_model


class ContrastiveTargetIDTestCase(TestCase):

    def setUp(self) -> None:
        self.model = load_translation_model("m2m100_418M")

    def test_translate_multi_source_greedy(self):
        translation = self.model.translate_multi_source(
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
        self.assertEqual(translation, "Ich mag Äpfel")

        translation = self.model.translate_multi_source(
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
        self.assertEqual(translation, "J’aime les pommes")

    def test_translate_multi_source_beam_search(self):
        translation = self.model.translate_multi_source(
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
            num_beams=5,
        )
        self.assertEqual(translation, "Ich mag Äpfel")

        translation = self.model.translate_multi_source(
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
            num_beams=5,
        )
        self.assertEqual(translation, "J’aime les pommes")
