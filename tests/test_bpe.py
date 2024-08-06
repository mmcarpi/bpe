import unittest
from tokenizer import Tokenizer


class TestBPE(unittest.TestCase):
    def test_wikipedia(self):
        text = "aaabdaaabac"
        tokenizer = Tokenizer()
        tokenizer.train(text, 256 + 3)

        ids = tokenizer.encode(text)
        self.assertTrue(ids, [258, 100, 258, 97, 99])
        self.assertEqual(text, tokenizer.decode(ids))
