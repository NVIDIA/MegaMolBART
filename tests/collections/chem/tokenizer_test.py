# coding=utf-8

import pytest
import random
import torch
from nemo.collections.chem.tokenizer import MolEncTokenizer, MolEncTokenizerFromSmilesConfig
# from nemo.collections.chem.tokenizer.tokenizer import DEFAULT_REGEX

# Use dummy SMILES strings
smiles_data = [
    "CCO.Ccc",
    "CCClCCl",
    "C(=O)CBr"
]
cfg = MolEncTokenizerFromSmilesConfig({'smiles': smiles_data})

example_tokens = [
    ["^", "C", "(", "=", "O", ")", "unknown", "&"], 
    ["^", "C", "C", "<SEP>", "C", "Br", "&"]
]

# Setting seed here only applies when all tests run in same order, so these are now set per test
SEED = 0

def test_create_vocab():
    tokenizer = MolEncTokenizer.from_smiles(smiles_data, cfg.regex)
    expected = {
        "<PAD>": 0,
        "?": 1,
        "^": 2,
        "&": 3,
        "<MASK>": 4,
        "<SEP>": 5,
        "C": 6,
        "O": 7,
        ".": 8,
        "c": 9,
        "Cl": 10,
        "(": 11,
        "=": 12,
        ")": 13,
        "Br": 14
    }

    vocab = dict(sorted(tokenizer.vocab.items(), key=lambda x: x[1]))
    assert expected == vocab


def test_pad_seqs_padding():
    seqs = [[1,2], [2,3,4,5], []]
    padded, _ = MolEncTokenizer._pad_seqs(seqs, " ")
    expected = [[1,2, " ", " "], [2,3,4,5], [" ", " ", " ", " "]]

    assert padded == expected


def test_pad_seqs_mask():
    seqs = [[1,2], [2,3,4,5], []]
    _, mask = MolEncTokenizer._pad_seqs(seqs, " ")
    expected_mask = [[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]

    assert expected_mask == mask


def test_mask_tokens_empty_mask():
    tokenizer = MolEncTokenizer.from_smiles(cfg.smiles, cfg.regex)
    masked, token_mask = tokenizer._mask_tokens(example_tokens, empty_mask=True)
    expected_sum = 0
    mask_sum = sum([sum(m) for m in token_mask])

    assert masked == example_tokens
    assert expected_sum == mask_sum


# Run tests which require random masking first so we get deterministic masking
# NB: ordered running no longer required, this is kept for compatiiblity with previous versions
@pytest.mark.order(1)
def test_mask_tokens_replace():
    random.seed(a=1)
    torch.manual_seed(SEED) 
    tokenizer = MolEncTokenizer.from_smiles(cfg.smiles, cfg.regex, mask_prob=0.4, mask_scheme='replace')
    masked, token_mask = tokenizer._mask_tokens(example_tokens)

    expected_masks = [
        [True, False, False, True, False, False, False, False],
        [False, False, False, True, False, False, True]
    ]

    assert expected_masks == token_mask


@pytest.mark.order(3)
def test_mask_tokens_span():
    random.seed(a=1)
    torch.manual_seed(SEED) 

    tokenizer = MolEncTokenizer.from_smiles(cfg.smiles, cfg.regex, mask_prob=0.4)
    masked, token_mask = tokenizer._mask_tokens(example_tokens)

    expected_masks = [
        [True, False, False], [True, False, False, True]
    ]

    assert token_mask == expected_masks


def test_convert_tokens_to_ids():
    tokenizer = MolEncTokenizer.from_smiles(smiles_data[2:3], cfg.regex)
    ids = tokenizer.convert_tokens_to_ids(example_tokens)
    expected_ids = [[2, 6, 7, 8, 9, 10, 1, 3], [2, 6, 6, 5, 6, 11, 3]]

    assert expected_ids == ids


def test_tokenize_one_sentence():
    tokenizer = MolEncTokenizer.from_smiles(cfg.smiles, cfg.regex)
    tokens = tokenizer.tokenize(smiles_data)
    expected = [
        ["^", "C", "C", "O", ".", "C", "c", "c", "&"],
        ["^", "C", "C", "Cl", "C", "Cl", "&"],
        ["^", "C", "(", "=", "O", ")", "C", "Br", "&"]
    ]

    assert expected == tokens["original_tokens"]


def test_tokenize_two_sentences():
    tokenizer = MolEncTokenizer.from_smiles(cfg.smiles, cfg.regex)
    tokens = tokenizer.tokenize(smiles_data, sents2=smiles_data)
    expected = [
        ["^", "C", "C", "O", ".", "C", "c", "c", "<SEP>", "C", "C", "O", ".", "C", "c", "c", "&"],
        ["^", "C", "C", "Cl", "C", "Cl", "<SEP>", "C", "C", "Cl", "C", "Cl", "&"],
        ["^", "C", "(", "=", "O", ")", "C", "Br", "<SEP>", "C", "(", "=", "O", ")", "C", "Br", "&"]
    ]
    expected_sent_masks = [
        ([0] * 9) + ([1] * 8),
        ([0] * 7) + ([1] * 6),
        ([0] * 9) + ([1] * 8),
    ]

    assert expected == tokens["original_tokens"]
    assert expected_sent_masks == tokens["sentence_masks"]


@pytest.mark.order(2)
def test_tokenize_mask_replace():
    random.seed(a=1)
    torch.manual_seed(SEED) 

    tokenizer = MolEncTokenizer.from_smiles(cfg.smiles, cfg.regex, mask_prob=0.4, mask_scheme="replace")
    tokens = tokenizer.tokenize(smiles_data, sents2=smiles_data, mask=True)

    expected_m_tokens = [
        ['^', '<MASK>', 'C', 'O', '<MASK>', 'C', 'c', 'c', '<SEP>', '<MASK>', '<MASK>', 'O', '.', '<MASK>', '<MASK>', '<MASK>', '&'],
        ['^', '<MASK>', 'C', 'Cl', 'C', '<MASK>', '<SEP>', 'C', '<MASK>', 'Cl', 'C', '<MASK>', '&'],
        ['^', '<MASK>', '(', '=', '<MASK>', '<MASK>', 'C', 'Br', '<SEP>', 'C', '(', '=', 'O', ')', '<MASK>', 'Br', '&']
    ]
    expected_tokens = [
        ['^', 'C', 'C', 'O', '.', 'C', 'c', 'c', '<SEP>', 'C', 'C', 'O', '.', 'C', 'c', 'c', '&'],
        ['^', 'C', 'C', 'Cl', 'C', 'Cl', '<SEP>', 'C', 'C', 'Cl', 'C', 'Cl', '&'],
        ['^', 'C', '(', '=', 'O', ')', 'C', 'Br', '<SEP>', 'C', '(', '=', 'O', ')', 'C', 'Br', '&']
    ]

    assert expected_m_tokens == tokens["masked_tokens"]
    assert expected_tokens == tokens["original_tokens"]


@pytest.mark.order(4)
def test_tokenize_mask_span():
    random.seed(a=1)
    torch.manual_seed(SEED) 

    tokenizer = MolEncTokenizer.from_smiles(cfg.smiles, cfg.regex, mask_prob=0.4)
    tokens = tokenizer.tokenize(smiles_data, sents2=smiles_data, mask=True)

    expected_m_tokens = [
        ['^', '<MASK>', 'c', '<SEP>', '<MASK>', '<MASK>', '&'],
        ['^', 'C', '<MASK>', 'Cl', '<SEP>', '<MASK>', '&'],
        ['^', 'C', '<MASK>', 'O', '<MASK>', '<SEP>', '<MASK>', '<MASK>', 'Br', '&']
    ]
    expected_tokens = [
        ['^', 'C', 'C', 'O', '.', 'C', 'c', 'c', '<SEP>', 'C', 'C', 'O', '.', 'C', 'c', 'c', '&'],
        ['^', 'C', 'C', 'Cl', 'C', 'Cl', '<SEP>', 'C', 'C', 'Cl', 'C', 'Cl', '&'],
        ['^', 'C', '(', '=', 'O', ')', 'C', 'Br', '<SEP>', 'C', '(', '=', 'O', ')', 'C', 'Br', '&']
    ]

    assert expected_m_tokens == tokens["masked_tokens"]
    assert expected_tokens == tokens["original_tokens"]
    assert len(tokens["masked_tokens"]) == len(tokens["token_masks"])

    for ts, tms in zip(tokens["masked_tokens"], tokens["token_masks"]):
        assert len(ts) == len(tms)


@pytest.mark.order(5)
def test_tokenize_mask_span_pad():
    random.seed(a=1)
    torch.manual_seed(SEED) 

    tokenizer = MolEncTokenizer.from_smiles(cfg.smiles, cfg.regex, mask_prob=0.4)
    tokens = tokenizer.tokenize(smiles_data, mask=True, pad=True)

    expected_m_tokens = [
        ['^', '<MASK>', 'c', '&', '<PAD>', '<PAD>'],
        ['^', 'C', '<MASK>', 'Cl', '&', '<PAD>'],
        ['^', 'C', '<MASK>', 'O', '<MASK>', '&']
    ]
    expected_tokens = [
        ['^', 'C', 'C', 'O', '.', 'C', 'c', 'c', '&'],
        ['^', 'C', 'C', 'Cl', 'C', 'Cl', '&', '<PAD>', '<PAD>'],
        ['^', 'C', '(', '=', 'O', ')', 'C', 'Br', '&']
    ]

    assert expected_m_tokens == tokens["masked_tokens"]
    assert expected_tokens == tokens["original_tokens"]
    assert len(tokens["masked_tokens"]) == len(tokens["token_masks"])
    assert len(tokens["masked_tokens"]) == len(tokens["masked_pad_masks"])

    for ts, tms in zip(tokens["masked_tokens"], tokens["token_masks"]):
        assert len(ts) == len(tms)

    for ts, pms in zip(tokens["masked_tokens"], tokens["masked_pad_masks"]):
        assert len(ts) == len(pms)


def test_tokenize_padding():
    tokenizer = MolEncTokenizer.from_smiles(cfg.smiles, cfg.regex)
    output = tokenizer.tokenize(smiles_data, sents2=smiles_data, pad=True)
    expected_tokens = [
        ["^", "C", "C", "O", ".", "C", "c", "c", "<SEP>", "C", "C", "O", ".", "C", "c", "c", "&"],
        ["^", "C", "C", "Cl", "C", "Cl", "<SEP>", "C", "C", "Cl", "C", "Cl", "&", "<PAD>", "<PAD>", "<PAD>", "<PAD>"],
        ["^", "C", "(", "=", "O", ")", "C", "Br", "<SEP>", "C", "(", "=", "O", ")", "C", "Br", "&"]
    ]
    expected_pad_masks = [
        [0] * 17,
        ([0] * 13) + ([1] * 4),
        [0] * 17
    ]
    expected_sent_masks = [
        ([0] * 9) + ([1] * 8),
        ([0] * 7) + ([1] * 6) + ([0] * 4),
        ([0] * 9) + ([1] * 8),
    ]

    assert expected_tokens == output["original_tokens"]
    assert expected_pad_masks == output["original_pad_masks"]
    assert expected_sent_masks == output["sentence_masks"]
