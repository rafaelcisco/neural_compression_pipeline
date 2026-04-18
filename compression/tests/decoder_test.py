import pytest
from compression.huffman.encoder import compress
from compression.huffman.decoder import decompress


def test_decompress_empty_string():
    assert decompress("") == ""


def test_round_trip_basic():
    text = "LOLOLOL"
    encoded = compress(text)
    decoded = decompress(encoded["compressed_data"])

    assert decoded == text


def test_round_trip_single_character():
    text = "A"
    encoded = compress(text)
    decoded = decompress(encoded["compressed_data"])

    assert decoded == text


def test_round_trip_repeated_character():
    text = "AAAAAA"
    encoded = compress(text)
    decoded = decompress(encoded["compressed_data"])

    assert decoded == text


def test_round_trip_mixed_characters():
    text = "HELLOronitJAH123"
    encoded = compress(text)
    decoded = decompress(encoded["compressed_data"])

    assert decoded == text


def test_round_trip_whitespace_and_symbols():
    text = "A B!\n"
    encoded = compress(text)
    decoded = decompress(encoded["compressed_data"])

    assert decoded == text


def test_round_trip_longer_text():
    text = "Yanin needs a girlfriend badly"
    encoded = compress(text)
    decoded = decompress(encoded["compressed_data"])

    assert decoded == text


def test_round_trip_matches_original_text_exactly():
    text = "hackathon2026"
    encoded = compress(text)
    decoded = decompress(encoded["compressed_data"])

    assert decoded == text
    assert isinstance(decoded, str)
    assert len(decoded) == len(text)