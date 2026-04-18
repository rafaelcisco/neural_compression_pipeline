import pytest
from compression.huffman.encoder import compress


def test_compress_returns_expected_keys():
    result = compress("ABBAACA")

    assert "compressed_data" in result
    assert "original_text" in result
    assert "original_bits" in result
    assert "compressed_bits" in result
    assert "compression_ratio" in result


def test_compress_preserves_original_text():
    text = "ABBAACA"
    result = compress(text)

    assert result["original_text"] == text


def test_compress_returns_bitstring():
    result = compress("ABBAACA")
    compressed_data = result["compressed_data"]

    assert isinstance(compressed_data, str)
    assert all(bit in {"0", "1"} for bit in compressed_data)


def test_original_bits_is_8_times_text_length():
    text = "ABBAACA"
    result = compress(text)

    assert result["original_bits"] == len(text) * 8


def test_compressed_bits_matches_bitstring_length():
    result = compress("ABBAACA")

    assert result["compressed_bits"] == len(result["compressed_data"])


def test_compression_ratio_is_computed_correctly():
    text = "ABBAACA"
    result = compress(text)

    expected_ratio = result["compressed_bits"] / result["original_bits"]
    assert result["compression_ratio"] == expected_ratio


def test_empty_string():
    result = compress("")

    assert result["original_text"] == ""
    assert result["compressed_data"] == ""
    assert result["original_bits"] == 0
    assert result["compressed_bits"] == 0
    assert result["compression_ratio"] == 0.0


def test_single_character():
    result = compress("A")

    assert result["original_text"] == "A"
    assert result["original_bits"] == 8
    assert result["compressed_bits"] >= 8
    assert all(bit in {"0", "1"} for bit in result["compressed_data"])


def test_repeated_single_character():
    text = "AAAAAA"
    result = compress(text)

    assert result["original_text"] == text
    assert result["original_bits"] == len(text) * 8
    assert result["compressed_bits"] == len(result["compressed_data"])


def test_mixed_characters():
    text = "HELLOchinMAY123"
    result = compress(text)

    assert result["original_text"] == text
    assert result["original_bits"] == len(text) * 8
    assert isinstance(result["compression_ratio"], float)


def test_whitespace_and_symbols():
    text = "A B!\n"
    result = compress(text)

    assert result["original_text"] == text
    assert result["original_bits"] == len(text) * 8
    assert all(bit in {"0", "1"} for bit in result["compressed_data"])