from utils import load_gemma_saes


def test_load_gemma_saes():
    saes = load_gemma_saes("2b", [9])
    assert len(saes) == 1
    assert saes[9] is not None
