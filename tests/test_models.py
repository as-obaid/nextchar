import pytest
import torch

from data import CharVocab, eval_windows, get_batch
from models import MODELS, generate
from train import HPARAMS

V, T = 11, 32  # small vocabulary; T fits inside the transformer's 128 positions


def build(name):
    torch.manual_seed(0)
    return MODELS[name](V, **HPARAMS[name]).eval()


@pytest.mark.parametrize("name", MODELS)
def test_logits_shape(name):
    x = torch.randint(V, (2, T))
    assert build(name)(x).shape == (2, T, V)


@pytest.mark.parametrize("name", MODELS)
def test_no_model_sees_the_future(name):
    model, k = build(name), 10
    x = torch.randint(V, (2, T))
    x2 = x.clone()
    x2[:, k:] = (x[:, k:] + 1) % V  # change every character from position k on
    with torch.no_grad():
        assert torch.allclose(model(x)[:, :k], model(x2)[:, :k], atol=1e-6)


def test_bigram_sees_only_the_current_character():
    model, t = build("bigram"), 7
    x = torch.randint(V, (1, T))
    x2 = (x + 1) % V
    x2[0, t] = x[0, t]  # same current character, different everything else
    with torch.no_grad():
        assert torch.equal(model(x)[0, t], model(x2)[0, t])


def test_context_matters_for_lstm_and_transformer():
    x = torch.randint(V, (1, T))
    x2 = x.clone()
    x2[0, 0] = (x[0, 0] + 1) % V  # change only the first character
    with torch.no_grad():
        for name in ("lstm", "transformer"):
            model = build(name)
            # A few positions later; an untrained LSTM forgets position 0 within ~30 steps.
            assert not torch.allclose(model(x)[0, 3], model(x2)[0, 3])


def test_vocab_round_trip():
    vocab = CharVocab("to be or not")
    assert len(vocab) == 7
    assert vocab.decode(vocab.encode("not to be")) == "not to be"


def test_targets_are_inputs_shifted_by_one():
    data = torch.arange(1000)
    x, y = get_batch(data, 4, 16, torch.Generator().manual_seed(0))
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y, x + 1)


def test_eval_windows_do_not_overlap_and_stay_in_range():
    data = torch.arange(1000)
    x, y = eval_windows(data, 16)
    assert torch.equal(x[:, 0], torch.arange(0, len(x) * 16, 16))
    assert y.max() < len(data)
    assert len(eval_windows(data, 16, max_windows=10)[0]) == 10


def test_generate_crops_to_the_context_length():
    model = build("transformer")  # would fail past 128 positions without cropping
    out = generate(model, torch.zeros(1, 5, dtype=torch.long), 200, block_size=128)
    assert out.shape == (1, 205)
