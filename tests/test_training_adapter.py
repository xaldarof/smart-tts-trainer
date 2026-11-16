"""Training adapter unit tests (no XTTS weights needed)."""
import copy
import json

import torch

from app.services.training import manager
from app.services.training.xtts_runner import _patch_resume_config, enable_gradient_checkpointing


def _tiny_gpt2():
    from transformers import GPT2Config, GPT2Model

    torch.manual_seed(0)
    m = GPT2Model(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=8, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
    del m.wte   # XTTS removes the token embedding and feeds inputs_embeds
    return m.train()


def test_gradient_checkpointing_matches_plain_backward():
    a = _tiny_gpt2()
    b = copy.deepcopy(a)
    assert enable_gradient_checkpointing(b) >= 2
    x = torch.randn(2, 8, 32)
    a(inputs_embeds=x).last_hidden_state.pow(2).mean().backward()
    b(inputs_embeds=x).last_hidden_state.pow(2).mean().backward()
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)
    assert all(getattr(blk, "gradient_checkpointing", False) for blk in b.h)


def test_patch_resume_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"batch_size": 4, "epochs": 10, "other": 1}))
    _patch_resume_config(tmp_path, {"batch_size": 2, "num_workers": 0, "epochs": 12, "precision": "bf16"})
    c = json.loads((tmp_path / "config.json").read_text())
    assert c["batch_size"] == 2 and c["eval_batch_size"] == 1 and c["epochs"] == 12 and c["other"] == 1
    assert c["mixed_precision"] is True and c["precision"] == "bf16"


def test_failure_explanations():
    assert "out of memory" in manager._explain_failure("torch.OutOfMemoryError: CUDA out of memory", 1).lower()
    assert "disk" in manager._explain_failure("OSError: [Errno 28] No space left on device", 1).lower()
    assert "incompatible" in manager._explain_failure("size mismatch for gpt.text_embedding", 1).lower()
    assert "code 3" in manager._explain_failure("something odd", 3)


def test_has_checkpoint(tmp_path):
    assert not manager._has_checkpoint(tmp_path)
    (tmp_path / "checkpoint_10.pth").write_bytes(b"x")
    assert manager._has_checkpoint(tmp_path)
