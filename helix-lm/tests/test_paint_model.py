# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Pinxit, the trained hand on the Helix trunk: it learns a practice sheet, paints through the engine, keeps a
memory of each place, saves and resumes; the eye and its adapters; and no step mixes CPU and GPU tensors."""

import json

import pytest
import torch

from helix_lm.anansi import AnansiModel
from helix_lm.paint import Painting, paint
from helix_lm.paint.emotion import feeling_of
from helix_lm.paint.engine import plan_levels
from helix_lm.paint.eye import LoRALinear, TinyEye, add_lora, eye_of, load_lora, lora_rank, lora_state
from helix_lm.paint.mind import Settings, make_plan
from helix_lm.paint.model import FEEL, ModelHand, Pinxit, PinxitConfig, feeling_vector, load_hand
from helix_lm.paint.style import StyleCard, own_card
from helix_lm.paint.synth import practice_sheet, sheet_loss
from helix_lm.paint.train import (
    PlanPool,
    largest_batch,
    load_state,
    open_pass,
    picture_batch,
    picture_step,
    save_state,
    train_pictures,
    train_sheets,
    training,
)


def _picture():
    g = torch.Generator().manual_seed(0)
    img = torch.rand(3, 96, 96, generator=g) * 0.1 + torch.tensor([0.3, 0.4, 0.5])[:, None, None]
    img[:, 30:60, 20:50] = torch.tensor([0.9, 0.6, 0.2])[:, None, None]
    return img.clamp(0, 1)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return Pinxit(PinxitConfig.named("px-tiny", eye_dim=TinyEye().dim))


def test_the_model_is_built_on_the_helix_trunk_and_answers_a_sheet(tiny):
    assert isinstance(tiny.trunk, AnansiModel) and feeling_vector(feeling_of("awe")).shape == (FEEL,)
    c = tiny.config
    sheet = practice_sheet(4, c.strokes, c.palette, torch.Generator().manual_seed(1))
    out = tiny(sheet["target"], sheet["canvas"], sheet["brief"], sheet["palette"], StyleCard().vector(),
               feeling_vector(feeling_of(None)))
    s = out["strokes"]
    assert s.geom.shape == (4, c.strokes, 10) and s.color.shape == (4, c.strokes, 3)
    assert float(s.geom.min()) >= 0 and float(s.geom.max()) <= 1 and float(s.color.min()) >= 0
    assert torch.allclose(s.brush.sum(-1), torch.ones(4, c.strokes), atol=1e-5)
    assert out["memory"].shape == (4, c.memory_dim) and out["mix"].shape == (4, c.strokes, 2, c.palette)
    loss = sheet_loss(out, sheet)
    assert all(torch.isfinite(v) for v in loss.values())
    loss["total"].backward()
    assert all(p.grad is not None for n, p in tiny.named_parameters() if n.startswith(("geom", "keep", "view.0")))


def test_it_learns_one_sheet_by_heart():
    torch.manual_seed(0)
    model = Pinxit(PinxitConfig.named("px-tiny")).train()
    c = model.config
    sheet = practice_sheet(8, c.strokes, c.palette, torch.Generator().manual_seed(2))
    style, feel = StyleCard().vector(), feeling_vector(feeling_of(None))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(70):
        loss = sheet_loss(model(sheet["target"], sheet["canvas"], sheet["brief"], sheet["palette"], style, feel,
                                loops=1), sheet)
        first = {k: float(v) for k, v in loss.items()} if first is None else first
        opt.zero_grad()
        loss["total"].backward()
        opt.step()
    assert float(loss["total"]) < 0.8 * first["total"] and float(loss["exists"]) < 0.6 * first["exists"]


def test_training_saves_and_goes_on(tmp_path):
    torch.manual_seed(0)
    model = Pinxit(PinxitConfig.named("px-tiny"))
    said = []
    done = train_sheets(model, steps=3, batch=4, checkpoint=tmp_path / "latest.pt", log=said.append)
    assert done["step"] == 3 and (tmp_path / "latest.json").exists()
    again = Pinxit(PinxitConfig.named("px-tiny"))
    more = train_sheets(again, steps=5, batch=4, checkpoint=tmp_path / "latest.pt", log=said.append)
    assert more["step"] == 5 and any("went on from step 3" in line for line in said)
    side = json.loads((tmp_path / "latest.json").read_text())
    assert side["progress"]["sheets"]["step"] == 5 and side["total_hours"] == side["hours"]
    loaded = Pinxit.load(tmp_path / "latest.pt")                              # the averaged weights, for painting
    assert not loaded.training and loaded.config.name == "px-tiny"
    got = load_state(tmp_path / "latest.pt", Pinxit(PinxitConfig.named("px-tiny")))
    assert (got["step"], got["stage"], list(got["progress"])) == (5, "sheets", ["sheets"])
    # each stage counts for itself: pictures starts at its own zero, and sheets is still at five after it
    plans = _plans(2)
    after = train_pictures(Pinxit(PinxitConfig.named("px-tiny")), iter(plans), steps=2, cells=32, pictures=2,
                           checkpoint=tmp_path / "latest.pt", log=said.append)
    assert after["step"] == 2 and any("[pictures] starts from the model as sheets left it" in line for line in said)
    back = train_sheets(Pinxit(PinxitConfig.named("px-tiny")), steps=6, batch=4, checkpoint=tmp_path / "latest.pt",
                        log=said.append)
    assert back["step"] == 6 and any("[sheets] went on from step 5" in line for line in said)
    side = json.loads((tmp_path / "latest.json").read_text())
    assert side["progress"]["pictures"]["step"] == 2 and side["progress"]["sheets"]["step"] == 6
    stopped = train_sheets(Pinxit(PinxitConfig.named("px-tiny")), steps=50, batch=4, keep_going=lambda: False,
                           log=said.append)
    assert stopped["step"] == 0                                               # a session about to end: no step taken


def test_the_model_paints_and_remembers(tiny, tmp_path):
    eye = TinyEye()
    hand = ModelHand(tiny, eye, polish=0.2)
    painting = paint(_picture(), emotion="awe", hand=hand, eye=eye, work=128, seed=2)
    levels = plan_levels(128, 128)
    assert len(painting) > 20 and painting.history[0]["hand"] == "pinxit"
    assert set(painting.memory) == {lv.index for lv in levels}
    assert painting.memory[1].shape == (4, tiny.config.memory_dim) and float(painting.memory[1].abs().sum()) > 0
    painting.save(tmp_path / "m.pinxit")
    back = Painting.load(tmp_path / "m.pinxit")
    assert torch.equal(back.memory[2], painting.memory[2])
    remembered = back.memory[3].clone()
    back.touch((0.1, 0.1, 0.5, 0.5), hand=hand)                               # worked again, with its memory
    assert back.history[-1]["op"] == "touch" and not torch.equal(back.memory[3], remembered)


def test_a_pass_of_a_real_picture_trains_the_model_and_the_eye(tiny):
    img = _picture()
    plan = make_plan(img, own_card(img), feeling_of("joy"), Settings(work=128))
    levels = plan_levels(*plan.target.shape[-2:])
    eye = TinyEye()
    for p in eye.parameters():
        p.requires_grad_(True)
    g = torch.Generator().manual_seed(0)
    seen = set()
    for index in (0, 2, 3):
        loss = picture_step(tiny, plan, levels, index, g, eye)
        assert loss is not None and all(bool(torch.isfinite(torch.as_tensor(v))) for v in loss.values())
        tiny.zero_grad()
        eye.zero_grad()
        loss["total"].backward()
        seen |= {n for n, p in tiny.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
        assert float(sum(p.grad.abs().sum() for p in eye.parameters() if p.grad is not None)) > 0
    assert {"memory_in.weight", "eye_in.weight", "geom.weight"} <= seen


def test_adapters_are_the_only_thing_trained_in_the_eye(tmp_path):
    net = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(8, 8), "mlp": torch.nn.Linear(8, 8)})
    before = net["q_proj"](torch.ones(8))
    assert add_lora(net, rank=2) == 1 and isinstance(net["q_proj"], LoRALinear)
    assert torch.allclose(net["q_proj"](torch.ones(8)), before)               # starts as the model it was
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    assert sorted(trainable) == ["q_proj.down", "q_proj.up"]
    net["q_proj"].up.data.fill_(0.5)
    state = lora_state(net)
    assert set(state) == {"q_proj.down", "q_proj.up"}
    pooled = torch.nn.ModuleDict({"attention": torch.nn.MultiheadAttention(8, 2, batch_first=True),
                                  "q_proj": torch.nn.Linear(8, 8)})
    assert add_lora(pooled, rank=2) == 1                          # torch's attention reads its weights itself
    x = torch.rand(1, 3, 8)
    assert pooled["attention"](x, x, x)[0].shape == (1, 3, 8)
    other = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(8, 8), "mlp": torch.nn.Linear(8, 8)})
    add_lora(other, rank=2)
    load_lora(other, state)
    assert torch.equal(other["q_proj"].up, net["q_proj"].up)
    model = Pinxit(PinxitConfig.named("px-tiny", eye_dim=TinyEye().dim))
    eye = TinyEye()
    save_state(tmp_path / "c.pt", model, None, None, 7, 0.5, "pictures", eye)
    assert load_state(tmp_path / "c.pt", model, eye=eye)["stage"] == "pictures"
    hand, again = load_hand(tmp_path / "c.pt", polish=0.1)       # the painter and the eye it was trained with
    assert isinstance(hand, ModelHand) and isinstance(again, TinyEye) and hand.eye is again
    f = eye.features(torch.rand(3, 50, 70))
    assert f["dense"].shape[0] == eye.dim and eye.saliency(torch.rand(3, 50, 70)).dim() == 2
    many = eye.features_many([torch.rand(3, 50, 70), torch.rand(3, 90, 40)])   # pictures of any sizes at once
    assert many["dense"].shape[:2] == (2, eye.dim) and many["pooled"].shape == (2, eye.dim)
    assert eye.text(["a red boat"]).shape == (1, eye.dim)
    assert lora_rank(state) == 2 and eye_of({}) is None
    layer = net["q_proj"]
    with torch.no_grad():
        changed = layer(torch.ones(8))
    wrapper = torch.nn.ModuleDict({"net": net})
    switched = type("E", (TinyEye,), {})()                       # any eye can set its adapters aside
    switched.net = wrapper
    with switched.as_pretrained(), torch.no_grad():
        assert torch.allclose(layer(torch.ones(8)), before) and not torch.allclose(changed, before)
    assert layer.active


def test_the_pretrained_eye_wrapper_reads_real_model_classes():
    try:
        import transformers
    except ImportError as err:                                   # not installed, or a checkout that will not load
        pytest.skip(f"transformers does not import here: {str(err).splitlines()[0][:80]}")
    from helix_lm.paint.eye import HubEye

    img = torch.rand(3, 80, 100)
    dino = HubEye("tiny-dino", model=transformers.Dinov2Model(transformers.Dinov2Config(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=56, patch_size=14)))
    f = dino.features(img)
    assert f["dense"].shape == (32, 4, 4) and f["pooled"].shape == (32,) and not dino.has_text
    assert add_lora(dino.model, 4) >= 3 and dino.features(img)["dense"].shape == (32, 4, 4)
    sig = HubEye("tiny-siglip", model=transformers.SiglipModel(transformers.SiglipConfig(
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2,
                       "image_size": 64, "patch_size": 16},
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2,
                     "vocab_size": 100, "max_position_embeddings": 64})))
    assert sig.features(img)["dense"].shape == (32, 4, 4) and sig.has_text
    about = sig.saliency(img)                                     # which part the picture is about, part by part
    assert about.shape == (8, 8) and float(about.min()) == 0.0 and 0.99 < float(about.max()) <= 1.0
    assert about.device.type == "cpu"
    words = torch.nn.functional.normalize(torch.randn(32), dim=0)
    plain = sig.score(img[None], words)
    assert plain.shape == (1,)
    assert add_lora(sig.vision, 4) == 4                           # the part that sees, not its pooling attention
    assert not [n for n in lora_state(sig) if "text_model" in n]
    for layer in sig.modules():
        if isinstance(layer, LoRALinear):
            layer.up.data.normal_(std=0.5)
    assert torch.allclose(sig.score(img[None], words), plain)     # words are judged by the eye as it came
    picture = img[None].clone().requires_grad_(True)
    sig.score(picture, words).sum().backward()                    # words can be painted in
    assert float(picture.grad.abs().sum()) > 0
    sig.features_many([img, img[:, :40]])["dense"].sum().backward()                     # adapters can be trained
    assert sum(float(p.grad.abs().sum()) for p in sig.model.parameters() if p.grad is not None) > 0


def _plans(count, work=96):
    out = []
    for i in range(count):
        g = torch.Generator().manual_seed(i)
        img = (torch.rand(3, work, work + 32 * (i % 2), generator=g) * 0.2 + 0.3).clamp(0, 1)
        img[:, 20:50, 10 + 8 * i:40 + 8 * i] = torch.rand(3, 1, 1, generator=g)
        out.append(make_plan(img, own_card(img, seed=i), feeling_of(("awe", "joy", None)[i % 3]),
                             Settings(work=work, seed=i)))
    return out


def test_a_step_takes_several_pictures_and_stays_inside_its_memory(tiny):
    plans = _plans(4)
    eye = TinyEye()
    g = torch.Generator().manual_seed(0)
    passes = [open_pass(p, g, torch.device("cpu"), index) for p, index in zip(plans, (2, 1, 2, 0), strict=True)]
    assert all(p is not None for p in passes)
    held = sum(p.cells for p in passes)
    with training(tiny):
        full = picture_batch(tiny, passes, torch.Generator().manual_seed(1), eye)
        assert full["pictures"] == 4.0 and full["cells"] == float(held)
        tiny.zero_grad()
        full["total"].backward()
        whole = sum(float(p.grad.abs().sum()) for p in tiny.parameters() if p.grad is not None)
        # the same step with room for a third of the cells: the same loss, gradients from fewer cells
        some = picture_batch(tiny, passes, torch.Generator().manual_seed(1), eye, cells=held // 3)
        assert torch.isfinite(some["total"]) and some["cells"] == float(held)
        tiny.zero_grad()
        some["total"].backward()
        part = sum(float(p.grad.abs().sum()) for p in tiny.parameters() if p.grad is not None)
    assert 0 < part < whole and not tiny.training                 # and the model is left as it was found
    assert not tiny.trunk.config.activation_checkpointing
    assert picture_batch(tiny, [None], g, eye) is None


def test_recomputing_activations_gives_the_same_gradients():
    torch.manual_seed(0)
    model = Pinxit(PinxitConfig.named("px-tiny")).train()
    c = model.config
    sheet = practice_sheet(4, c.strokes, c.palette, torch.Generator().manual_seed(3))
    style, feel = StyleCard().vector(), feeling_vector(feeling_of(None))

    def gradients():
        model.zero_grad()
        out = model(sheet["target"], sheet["canvas"], sheet["brief"], sheet["palette"], style, feel, loops=2)
        sheet_loss(out, sheet)["total"].backward()
        return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    plain = gradients()
    with training(model):
        assert model.trunk.config.activation_checkpointing
        saved = gradients()
    assert torch.allclose(plain, saved, rtol=1e-4, atol=1e-6)


def test_the_step_size_is_fitted_to_the_gpu_and_a_step_out_of_memory_is_survived(tmp_path):
    total = 96e9
    seen = []

    def card(base, each, most=10 ** 9):                          # a GPU that needs base + each * size bytes
        def measure(size):
            seen.append(size)
            real = min(size, most)
            need = base + each * real
            return (None if need > total else need), real
        return measure

    said = []
    cpu = torch.device("cpu")
    assert largest_batch(None, 256, 16384, cpu) == 256                                   # no GPU: as asked
    fit = largest_batch(None, 256, 1 << 16, cpu, said.append, measure=card(4e9, 40e6), total=total)
    assert 0.80 * total < 4e9 + 40e6 * fit <= 0.93 * total and "88%" in said[-1]         # filled, not overfilled
    small = largest_batch(None, 256, 1 << 16, cpu, said.append, measure=card(4e9, 600e6), total=total)
    assert small == 128 and 4e9 + 600e6 * small <= 0.88 * total                          # 256 did not fit: halved
    capped = largest_batch(None, 256, 1 << 16, cpu, said.append, measure=card(1e9, 1e6, most=300), total=total)
    assert capped == 300                                         # no more to give than 300: that is the answer
    assert largest_batch(None, 256, 512, cpu, said.append, measure=card(1e9, 1e6), total=total) == 512
    with pytest.raises(RuntimeError, match="does not fit"):
        largest_batch(None, 256, 1024, cpu, said.append, measure=card(95e9, 1e9), total=total)

    model = Pinxit(PinxitConfig.named("px-tiny"))
    plans = _plans(3)
    pool = PlanPool(iter(plans), size=8, reuse=2)
    drawn = [pool.draw() for _ in range(12)]
    assert pool.made == 3 and {id(p) for p in drawn} == {id(p) for p in plans} and "3 plans made" in pool.note()
    fails = {"left": 2}
    real_batch = picture_batch

    def flaky(*args, **kwargs):
        if fails["left"]:
            fails["left"] -= 1
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (a test)")
        return real_batch(*args, **kwargs)

    import helix_lm.paint.train as train

    train.picture_batch = flaky
    try:
        done = train_pictures(model, iter(plans), steps=3, cells=64, pictures=2, checkpoint=tmp_path / "p.pt",
                              log=said.append)
    finally:
        train.picture_batch = real_batch
    assert done["step"] == 3 and sum("ran out of GPU memory" in line for line in said) == 2
    assert any("a step now takes 45 cells" in line for line in said)                    # 64 -> 54 -> 45
    assert (tmp_path / "p.pt").exists() and not model.trunk.config.activation_checkpointing


def test_no_step_mixes_cpu_and_gpu_tensors():
    """The same check as ``test_gpu_paths``: the model's weights are tagged as if on the GPU and every op that
    would mix them with a CPU tensor is reported."""
    from test_gpu_paths import DeviceCheck, OnGpu, _on_gpu

    torch.manual_seed(0)
    eye = _on_gpu(TinyEye())
    model = _on_gpu(Pinxit(PinxitConfig.named("px-tiny", eye_dim=eye.dim)))
    c = model.config
    img = _picture()
    plan = make_plan(img, own_card(img), feeling_of("awe"), Settings(work=128))
    levels = plan_levels(*plan.target.shape[-2:])
    g = torch.Generator().manual_seed(0)
    with DeviceCheck() as check:
        sheet = practice_sheet(4, c.strokes, c.palette, g, device=torch.device("cpu"))
        sheet = {k: (v.as_subclass(OnGpu) if isinstance(v, torch.Tensor) else v) for k, v in sheet.items()}
        out = model(sheet["target"], sheet["canvas"], sheet["brief"], sheet["palette"],
                    StyleCard().vector().as_subclass(OnGpu), feeling_vector(feeling_of(None)).as_subclass(OnGpu))
        sheet_loss(out, sheet)["total"].backward()
        for index in (1, 3):
            picture_step(model, plan, levels, index, g, eye)["total"].backward()
        passes = [open_pass(p, g, torch.device("cpu")) for p in _plans(3)]
        with training(model):                                      # several pictures, some cells without gradients
            picture_batch(model, passes, g, eye, cells=8)["total"].backward()
        with torch.no_grad():                                      # the fit switches gradients back on itself
            paint(img, emotion="awe", hand=ModelHand(model.eval(), eye, polish=0.1), eye=eye, work=128)
    assert not check.found, sorted(check.found.items())


def test_painting_works_with_gradients_switched_off_and_says_why_it_cannot_under_inference_mode():
    img = _picture()
    with torch.no_grad():
        painting = paint(img, work=96, seed=1)
    assert len(painting) > 10
    with torch.inference_mode(), pytest.raises(RuntimeError, match="outside torch.inference_mode"):
        paint(img, work=96, seed=1)
