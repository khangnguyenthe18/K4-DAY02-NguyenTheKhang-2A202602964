"""Kiểm tra tự viết cho các phần dễ sai (RUBRIC H, C). Chạy được trên CPU, không cần dữ liệu thật:

    cd code && python -m unittest discover -s tests -v
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import dataset as D  # noqa: E402
import losses as Ls  # noqa: E402
import train as T  # noqa: E402
from benchmark import bench, latency_report  # noqa: E402
from eval import compute_metrics, ece_score, read_pred, save_predictions  # noqa: E402
from inference import (aggregate_views, apply_temperature, fit_temperature, fuse_conv_bn, softmax_np,  # noqa: E402
                       uniform_soup, views_multicrop, views_multiscale)
from model import build_model, count_gmacs, count_params, freeze_backbone, param_groups, set_train_mode  # noqa: E402


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits, self.y = torch.randn(256, 9) * 3, torch.randint(0, 9, (256,))

    def test_focal_gamma0_equals_ce(self):
        d = abs(Ls.FocalLoss(0.0)(self.logits, self.y).item() - F.cross_entropy(self.logits, self.y).item())
        self.assertLess(d, 1e-6)

    def test_focal_downweights_easy(self):
        self.assertLess(Ls.FocalLoss(2.0)(self.logits, self.y).item(), F.cross_entropy(self.logits, self.y).item())

    def test_focal_alpha(self):
        a = torch.rand(9)
        ref = (F.cross_entropy(self.logits, self.y, reduction="none") * a[self.y]).mean()
        self.assertAlmostEqual(Ls.FocalLoss(0.0, a)(self.logits, self.y).item(), ref.item(), places=5)

    def test_label_smoothing(self):
        self.assertAlmostEqual(Ls.LabelSmoothingCE(0.0)(self.logits, self.y).item(),
                               F.cross_entropy(self.logits, self.y).item(), places=5)
        self.assertAlmostEqual(Ls.LabelSmoothingCE(0.1)(self.logits, self.y).item(),
                               F.cross_entropy(self.logits, self.y, label_smoothing=0.1).item(), places=5)

    def test_class_weights(self):
        n = np.array([675, 638, 619, 613, 637, 605, 644, 610, 5464])
        w = Ls.class_weights(n, 0.0)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertAlmostEqual((w[0] * n[0]).item(), (w[8] * n[8]).item(), places=2)   # ∝ 1/n
        wb = Ls.class_weights(n, 0.999)
        self.assertAlmostEqual(wb.sum().item(), 9.0, places=4)
        self.assertGreater(wb[0].item(), wb[8].item())

    def test_build_criterion(self):
        for kind in ("ce", "ls", "focal"):
            self.assertTrue(math.isfinite(Ls.build_criterion(kind)(self.logits, self.y).item()))
        crit = Ls.build_criterion("ce_weighted", weight=torch.ones(9))
        self.assertAlmostEqual(crit(self.logits, self.y).item(), F.cross_entropy(self.logits, self.y).item(), places=5)
        with self.assertRaises(ValueError):
            Ls.build_criterion("ce_weighted")


class TestMix(unittest.TestCase):
    def test_cutmix_lambda_is_true_area(self):
        rng = np.random.default_rng(0)
        x = torch.zeros(8, 3, 32, 32)
        x[4:] = 1.0                       # nửa batch toàn 1, nửa toàn 0
        for _ in range(50):
            xm, (ya, yb, lam) = Ls.mix_batch(x, torch.arange(8), 1.0, "cutmix", rng=rng)
            # ở ảnh có hoán vị sang nhóm khác, tỉ lệ điểm ảnh bị thay = 1 - lam
            for i in range(8):
                if (i < 4) != (yb[i].item() < 4):
                    changed = (xm[i] != x[i]).float().mean().item()
                    self.assertAlmostEqual(changed, 1 - lam, places=5)
            self.assertTrue(0.0 <= lam <= 1.0)

    def test_mixup(self):
        x = torch.randn(6, 3, 8, 8)
        xm, (ya, yb, lam) = Ls.mix_batch(x, torch.arange(6), 0.4, "mixup", rng=np.random.default_rng(1))
        perm = yb                          # nhãn = chỉ số nên y[perm] = perm
        torch.testing.assert_close(xm, lam * x + (1 - lam) * x[perm])

    def test_mixed_loss(self):
        lg, ya, yb = torch.randn(4, 9), torch.tensor([0, 1, 2, 3]), torch.tensor([3, 2, 1, 0])
        ce = torch.nn.CrossEntropyLoss()
        v = Ls.mixed_loss(ce, lg, (ya, yb, 0.3)).item()
        self.assertAlmostEqual(v, 0.3 * ce(lg, ya).item() + 0.7 * ce(lg, yb).item(), places=5)


class TestModel(unittest.TestCase):
    def test_param_groups_no_wd_on_norm_bias(self):
        m = build_model("resnet18", pretrained=False)
        groups = param_groups(m, 1e-4, 1e-3, 0.05)
        head = {id(p) for p in m.get_classifier().parameters()}
        n_total = sum(p.numel() for p in m.parameters())
        self.assertEqual(sum(p.numel() for g in groups for p in g["params"]), n_total)
        for g in groups:
            for p in g["params"]:
                if p.ndim <= 1:
                    self.assertEqual(g["weight_decay"], 0.0)
                self.assertEqual(g["lr"], 1e-3 if id(p) in head else 1e-4)

    def test_vit_pos_embed_no_wd(self):
        m = build_model("test_vit", pretrained=False)
        groups = param_groups(m, 1e-4, 1e-3, 0.05)
        nd = [g for g in groups if g["name"] == "bb_no_decay"][0]
        self.assertTrue(any(p is m.pos_embed for p in nd["params"]))

    def test_freeze_keeps_bn_eval(self):
        m = build_model("resnet18", pretrained=False, init="frozen")
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        self.assertTrue(all(n.startswith("fc.") for n in trainable))
        set_train_mode(m)
        self.assertFalse(any(b.training for b in m.modules() if isinstance(b, torch.nn.BatchNorm2d)))
        self.assertTrue(m.fc.training)
        before = m.bn1.running_mean.clone()
        m(torch.randn(4, 3, 64, 64))
        torch.testing.assert_close(m.bn1.running_mean, before)
        self.assertEqual(len(param_groups(m, 1e-4, 1e-3, 0.05)), 2)  # chỉ nhóm head

    def test_counts(self):
        m = build_model("resnet18", pretrained=False)
        self.assertAlmostEqual(count_params(m), 11.18, delta=0.05)
        self.assertAlmostEqual(count_gmacs(m, 224), 1.82, delta=0.1)   # ResNet-18 ~1.8 GMAC


class TestInference(unittest.TestCase):
    def test_fuse_conv_bn_exact(self):
        for name in ("resnet18", "test_efficientnet"):
            m = build_model(name, pretrained=False).eval()
            for mod in m.modules():                       # thống kê BN khác mặc định
                if isinstance(mod, torch.nn.BatchNorm2d):
                    mod.running_mean.uniform_(-0.5, 0.5)
                    mod.running_var.uniform_(0.5, 2.0)
                    mod.weight.data.uniform_(0.5, 1.5)
            fm = fuse_conv_bn(m, check=False)
            x = torch.randn(2, 3, 96, 96)
            with torch.no_grad():
                d = (m(x) - fm(x)).abs().max().item()
            self.assertGreater(fm.n_fused, 5, name)
            self.assertLess(d, 1e-3, name)
            self.assertEqual(sum(isinstance(b, torch.nn.BatchNorm2d) for b in fm.modules()), 0, name)

    def test_temperature_recovers_T(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(5000, 9)) * 2
        p = softmax_np(z / 2.5)
        y = np.array([rng.choice(9, p=pi) for pi in p])
        self.assertAlmostEqual(fit_temperature(z, y), 2.5, delta=0.15)
        self.assertTrue((apply_temperature(z, 2.5).argmax(1) == z.argmax(1)).all())

    def test_ts_reduces_ece_on_overconfident(self):
        rng = np.random.default_rng(1)
        z = rng.normal(size=(4000, 9)) * 2
        y = np.array([rng.choice(9, p=pi) for pi in softmax_np(z)])
        z_hot = z * 3                                     # quá tự tin
        t = fit_temperature(z_hot, y)
        self.assertLess(ece_score(apply_temperature(z_hot, t), y), ece_score(softmax_np(z_hot), y))

    def test_aggregate(self):
        a, b = np.random.randn(10, 9), np.random.randn(10, 9)
        np.testing.assert_allclose(aggregate_views([a, b], "prob"), (softmax_np(a) + softmax_np(b)) / 2)
        np.testing.assert_allclose(aggregate_views([a, b], "logit"), softmax_np((a + b) / 2))

    def test_views(self):
        x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).view(2, 3, 8, 8)
        v = views_multicrop(x, 6, flip=True)
        self.assertEqual(len(v), 10)
        torch.testing.assert_close(v[4], x[..., 1:7, 1:7])
        self.assertEqual([t.shape[-1] for t in views_multiscale(x, [8, 12])], [8, 12])

    def test_soup_of_identical_is_identity(self):
        m = build_model("resnet18", pretrained=False)
        sd = uniform_soup([m.state_dict(), m.state_dict()])
        for k, v in m.state_dict().items():
            torch.testing.assert_close(sd[k], v)


class TestTrainHelpers(unittest.TestCase):
    def test_lr_schedule_shape(self):
        total, warm = 1000, 100
        f = [T.lr_factor(s, total, warm) for s in range(total)]
        self.assertAlmostEqual(f[0], 0.01)
        self.assertTrue(all(a < b for a, b in zip(f[:warm], f[1:warm])))   # tăng khi warmup
        self.assertAlmostEqual(f[warm], 1.0)
        self.assertTrue(all(a >= b for a, b in zip(f[warm:], f[warm + 1:])))  # giảm theo cosine
        self.assertLess(f[-1], 0.01)

    def test_ema(self):
        m = torch.nn.Linear(3, 2)
        e = T.EMA(m, 0.9)
        with torch.no_grad():
            m.weight.add_(1.0)
        for _ in range(500):
            e.update(m)
        torch.testing.assert_close(e.module.weight, m.weight, atol=1e-4, rtol=0)

    def test_parse_overrides(self):
        d = T.parse_overrides(["seed=3", "ema_decay=none", "amp=false", "lr_head=2e-3", "sampler=balanced",
                               "class_weight_beta=0.99"])
        self.assertEqual(d, {"seed": 3, "ema_decay": None, "amp": False, "lr_head": 2e-3, "sampler": "balanced",
                             "class_weight_beta": 0.99})
        with self.assertRaises(KeyError):
            T.parse_overrides(["nope=1"])

    def test_defaults_match_guide(self):
        c = T.Config()
        self.assertEqual((c.epochs, c.batch_size, c.lr_backbone, c.lr_head, c.weight_decay, c.save_test_predictions),
                         (12, 64, 1e-4, 1e-3, 0.05, False))

    def test_test_access_only_once(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = T.Config(exp_id="F01", seed=10, out_dir=d)
            T._log_test_access(cfg)
            with self.assertRaises(RuntimeError):
                T._log_test_access(cfg)


class TestBenchmark(unittest.TestCase):
    def test_bench_rules(self):
        with self.assertRaises(ValueError):
            bench(lambda: None, warmup=5)
        with self.assertRaises(ValueError):
            bench(lambda: None, iters=10)
        calls = []
        r = bench(lambda: calls.append(1), warmup=10, iters=50)
        self.assertEqual(len(calls), 60)
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])

    def test_latency_report_cpu(self):
        r = latency_report(build_model("test_resnet", pretrained=False), 1, 64, device="cpu", iters=50)
        self.assertEqual(r["n_iters"], 50)
        self.assertIn("p95_ms", r)


class TestDataset(unittest.TestCase):
    def _fake(self, d, n=90):
        from PIL import Image
        img_dir = Path(d) / "images"
        img_dir.mkdir()
        rows = []
        for i in range(n):
            f = f"img{i:04d}.jpg"
            Image.fromarray(np.full((256, 256, 3), i % 255, np.uint8)).save(img_dir / f)
            rows.append({"Filename": f, "Label": i % 9, "Species": D.CLASS_NAMES[i % 9]})
        df = pd.DataFrame(rows)
        return img_dir, df.iloc[:54], df.iloc[54:72], df.iloc[72:]

    def test_check_split_and_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            img_dir, tr, va, te = self._fake(d)
            rep = D.check_split(tr, va, te, img_dir, expected_total=90, verbose=False)
            self.assertEqual(rep["union"], 90)
            with self.assertRaises(AssertionError):     # rò rỉ: một ảnh nằm ở cả train và test
                D.check_split(tr, va, pd.concat([te, tr.iloc[:1]]), img_dir, expected_total=90, verbose=False)
            with self.assertRaises(AssertionError):     # file thiếu
                (img_dir / tr.Filename.iloc[0]).unlink()
                D.check_split(tr, va, te, img_dir, expected_total=90, verbose=False)

    def test_loader_and_cache_equivalent(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            img_dir, tr, va, te = self._fake(d)
            cache = D.ImageCache.build(list(va.Filename), img_dir, Path(d) / "cache")
            t = D.build_transforms(False, 224)
            a = D.DeepWeedsDataset(va, img_dir, t)[3]
            b = D.DeepWeedsDataset(va, img_dir, t, cache)[3]
            torch.testing.assert_close(a[0], b[0])
            self.assertEqual((a[1], a[2]), (b[1], b[2]))
            ld = D.make_loader(va, img_dir, t, 4, train=False, num_workers=0)
            names = [f for _, _, fs in ld for f in fs]
            self.assertEqual(names, va.Filename.tolist())   # giữ đúng thứ tự
            ldb = D.make_loader(tr, img_dir, D.build_transforms(True, 64, "trivial"), 8, train=True,
                                sampler="balanced", num_workers=0)
            x, y, _ = next(iter(ldb))
            self.assertEqual(tuple(x.shape), (8, 3, 64, 64))

    def test_eval_transform_deterministic(self):
        from PIL import Image
        im = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
        t = D.build_transforms(False, 224)
        torch.testing.assert_close(t(im), t(im))
        self.assertEqual(tuple(t(im).shape), (3, 224, 224))


class TestEvalContract(unittest.TestCase):
    def test_metrics_match_sklearn(self):
        from sklearn.metrics import f1_score
        rng = np.random.default_rng(0)
        y = rng.integers(0, 9, 500)
        z = rng.normal(size=(500, 9)) + np.eye(9)[y] * 2
        m = T.metrics_from_logits(y, z)
        self.assertAlmostEqual(m["macro_f1"], f1_score(y, z.argmax(1), average="macro"), places=10)

    def test_saved_predictions_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = softmax_np(np.random.randn(20, 9))
            path = save_predictions(Path(d) / "F01_seed10_test.csv", [f"{i}.jpg" for i in range(20)], np.arange(20) % 9, p)
            r = read_pred(str(path))
            self.assertEqual(r.seed, 10)
            np.testing.assert_allclose(r.probs, p, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
