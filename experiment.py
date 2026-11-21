"""Train and evaluate a calibrated, OptiX-free image-based view renderer."""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml


def read_points(path):
    types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}
    with path.open("rb") as f:
        props = []
        for line in iter(f.readline, b""):
            words = line.decode().split()
            if words[:2] == ["element", "vertex"]:
                count = int(words[2])
            if words[:1] == ["property"]:
                props.append((words[2], types[words[1]]))
            if words == ["end_header"]:
                break
        pts = np.fromfile(f, dtype=props, count=count)
    return np.stack([pts[k] for k in "xyz"], axis=1)[::3]


def grid(h, w):
    y, x = np.indices((h, w), dtype=np.float32)
    return np.stack((x, y, np.ones_like(x)), -1)


def blur(a, sigma):
    return cv2.GaussianBlur(a, (0, 0), sigma)


def remap(image, u, v):
    return cv2.remap(image, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class Scene:
    def __init__(self, root, config):
        self.root = root
        spec = yaml.safe_load(config.read_text())
        train = spec["dataloader_cfg"]["dataset_cfg"]
        self.train = set(train["view_sample"])
        self.test = set(spec["val_dataloader_cfg"]["dataset_cfg"]["view_sample"])
        assert not self.test & self.train
        self.K, self.R, self.C, self.names, self.undistort = {}, {}, {}, {}, {}
        intr = cv2.FileStorage(str(root / "intri.yml"), cv2.FILE_STORAGE_READ)
        extr = cv2.FileStorage(str(root / "extri.yml"), cv2.FILE_STORAGE_READ)
        ratio = train["ratio"]
        for i in sorted(self.train | self.test):
            n = intr.getNode("names").at(i).string()
            self.names[i] = n
            w = int(intr.getNode("W_" + n).real())
            h = int(intr.getNode("H_" + n).real())
            K = intr.getNode("K_" + n).mat()
            D = intr.getNode("D_" + n).mat()
            if D is not None and np.any(D):
                new, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
                self.undistort[i] = (K, D, new)
                K = new
            self.w, self.h = int(w * ratio), int(h * ratio)
            self.K[i] = (np.diag([ratio, ratio, 1]) @ K).astype("f")
            R = cv2.Rodrigues(extr.getNode("R_" + n).mat())[0]
            self.R[i] = R.astype("f")
            self.C[i] = (-R.T @ extr.getNode("T_" + n).mat()).ravel().astype("f")
        intr.release()
        extr.release()
        self.pts = read_points(root / "sparse/0/points3D.ply")

    @lru_cache(maxsize=24)
    def image(self, i):
        p = next((self.root / "images" / self.names[i]).glob("000000.*"))
        image = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if i in self.undistort:
            k, d, new = self.undistort[i]
            image = cv2.undistort(image, k, d, newCameraMatrix=new)
        return cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_AREA)

    def sources(self, target, allowed):
        assert target not in allowed and not allowed & self.test
        xyz = (self.pts - self.C[target]) @ self.R[target].T
        z = xyz[:, 2]
        keep = (z > 0) & (abs(xyz[:, 0]) < z) & (abs(xyz[:, 1]) < z * .7)
        depths = z[keep] if keep.sum() > 80 else z[z > 0]
        mid = np.median(depths)
        def distance(s):
            return (np.linalg.norm(self.C[s] - self.C[target]) / mid
                    + .7 * np.linalg.norm(self.R[s] - self.R[target]))
        src = sorted(allowed, key=distance)[:4]
        return src, min(1 / np.percentile(depths, 1) * 1.45, 12 / mid)


def render(scene, target, allowed):
    src, max_inv = scene.sources(target, allowed)
    h, w = scene.h, scene.w
    K = scene.K[target]
    A, B, images = [], [], []
    for s in src:
        A.append(scene.K[s] @ scene.R[s] @ scene.R[target].T @ np.linalg.inv(K))
        B.append(scene.K[s] @ scene.R[s] @ (scene.C[target] - scene.C[s]))
        images.append(scene.image(s))
    # Match candidate virtual depths using only source-to-source photometric error.
    small = (w // 2, h // 2)
    scale = np.diag([small[0] / w, small[1] / h, 1]).astype("f")
    costs = []
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    n = 40
    for d in np.linspace(0, max_inv, n):
        aligned = []
        for a, b, gray in zip(A, B, grays):
            mat = a.copy()
            mat[:, 2] += b * d
            warp = cv2.warpPerspective(gray, mat @ np.linalg.inv(scale), small,
                                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                       borderMode=cv2.BORDER_CONSTANT)
            aligned.append(warp.astype("f"))
        pairs = []
        for i, j in ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)):
            a, b = aligned[i], aligned[j]
            err = np.where((a == 0) | (b == 0), 50, np.minimum(abs(a - b), 50))
            pairs.append(blur(err, 1.05))
        ordered = np.sort(np.stack(pairs), axis=0)
        costs.append(.6 * ordered[1] + .3 * ordered[2] + .1 * ordered[3])
    cost = np.stack(costs)
    best = np.argmin(cost, axis=0)
    def get(ix):
        return np.take_along_axis(cost, ix[None], axis=0)[0]
    left, right, center = get((best-1).clip(0)), get((best+1).clip(max=n-1)), get(best)
    offset = np.clip((left-right) / (2 * np.maximum(left+right-2*center, .001)), -.5, .5)
    inv = cv2.resize(cv2.medianBlur(((best + offset) * max_inv / (n-1)).astype("f"), 3),
                     (w, h), interpolation=cv2.INTER_LINEAR)
    pix = grid(h, w)
    base = [pix @ a.T for a in A]
    def sample(d):
        colors, masks = [], []
        for im, ray, b in zip(images, base, B):
            p = ray + d[..., None] * b
            u, v = (p[:, :, j] / np.maximum(p[:, :, 2], .001) for j in (0, 1))
            masks.append((u >= 0) & (u <= w-1) & (v >= 0) & (v <= h-1) & (p[:, :, 2] > 0))
            colors.append(remap(im, u, v).astype("f") / 255)
        return np.stack(colors), np.stack(masks)
    # Refine disparity around the coarse estimate at the evaluation resolution.
    candidates, errors = [], []
    for off in (-2., -1., 0., 1., 2.):
        d = np.maximum(inv + off * max_inv / (n-1), 0)
        colors, masks = sample(d)
        es = []
        for i, j in ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)):
            err = np.minimum(np.mean(abs(colors[i]-colors[j]), axis=-1), .2)
            es.append(blur(np.where(masks[i] & masks[j], err, .2), 1))
        errors.append(np.sort(np.stack(es), axis=0)[1:3].mean(0))
        candidates.append(d)
    ix = np.argmin(np.stack(errors), axis=0)
    inv = np.take_along_axis(np.stack(candidates), ix[None], axis=0)[0]
    colors, valid = sample(inv)
    median = np.median(colors, axis=0)
    err = np.mean(abs(colors - median), -1)
    weights = np.exp(-err * 18) * valid * np.array([1., .85, .55, .4], "f")[:, None, None]
    total = weights.sum(0)
    fused = (colors * weights[..., None]).sum(0) / np.maximum(total[..., None], 1e-8)
    fused[total == 0] = colors[0][total == 0]
    # Linear residuals let each scene learn sharpness and reference blending.
    features = np.stack((fused-blur(fused, .8), fused-blur(fused, 1.8),
                         colors[0]-fused, median-fused), axis=-1)
    return fused, features, images[0].astype("f") / 255, inv


def psnr(a, b):
    return float(-10*np.log10(max(np.mean((a-b)**2), 1e-12)))


def ssim(a, b):
    def g(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)[5:-5, 5:-5]
    x, y = g(a), g(b)
    return float(np.mean(((2*x*y+.0001)*(2*(g(a*b)-x*y)+.0009)) /
                         ((x*x+y*y+.0001)*(g(a*a)+g(b*b)-x*x-y*y+.0009))))


def fit(scene, train, fitting):
    lhs, rhs = np.eye(4) * 2, np.zeros(4)
    for t in fitting:
        pred, feat, _, _ = render(scene, t, train - {t})
        x = feat[::4, ::4].reshape(-1, 4)
        y = (scene.image(t).astype("f")/255 - pred)[::4, ::4].reshape(-1)
        lhs += x.T @ x
        rhs += x.T @ y
    return np.linalg.solve(lhs, rhs).astype("f")


def run(scene, out, final):
    out.mkdir(parents=True, exist_ok=True)
    dev = sorted(scene.train)[3::19][:12]
    train = scene.train - set(dev)
    fitting = sorted(train)[::max(1, len(train)//12)][:12]
    coef = fit(scene, train, fitting)
    def evaluate(ids, refs, split, learned=True):
        rows = []
        for t in ids:
            base, x, nearest, inv = render(scene, t, refs)
            pred = np.clip(base + x @ coef, 0, 1) if learned else base
            gt = scene.image(t).astype("f") / 255
            rows.append(dict(view=t, nearest=psnr(nearest, gt), sweep=psnr(base, gt),
                             model=psnr(pred, gt), ssim=ssim(pred, gt)))
            if len(rows) == 1:
                panel = np.concatenate([nearest, base, pred, gt], axis=1)
                cv2.imwrite(str(out / (split + ".png")), np.rint(panel*255).astype("uint8"))
                np.save(out / (split + "_apparent_inverse_depth.npy"), inv)
        return rows
    val = evaluate(dev, train, "development")
    learned = sum(r["model"]-r["sweep"] for r in val) >= 0
    report = dict(scene=scene.root.name, resolution=[scene.w, scene.h], coefficients=coef.tolist(),
                  learning_targets=fitting, development=val, learned_selected=bool(learned))
    if final:
        report["test"] = evaluate(sorted(scene.test), scene.train, "test", learned)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    for key in ("development", "test"):
        if key in report:
            print(scene.root.name[:4], key, {k:round(float(np.mean([r[k] for r in report[key]])),4)
                  for k in ("nearest","sweep","model","ssim")}, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("scene", type=Path)
    cli.add_argument("--configs", type=Path, default=Path(__file__).parent/"configs")
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--final", action="store_true")
    args = cli.parse_args()
    cv2.setNumThreads(2)
    run(Scene(args.scene, args.configs / (args.scene.name+".yaml")), args.out, args.final)
