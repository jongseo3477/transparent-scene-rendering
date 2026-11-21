"""Audit split membership, view counts and equal-scene means before reporting."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import yaml


def aggregate(configs,runs):
    metrics=["psnr","ssim","lpips_glint","lpips_standard"]
    rows=[]
    all_views=[]
    for conf in sorted(configs.glob("*.yaml")):
        if len(conf.stem)!=64:
            continue
        cfg=yaml.safe_load(conf.read_text())
        ids=cfg["val_dataloader_cfg"]["dataset_cfg"]["view_sample"]
        train=cfg["dataloader_cfg"]["dataset_cfg"]["view_sample"]
        folder=runs/"fusion"/conf.stem[:4]
        report=json.loads((folder/"official.json").read_text())
        cal=json.loads((folder/"calibration.json").read_text())
        views=report["views"]
        assert report["scene"]==conf.stem and [r["view"] for r in views]==ids
        assert not set(ids)&set(train)
        assert set(cal["fit"]+cal["check"])<=set(train)
        assert not set(cal["fit"])&set(cal["check"])
        assert report["choice"]==cal["choice"]==max(cal["means"],key=cal["means"].get)
        assert len(views)==report["n_test"] and len(train)==report["n_train"]
        means={k:statistics.mean(r[k] for r in views) for k in metrics}
        assert all(abs(means[k]-report["mean"][k])<1e-6 for k in metrics)
        rows.append(dict(scene=conf.stem,views=len(views),resolution=report["resolution"],
                         selected_model=report["choice"],**means,
                         official_json_sha256=hashlib.sha256((folder/"official.json").read_bytes()).hexdigest()))
        all_views.extend(views)
    assert len(rows)==8
    return dict(scenes=rows,n_views=len(all_views),
                scene_average={k:statistics.mean(r[k] for r in rows) for k in metrics},
                pooled_view_average={k:statistics.mean(r[k] for r in all_views) for k in metrics},
                paper_table_2={"psnr":30.21,"ssim":.92,"lpips":.11})


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--configs",type=Path,default=Path(__file__).parent/"configs")
    cli.add_argument("--runs",type=Path,required=True)
    args=cli.parse_args()
    result=aggregate(args.configs,args.runs)
    (args.runs/"summary.json").write_text(json.dumps(result,indent=2))
    print(result["n_views"],result["scene_average"],flush=True)
