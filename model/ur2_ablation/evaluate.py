import os, json, pickle, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from . import config as cfg
from .dataset import UR2Dataset
from .model import UR2AblationModel

NAMES=["DAPI","CD3","panCK"]; EPS=1e-12


def get_dirs(split):
    if split=="val": return cfg.VAL_PKL_DIR,cfg.VAL_UR_DIR
    if split=="test": return cfg.TEST_PKL_DIR,cfg.TEST_UR_DIR
    raise ValueError(f"EVAL_SPLIT must be val/test, got {split}")


def load_model(device):
    ck=torch.load(cfg.EVAL_CHECKPOINT,map_location="cpu"); state=ck["model"] if "model" in ck else ck; state={k[7:] if k.startswith("module.") else k:v for k,v in state.items()}
    model=UR2AblationModel(cfg.DIM,cfg.HEADS,cfg.FFN_DIM,cfg.DROPOUT).to(device); model.load_state_dict(state,strict=True); model.eval(); return model,ck.get("epoch",-1)


def train_mean():
    v=[]
    for f in sorted(os.listdir(cfg.TRAIN_PKL_DIR)):
        if f.endswith(".pkl"):
            with open(os.path.join(cfg.TRAIN_PKL_DIR,f),"rb") as h: v.append(np.asarray(pickle.load(h)["concept_scores"],dtype=np.float64))
    return np.stack(v).mean(0)


def pearson(x,y):
    x,y=np.asarray(x),np.asarray(y); x=x-x.mean(); y=y-y.mean(); den=np.sqrt(np.sum(x*x)*np.sum(y*y)); return float(np.sum(x*y)/den) if den>EPS else None


def metrics(pred,target,base):
    mse=float(np.mean((pred-target)**2)); b=float(np.mean((base-target)**2)); ssr=float(np.sum((pred-target)**2)); sst=float(np.sum((target-target.mean())**2)); ts=float(target.std())
    return {"mse":mse,"mae":float(np.mean(np.abs(pred-target))),"rmse":float(np.sqrt(mse)),"pearson":pearson(pred,target),"r2":1-ssr/sst if sst>EPS else None,"target_mean":float(target.mean()),"target_std":ts,"pred_mean":float(pred.mean()),"pred_std":float(pred.std()),"pred_std_over_target_std":float(pred.std()/ts) if ts>EPS else None,"train_mean_baseline":float(base),"baseline_mse":b,"mse_over_baseline":mse/b if b>EPS else None,"baseline_improvement_percent":100*(1-mse/b) if b>EPS else None}


def main():
    split=cfg.EVAL_SPLIT; pkl_dir,ur_dir=get_dirs(split); ds=UR2Dataset(pkl_dir,ur_dir); loader=DataLoader(ds,batch_size=cfg.BATCH_SIZE,shuffle=False,num_workers=cfg.NUM_WORKERS,pin_memory=True)
    device=torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu"); model,epoch=load_model(device); preds=[]; targets=[]; names=[]
    with torch.inference_mode():
        for b in loader:
            preds.append(model(b["ur"].to(device,non_blocking=True),b["coord"].to(device,non_blocking=True)).cpu().numpy()); targets.append(b["concept"].numpy()); names.extend(b["sample_name"])
    pred=np.concatenate(preds).astype(np.float64); target=np.concatenate(targets).astype(np.float64); tm=train_mean(); base=np.broadcast_to(tm,target.shape)
    mse=float(np.mean((pred-target)**2)); mae=float(np.mean(np.abs(pred-target))); bmse=float(np.mean((base-target)**2)); per={NAMES[i]:metrics(pred[:,i],target[:,i],tm[i]) for i in range(3)}
    report={"ablation":"UR2-only","checkpoint":cfg.EVAL_CHECKPOINT,"checkpoint_epoch":epoch,"split":split,"samples":len(ds),"overall":{"mse":mse,"mae":mae,"rmse":float(np.sqrt(mse)),"train_mean_baseline_mse":bmse,"mse_over_baseline":mse/bmse,"baseline_improvement_percent":100*(1-mse/bmse)},"train_concept_mean":tm.tolist(),"per_concept":per}
    tag=os.path.splitext(os.path.basename(cfg.EVAL_CHECKPOINT))[0]; out=os.path.join(cfg.OUTPUT_DIR,"evaluation",f"{tag}_{split}"); os.makedirs(out,exist_ok=True)
    with open(os.path.join(out,"ur2_evaluation.json"),"w") as f: json.dump(report,f,indent=2)
    with open(os.path.join(out,"predictions.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["sample_name"]+[f"target_{x}" for x in NAMES]+[f"pred_{x}" for x in NAMES]); [w.writerow([n,*t,*p]) for n,t,p in zip(names,target,pred)]
    print("="*78); print(f"UR2-only | checkpoint={cfg.EVAL_CHECKPOINT} | epoch={epoch} | split={split} | N={len(ds)}"); print(f"Overall MSE={mse:.6f} | MAE={mae:.6f} | baseline={bmse:.6f} | improvement={100*(1-mse/bmse):.1f}%"); print("-"*78)
    for name in NAMES:
        m=per[name]; print(f"{name:5s} | MSE={m['mse']:.6f} MAE={m['mae']:.6f} r={m['pearson']:.4f} R2={m['r2']:.4f} std={m['pred_std']:.4f}/{m['target_std']:.4f} improve={m['baseline_improvement_percent']:.1f}%")
    print(f"Saved: {out}"); print("="*78)


if __name__=="__main__": main()
