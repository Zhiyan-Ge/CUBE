import os, json, time, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from .dataset import UR1Dataset
from .model import UR1AblationModel
from . import config as cfg
from loss.concept_loss import concept_loss


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def select_samples(ds,n,seed):
    if n is None or n>=len(ds): return ds
    g=torch.Generator().manual_seed(seed); return Subset(ds,torch.randperm(len(ds),generator=g)[:n].tolist())


def build_dataset(pkl_dir,ur_dir,n,seed): return select_samples(UR1Dataset(pkl_dir,ur_dir),n,seed)


def build_loader(ds,shuffle):
    return DataLoader(ds,batch_size=cfg.BATCH_SIZE,shuffle=shuffle,num_workers=cfg.NUM_WORKERS,pin_memory=True)


def run_epoch(model,loader,device,optimizer=None,scaler=None,desc="Train"):
    training=optimizer is not None; model.train(training); loss_sum=n=0
    for b in tqdm(loader,desc=desc,unit="batch",dynamic_ncols=True):
        ur=b["ur"].to(device,non_blocking=True); coord=b["coord"].to(device,non_blocking=True); target=b["concept"].to(device,non_blocking=True); bs=ur.size(0)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda",enabled=cfg.USE_AMP): loss=concept_loss(model(ur,coord),target)
            if training: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        loss_sum+=loss.item()*bs; n+=bs
    return loss_sum/n


def save_checkpoint(model,opt,epoch,path):
    target=model.module if isinstance(model,nn.DataParallel) else model
    torch.save({"epoch":epoch,"model":target.state_dict(),"optimizer":opt.state_dict()},path)


def save_outputs(history,train_n,val_n,best_epoch,best_loss,elapsed):
    e=np.arange(1,len(history["train"])+1); plt.figure(figsize=(7,5)); plt.plot(e,history["train"],label="Train")
    if cfg.USE_VALIDATION: plt.plot(e,history["val"],label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("UR1-only Concept Loss"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(cfg.OUTPUT_DIR,"concept_loss.png"),dpi=200); plt.close()
    report={"ablation":"UR1-only","train_samples":train_n,"validation_enabled":cfg.USE_VALIDATION,"val_samples":val_n,"epochs":cfg.EPOCHS,"batch_size":cfg.BATCH_SIZE,"learning_rate":cfg.LEARNING_RATE,"weight_decay":cfg.WEIGHT_DECAY,"gpu_ids":cfg.GPU_IDS,"training_seconds":elapsed,"model":{"dim":cfg.DIM,"heads":cfg.HEADS,"ffn_dim":cfg.FFN_DIM,"dropout":cfg.DROPOUT,"blocks":"self-attn+FFN -> self-attn+FFN"},"loss":"concept MSE","best_epoch":best_epoch,"best_loss":best_loss,"best_metric":"val_loss" if cfg.USE_VALIDATION else "train_loss","final_train":history["train"][-1],"final_val":history["val"][-1] if cfg.USE_VALIDATION else None,"history":history}
    with open(os.path.join(cfg.OUTPUT_DIR,"training_report.json"),"w",encoding="utf-8") as f: json.dump(report,f,indent=2,ensure_ascii=False)


def main():
    set_seed(cfg.SEED); torch.backends.cudnn.benchmark=True; os.makedirs(cfg.OUTPUT_DIR,exist_ok=True); t0=time.time()
    train=build_dataset(cfg.TRAIN_PKL_DIR,cfg.TRAIN_UR_DIR,cfg.TRAIN_SAMPLES,cfg.SEED)
    val=build_dataset(cfg.VAL_PKL_DIR,cfg.VAL_UR_DIR,cfg.VAL_SAMPLES,cfg.SEED+1) if cfg.USE_VALIDATION else None
    train_loader=build_loader(train,True); val_loader=build_loader(val,False) if val else None
    device=torch.device(f"cuda:{cfg.GPU_IDS[0]}"); model=UR1AblationModel(cfg.DIM,cfg.HEADS,cfg.FFN_DIM,cfg.DROPOUT).to(device)
    if len(cfg.GPU_IDS)>1: model=nn.DataParallel(model,device_ids=cfg.GPU_IDS)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.LEARNING_RATE,weight_decay=cfg.WEIGHT_DECAY); scaler=torch.amp.GradScaler("cuda",enabled=cfg.USE_AMP)
    history={"train":[],"val":[]}; best_loss,best_epoch=float("inf"),0
    print("="*70); print(f"UR1-only ablation | train={len(train)} | validation={cfg.USE_VALIDATION}"); print(f"GPU={cfg.GPU_IDS} | batch={cfg.BATCH_SIZE} | epochs={cfg.EPOCHS}"); print("="*70)
    for epoch in range(1,cfg.EPOCHS+1):
        tr=run_epoch(model,train_loader,device,opt,scaler,f"Epoch {epoch}/{cfg.EPOCHS} Train"); history["train"].append(tr)
        if cfg.USE_VALIDATION:
            va=run_epoch(model,val_loader,device,desc=f"Epoch {epoch}/{cfg.EPOCHS} Val"); history["val"].append(va); metric=va; print(f"Epoch {epoch} | train={tr:.6f} | val={va:.6f}")
        else: metric=tr; print(f"Epoch {epoch} | train={tr:.6f}")
        if metric<best_loss: best_loss,best_epoch=metric,epoch; save_checkpoint(model,opt,epoch,os.path.join(cfg.OUTPUT_DIR,"best.pt"))
        if epoch%cfg.SAVE_EVERY==0: save_checkpoint(model,opt,epoch,os.path.join(cfg.OUTPUT_DIR,f"epoch_{epoch}.pt"))
    save_checkpoint(model,opt,cfg.EPOCHS,os.path.join(cfg.OUTPUT_DIR,"last.pt")); save_outputs(history,len(train),len(val) if val else 0,best_epoch,best_loss,time.time()-t0)
    print("="*70); print(f"Training finished | best epoch={best_epoch} | best loss={best_loss:.6f}"); print(f"Results: {cfg.OUTPUT_DIR}"); print("="*70)


if __name__=="__main__": main()
