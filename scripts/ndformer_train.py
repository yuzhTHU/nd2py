import re
import sys
import json
import torch
import shlex
import random
import logging
import numpy as np
import nd2py as nd
import torch.utils.data as D
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from socket import gethostname
from collections import defaultdict
from argparse import ArgumentParser
from setproctitle import setproctitle
from nd2py.search.ndformer import NDFormerModelConfig, NDFormerDataset, NDFormerTokenizer, NDFormerModel

_logger = logging.getLogger("nd2py.ndformer_train")


def load_dataset(args, config, tokenizer):
    train_dataset = NDFormerDataset(
        config=config, 
        tokenizer=tokenizer,
        eqtree_generator=nd.search.ndformer.NDFormerEqtreeGenerator(tokenizer.variables, depth_range=(1, 6)), 
        topo_generator=nd.search.ndformer.NDFormerGraphGenerator(config),
        data_generator=nd.search.ndformer.NDFormerDataGenerator(config), 
        n_samples=480,
        #random_state=args.random_state,
    )
    eval_dataset = NDFormerDataset(
        config=config, 
        tokenizer=tokenizer,
        eqtree_generator=nd.search.ndformer.NDFormerEqtreeGenerator(tokenizer.variables, depth_range=(1, 6)), 
        topo_generator=nd.search.ndformer.NDFormerGraphGenerator(config),
        data_generator=nd.search.ndformer.NDFormerDataGenerator(config), 
        n_samples=480,
        random_state=args.random_state,
    )
    test_dataset = NDFormerDataset(
        config=config, 
        tokenizer=tokenizer,
        eqtree_generator=nd.search.ndformer.NDFormerEqtreeGenerator(tokenizer.variables, depth_range=(1, 6)), 
        topo_generator=nd.search.ndformer.NDFormerGraphGenerator(config),
        data_generator=nd.search.ndformer.NDFormerDataGenerator(config), 
        n_samples=480,
        random_state=args.random_state+1,
    )
    train_loader = D.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        sampler=train_dataset.get_sampler(), 
        collate_fn=train_dataset.collate_fn,
    )
    eval_loader = D.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=eval_dataset.get_sampler(),
        collate_fn=eval_dataset.collate_fn,
    )
    test_loader = D.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=test_dataset.get_sampler(),
        collate_fn=test_dataset.collate_fn,
    )
    _logger.note(
        f"Dataset loaded with {len(train_dataset) if args.n_samples else 'infinite'} training samples, "
        f"{len(eval_dataset)} evaluation samples, and {len(test_dataset)} test samples."
    )
    return train_loader, eval_loader, test_loader


def load_model(args, config, tokenizer, train_loader):
    model = NDFormerModel.create(config, tokenizer).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=0.1
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR( # Warmup + Cosine Decay
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        cycle_momentum=False,
    )
    _logger.note(
        "Model Parameters:\n"
        f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
        f"Total: {sum(p.numel() for p in model.parameters()):,}\n"
        f"Optimizer: AdamW(lr={args.lr}, weight_decay=0.01, betas=(0.9, 0.98))\n"
        f"Scheduler: OneCycleLR(max_lr={args.lr}, pct_start=0.1, warmup epochs={int(args.epochs * 0.1)})"
    )
    return model, optimizer, criterion, scheduler


def reload_checkpoint(args, model, optimizer, scheduler):
    if args.force_new_experiment:
        checkpoint_path = None
    elif args.reload_checkpoint is not None:
        # 如果指定了 checkpoint 路径，则从该路径加载
        checkpoint_path = Path(args.reload_checkpoint)
    elif (Path(args.save_path) / "checkpoint.pth").exists():
        # 如果当前保存路径下存在 checkpoint，则从该路径加载
        checkpoint_path = Path(args.save_path) / "checkpoint.pth"
    else:
        checkpoint_path = None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        start_epoch = checkpoint["epoch"] + 1
        if 'args' in checkpoint:
            saved_args = checkpoint['args']
            for key in sorted(set(saved_args.keys()) | set(vars(args).keys())):
                if key in [
                    'device', 'save_dir', 'save_path', 'command', 'name', 'exp_name',
                    'random_state', 'reload_checkpoint', 'test_before_train', 'test_per_epoch',
                ]: continue
                val1 = saved_args.get(key, None)
                val2 = getattr(args, key, None)
                if val1 != val2:
                    _logger.warning(
                        f"Argument '{key}' differs from the saved checkpoint: "
                        f"saved_args={val1} vs. current_args={val2}"
                    )
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint: optimizer.load_state_dict(checkpoint["optimizer"])
        else: _logger.warning("Optimizer state not found in checkpoint, optimizer re-initialized.")
        if "scheduler" in checkpoint: scheduler.load_state_dict(checkpoint["scheduler"])
        else: _logger.warning("Scheduler state not found in checkpoint, scheduler re-initialized.")
        if "timer" in checkpoint: timer = nd.utils.NamedTimer.from_dict(checkpoint["timer"])
        else: timer = nd.utils.NamedTimer()
        if "best_records" in checkpoint: best_records = checkpoint["best_records"]
        else: best_records = None
        _logger.note(nd.utils.tag2ansi(f"Checkpoint loaded from [underline green]{checkpoint_path}[reset], resume from epoch [underline green]{start_epoch}[reset]."))
    else:
        start_epoch = 0
        timer = nd.utils.NamedTimer()
        best_records = None
    return start_epoch, timer, best_records


def log_train_record(args, train_records, timer):
    return nd.utils.tag2ansi(
        f"[Epoch {train_records['epoch']}/{args.epochs}] "
        f"[#66CCFF]Train Loss[reset]={train_records['loss']:.4f}, "
        f"[#66CCFF]Train Accuracy[reset]={train_records['accuracy']:.2%}, "
        f"[#66CCFF]Time Usage[reset]={timer.to_str()}"
    )

def log_eval_record(args, eval_records, timer):
    return nd.utils.tag2ansi(
        f"[Epoch {eval_records['epoch']}/{args.epochs}] "
        f"[#66CCFF]Eval Loss[reset]={eval_records['loss']:.4f}, "
        f"[#66CCFF]Eval Accuracy[reset]={eval_records['accuracy']:.2%}, "
        f"[#66CCFF]Time Usage[reset]={timer.to_str()}"
    )

def log_test_record(args, test_records, timer):
    return re.sub('Eval', 'Test', log_eval_record(args, test_records, timer))


def main(args):
    config = NDFormerModelConfig(model=args.model)

    ## Load Tokenizer
    tokenizer = NDFormerTokenizer(config, variables=None)

    ## Load Dataset
    train_loader, eval_loader, test_loader = load_dataset(args, config, tokenizer)

    ## Load Model
    model, optimizer, criterion, scheduler = load_model(args, config, tokenizer, train_loader)

    ## Reload Checkpoint
    start_epoch, timer, best_records = reload_checkpoint(args, model, optimizer, scheduler)

    ## Train
    for epoch in range(start_epoch, args.epochs+1):
        # 训练一个 epoch
        if epoch > 0:
            train_timer = nd.utils.NamedTimer()
            torch.set_grad_enabled(True)
            model.train()

            records = defaultdict(list)
            for batch_idx, batch_dict in enumerate(pbar := tqdm(train_loader, leave=False, dynamic_ncols=True)):
                for k, v in batch_dict.items():
                    if isinstance(v, torch.Tensor):
                        batch_dict[k] = v.to(args.device)
                torch.cuda.synchronize()
                train_timer.add('Prepare-Data')

                logits = model(batch_dict) # (B_seq, seq_len, n_words)
                torch.cuda.synchronize()
                train_timer.add('Forward')

                targets = batch_dict["next_tokens"] # (B_seq,)
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()  # OneCycleLR 需要在每个 batch 后更新
                torch.cuda.synchronize()
                train_timer.add('Backward')

                preds = logits.argmax(dim=-1) # (B_seq,)
                records['loss'].extend([loss.item()] * targets.size(0))
                records['correct'].extend((preds == targets).detach().cpu().tolist())
                train_timer.add('Statistics')

            train_records = {
                'epoch': epoch,
                'phase': 'train',
                'loss': sum(records['loss']) / len(records['loss']),
                'accuracy': sum(records['correct']) / len(records['correct'])
            }
            _logger.info(log_train_record(args, train_records, train_timer))
            timer.add('Train')
        else:
            train_records = None

        # 测试一个 epoch
        if (epoch > 0 and not epoch % args.test_per_epoch) or (epoch == 0 and args.test_before_train):
            eval_timer = nd.utils.NamedTimer()
            torch.set_grad_enabled(False)
            model.eval()

            records = defaultdict(list)
            for batch_idx, batch_dict in enumerate(pbar := tqdm(eval_loader, leave=False, dynamic_ncols=True)):
                for k, v in batch_dict.items():
                    if isinstance(v, torch.Tensor):
                        batch_dict[k] = v.to(args.device)
                eval_timer.add('Prepare-Data')
                logits = model(batch_dict, timer=eval_timer) # (B_seq, n_words)
                targets = batch_dict["next_tokens"] # (B_seq,)
                loss = criterion(logits, targets)
                preds = logits.argmax(dim=-1) # (B_seq,)
                records['loss'].extend([loss.item()] * targets.size(0))
                records['correct'].extend((preds == targets).detach().cpu().tolist())
                eval_timer.add('Statistics')

            eval_records = {
                'epoch': epoch,
                'phase': 'eval',
                'loss': sum(records['loss']) / len(records['loss']),
                'accuracy': sum(records['correct']) / len(records['correct'])
            }
            _logger.info(log_eval_record(args, eval_records, eval_timer))
            timer.add('Eval')
        else:
            eval_records = None

        # 保存日志
        with open(f"{args.save_path}/records.jsonl", "a") as f:
            if train_records is not None:
                f.write(json.dumps(train_records) + "\n")
            if eval_records is not None:
                f.write(json.dumps(eval_records) + "\n")
            timer.add('save_records')

        # 保存加载点
        if '_last_checkpoint_time' not in locals() or (datetime.now() - _last_checkpoint_time).seconds > 300:
            # 只在间隔超过 5 min 时保存
            _last_checkpoint_time = datetime.now()
            save_path = f"{args.save_path}/checkpoint.pth"
            torch.save({
                "epoch": epoch,
                "args": vars(args),
                "config": vars(config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "timer": timer.to_dict(),
                "best_records": best_records,
            }, save_path)
            _logger.info(nd.utils.tag2ansi(f"Checkpoint saved to [underline green]{save_path}[reset]."))
            timer.add('save_checkpoint')

        # 定期保存
        if set(str(epoch)[1:]) == {'0'}:
            # 只在 epoch=10,20,...,100,...,1000,... 时保存
            save_path = Path(args.save_path) / "checkpoints" / f"epoch{epoch}.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "args": vars(args),
                "config": vars(config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "timer": timer.to_dict(),
                "best_records": best_records,
            }, save_path)
            _logger.note(nd.utils.tag2ansi(f"Model saved to [underline green]{save_path}[reset]"))
            timer.add('save_periodly')

        # 保存最佳模型
        if eval_records is not None:
            if (
                best_records is None or
                np.mean(eval_records['loss']) < np.mean(best_records['loss'])
            ):
                best_records = eval_records
                best_records['patience'] = args.patience
                save_path = f"{args.save_path}/best.pth"
                torch.save({
                    "epoch": epoch,
                    "args": vars(args),
                    "config": vars(config),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "timer": timer.to_dict(),
                    "best_records": best_records,
                }, save_path)
                _logger.note(nd.utils.tag2ansi(f"Best model saved to [underline green]{save_path}[reset]"))
            else:
                best_records['patience'] -= 1
                _logger.info(
                    nd.utils.tag2ansi(f"No improvement in eval loss, patience decreased to [red bold]{best_records['patience']}/{args.patience}[reset]. ") +
                    nd.utils.tag2ansi("[red bold]Current Best Eval Record[reset]: ") + \
                    log_eval_record(args, best_records, timer)
                )
            timer.add('save_best')

        # 打印用时
        allocated = torch.cuda.memory_allocated(args.device) / 1024 / 1024 / 1024
        reserved = torch.cuda.memory_reserved(args.device) / 1024 / 1024 / 1024
        peak = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024 / 1024
        _logger.info(nd.utils.tag2ansi(
            f"[pink][Epoch {epoch}/{args.epochs}] finished.[reset] "
            f"Time Usage={timer.to_str(mode='time')}, "
            f"CUDA ({args.device}) usage: allocated={allocated:.1f}GiB, peak={peak:.1f}GiB, reserved={reserved:.1f}GiB"
        ))

        # 释放额外的显存
        if args.minimize_gpu and train_records is not None:
            peak = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024
            reserved_raw = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            torch.cuda.empty_cache() # 释放 reserved 但是未被 allocated 的 block
            reserved_new = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            if reserved_new < peak: # 释放了过多的显存，之后可能会 OOM
                allocated = torch.cuda.memory_allocated(args.device) / 1024 / 1024
                if (keep_MB := int(np.ceil(peak - allocated))) > 0: # 把需要的显存再占回来
                    tmp = nd.utils.AutoGPU.allocate_gpu(device=args.device, memory_MB=keep_MB, block_MB=None)
                    del tmp
                reserved_new = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            _logger.info(nd.utils.tag2ansi(
                f"[gray]Adjust reserved memory from {reserved_raw/1024:.1f}GiB to {reserved_new/1024:.1f}GiB.[reset]"
            ))
        
        # 提前终止
        if best_records is not None and best_records['patience'] <= 0:
            _logger.warning(nd.utils.tag2ansi(f"Early stopping at epoch [lightred]{epoch}/{args.epochs}[reset], "))
            break

    ## Log Best Result
    _logger.info(
        nd.utils.tag2ansi("[red]Best Eval Results[reset]: ") +
        log_eval_record(args, best_records, timer)
    )

    ## Load Best Model
    best_path = Path(args.save_path) / 'best.pth'
    checkpoint = torch.load(best_path, map_location=args.device)
    if checkpoint['epoch'] != best_records['epoch']:
        _logger.warning(nd.utils.tag2ansi(
            f"Best epoch in records.jsonl ({best_records['epoch']}) does not match that in best.pth ({checkpoint['epoch']})!"
        ))
    model.load_state_dict(checkpoint["model"])
    _logger.note(nd.utils.tag2ansi(
        f'Load best model from epoch {best_records["epoch"]} ([green]{best_path}[reset]) for final test.'
    ))

    ## Test
    test_timer = nd.utils.NamedTimer()
    torch.set_grad_enabled(False)
    model.eval()
    records = defaultdict(list)
    for batch_idx, batch_dict in enumerate(pbar := tqdm(test_loader, leave=False, dynamic_ncols=True)):
        for k, v in batch_dict.items():
            if isinstance(v, torch.Tensor):
                batch_dict[k] = v.to(args.device)
        test_timer.add('Prepare-Data')
        logits = model(batch_dict, timer=test_timer) # (B_seq, n_words)
        targets = batch_dict["next_tokens"] # (B_seq,)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=-1) # (B_seq,)
        records['loss'].extend([loss.item()] * targets.size(0))
        records['correct'].extend((preds == targets).detach().cpu().tolist())
        test_timer.add('Statistics')
    timer.add('Test')
    test_records = {
        'epoch': best_records['epoch'],
        'phase': 'test',
        'loss': sum(records['loss']) / len(records['loss']),
        'accuracy': sum(records['correct']) / len(records['correct'])
    }
    _logger.note(
        nd.utils.tag2ansi("[red]Final Test Results[reset]: ") + 
        log_test_record(args, test_records, test_timer)
    )

    _logger.note(f"Training finished. Re-run: {args.command}")


if __name__ == "__main__":
    parser = ArgumentParser()
    # 基础配置
    parser.add_argument("--name", type=str, default="train", help="实验任务名称，用于生成实验ID")
    parser.add_argument("--exp_name", type=str, default=None, help="手动指定实验名称（若指定则覆盖自动生成的名称）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备，可选 'cpu', 'cuda:0' 或 'auto'（自动选择显存充足的 GPU）")
    parser.add_argument("--random_state", type=int, default=None, help="随机种子，固定以复现实验结果")
    parser.add_argument("--save_dir", type=str, default="./logs/train", help="日志和模型权重的保存根目录")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式（输出更多日志，不保存部分文件）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 的工作线程数（0 表示主线程）")
    parser.add_argument("--minimize_gpu", action="store_true", default=False, help="是否在每个 epoch 结束后尽可能释放显存以供其他进程使用")

    # 模型配置
    parser.add_argument("--model", type=str, default="default", choices=["default", "flash_ansr"], help="模型架构类型")

    parser.add_argument('--n_samples', type=int, default=None, help="训练样本数量，默认为 None 表示无限生成样本")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=24, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率 (Learning Rate)")
    parser.add_argument("--epochs", type=int, default=10000, help="最大训练轮数")
    parser.add_argument('--patience', type=int, default=20, help="Early Stopping 的耐心值（多少个 epoch 验证集指标不提升则停止）")
    parser.add_argument('--loss_type', type=str, default='noise', choices=['position', 'accelerate', 'noise'], help="损失函数计算的目标类型")
    parser.add_argument('--reload_checkpoint', type=str, default=None, help="断点续训的 checkpoint 路径（.pth 文件）")
    parser.add_argument('--force_new_experiment', action='store_true', help="是否强制不使用 checkpoint 继续训练，即使存在 checkpoint 文件")
    parser.add_argument('--required_memory_MB', type=int, default=5000, help="自动选择 GPU 时要求的最小剩余显存 (MB)")

    parser.add_argument('--test_before_train', action='store_true', help="是否在正式训练前先进行一次测试（使用未训练的模型权重）")
    parser.add_argument('--test_per_epoch', type=int, default=1, help="每隔多少个 epoch 进行一次测试（0 表示不进行测试）")

    parser = nd.utils.add_minus_flags(parser) ## --key_name -> --key-name
    parser = nd.utils.add_negation_flags(parser) ## --action-as-true -> --no-action-as-true
    args, unknown = parser.parse_known_args()

    ## Build Save Path
    if args.exp_name is None:
        now = datetime.now()
        date = now.strftime("%Y%m%d")
        curr = now.strftime("%H%M%S")
        host = gethostname()
        exp_name = f'{date}_{args.name}_{curr}_{host}'
        exp_name = re.compile(r'[ <>:"/\\|?*\x00-\x1f]').sub('_', exp_name.strip())
        exp_name = exp_name or 'unnamed'
        exp_name = exp_name[:255] # Max filename length on most filesystems
        args.exp_name = exp_name
    save_path = Path(args.save_dir) / args.exp_name
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        _logger.warning(f"Save path {save_path} already exists.")
    args.save_path = str(save_path)

    ## Init Logger
    nd.utils.init_logger(
        "src",
        exp_name=args.exp_name,
        log_file=save_path / "info.log",
        info_level="debug" if args.debug else "info",
    )
    nd.utils.init_logger(
        "nd2py",
        exp_name=args.exp_name,
        log_file=save_path / "info.log",
        info_level="debug" if args.debug else "info",
    )

    ## Warm Unknown Args
    if unknown:
        _logger.warning(f"Unknown args: {unknown}")

    ## Set random_state
    if args.random_state is None:
        args.random_state = random.randint(1, 10000)
    nd.utils.seed_all(args.random_state)
    ## Set Command
    args.command = ' '.join(map(shlex.quote, [sys.executable, *sys.argv]))
    ## Select GPU
    if args.device == "auto":
        args.device = nd.utils.AutoGPU().choice_gpu(memory_MB=args.required_memory_MB, interval=15)

    ## Save Args
    args_path = save_path / "args.json"
    if args_path.exists():
        i = 1
        while args_path.with_suffix(f".json.{i}").exists(): i += 1
        args_path.rename(args_path.with_suffix(f".json.{i}"))
        _logger.warning(f"args.json already exists, backup to args.json.{i}")
    _logger.note(f"Args: {args}")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    ## Start Training
    setproctitle(f"{args.exp_name}@ZihanYu")
    main(args)
