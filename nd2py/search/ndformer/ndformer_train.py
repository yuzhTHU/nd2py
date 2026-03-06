import os
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
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from socket import gethostname
from collections import defaultdict
from argparse import ArgumentParser
from setproctitle import setproctitle
from nd2py.search.ndformer import NDformerConfig, NDformerDataset, NDformerTokenizer, NDformerModel

_logger = logging.getLogger("nd2py.ndformer_train")


def load_dataset(args, config, tokenizer):
    eq_generator = nd.generator.GPLearnGenerator(tokenizer.variables)
    topo_generator = nd.search.ndformer.NDformerGraphGenerator(config)
    data_generator = nd.search.ndformer.NDformerDataGenerator(config)
    train_dataset = NDformerDataset(
        config=config, 
        tokenizer=tokenizer,
        eq_generator=eq_generator, 
        topo_generator=topo_generator,
        data_generator=data_generator, 
        n_samples=36,
    )
    eval_dataset = NDformerDataset(
        config=config, 
        tokenizer=tokenizer,
        eq_generator=eq_generator, 
        topo_generator=topo_generator,
        data_generator=data_generator, 
        n_samples=36,
    )
    test_dataset = NDformerDataset(
        config=config, 
        tokenizer=tokenizer,
        eq_generator=eq_generator, 
        topo_generator=topo_generator,
        data_generator=data_generator, 
        n_samples=36,
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


def load_model(args, config, tokenizer):
    model = NDformerModel(config, tokenizer).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    _logger.note(
        "Model Parameters:\n"
        f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
        f"Total: {sum(p.numel() for p in model.parameters()):,}"
    )
    
    return model,optimizer,criterion


def reload_checkpoint(args, model, optimizer):
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
                    'seed', 'reload_checkpoint', 'test_before_train', 'test_per_epoch',
                ]: continue
                val1 = saved_args.get(key, None)
                val2 = getattr(args, key, None)
                if val1 != val2:
                    _logger.warning(
                        f"Argument '{key}' differs from the saved checkpoint: "
                        f"saved_args={val1} vs. current_args={val2}"
                    )
        model.load_state_dict(checkpoint["model"])
        _logger.note(nd.utils.tag2ansi(f"Checkpoint loaded from [underline green]{checkpoint_path}[reset], resume from epoch [underline green]{start_epoch}[reset]."))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            _logger.warning("Optimizer state not found in checkpoint, optimizer re-initialized.")
    else:
        start_epoch = 0
    return start_epoch


def log_train_record(args, train_records, timer):
    return (
        f"[Epoch {train_records['epoch']}/{args.epochs}] "
        f"Train Loss={train_records['loss']:.4f}, "
        f"Train Accuracy={train_records['accuracy']:.2%}, "
        f"Time Usage={timer.to_str()}"
    )


def log_test_record(args, test_records, timer):
    return (
        f"[Epoch {test_records['epoch']}/{args.epochs}] "
        f"Eval Loss={test_records['loss']:.4f}, "
        f"Eval Accuracy={test_records['accuracy']:.2%}, "
        f"Time Usage={timer.to_str()}"
    )


def main(args):
    config = NDformerConfig()

    ## Load Tokenizer
    tokenizer = NDformerTokenizer(config, variables=None)

    ## Load Dataset
    train_loader, eval_loader, test_loader = load_dataset(args, config, tokenizer)

    ## Load Model
    model, optimizer, criterion = load_model(args, config, tokenizer)

    ## Reload Checkpoint
    start_epoch = reload_checkpoint(args, model, optimizer)

    ## Train
    timer = nd.utils.NamedTimer()
    for epoch in range(start_epoch, args.epochs+1):
        # 训练一个 epoch
        if epoch > 0:
            torch.set_grad_enabled(True)
            model.train()

            records = defaultdict(list)
            for batch_idx, batch_dict in enumerate(pbar := tqdm(train_loader, leave=False, dynamic_ncols=True)):
                # 将数据迁移到 GPU
                for k, v in batch_dict.items():
                    if isinstance(v, torch.Tensor):
                        batch_dict[k] = v.to(args.device)
                torch.cuda.synchronize()
                timer.add('Generate-Data')

                optimizer.zero_grad()
                logits = model(batch_dict) # (B_seq, seq_len, n_words)
                torch.cuda.synchronize()
                timer.add('Drop1')
                
                # ⚠️ 取序列最后一个位置的预测结果
                targets = batch_dict["next_tokens"] # (B_seq,)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 建议加上梯度裁剪防爆炸
                optimizer.step()
                torch.cuda.synchronize()
                timer.add('Backward')
                
                # 统计
                preds = logits.argmax(dim=-1) # (B_seq,)
                records['loss'].extend([loss.item()] * targets.size(0))
                records['correct'].extend((preds == targets).detach().cpu().tolist())
                timer.add('Statistics')

            train_records = {
                'epoch': epoch,
                'phase': 'train',
                'loss': sum(records['loss']) / len(records['loss']),
                'accuracy': sum(records['correct']) / len(records['correct'])
            }
            _logger.info(log_train_record(args, train_records, timer))
            timer.add('train')
        else:
            train_records = None

        # 测试一个 epoch
        if (epoch > 0 and not epoch % args.test_per_epoch) or (epoch == 0 and args.test_before_train):
            torch.set_grad_enabled(False)
            model.eval()

            records = defaultdict(list)
            for batch_idx, batch_dict in enumerate(pbar := tqdm(eval_loader, leave=False, dynamic_ncols=True)):
                for k, v in batch_dict.items():
                    if isinstance(v, torch.Tensor):
                        batch_dict[k] = v.to(args.device)
                logits = model(batch_dict) # (B_seq, n_words)
                targets = batch_dict["next_tokens"] # (B_seq,)
                loss = criterion(logits, targets)
                preds = logits.argmax(dim=-1) # (B_seq,)
                records['loss'].extend([loss.item()] * targets.size(0))
                records['correct'].extend((preds == targets).detach().cpu().tolist())

            test_records = {
                'epoch': epoch,
                'phase': 'eval',
                'loss': sum(records['loss']) / len(records['loss']),
                'accuracy': sum(records['correct']) / len(records['correct'])
            }
            timer.add('eval')

        else:
            test_records = None

        # 保存日志
        with open(f"{args.save_path}/records.jsonl", "a") as f:
            if train_records is not None:
                f.write(json.dumps(train_records) + "\n")
            if test_records is not None:
                f.write(json.dumps(test_records) + "\n")

        # 保存加载点
        if '_last_checkpoint_time' not in locals() or (datetime.now() - _last_checkpoint_time).seconds > 300:
            # 只在间隔超过 5 min 时保存
            _last_checkpoint_time = datetime.now()
            save_path = f"{args.save_path}/checkpoint.pth"
            torch.save({
                "epoch": epoch,
                "args": vars(args),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
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
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_path)
            _logger.note(nd.utils.tag2ansi(f"Model saved to [underline green]{save_path}[reset]"))
            timer.add('save_periodly')

        # 保存最佳模型
        if test_records is not None:
            if (
                'best_records' not in locals() or 
                np.mean(test_records['loss']) < np.mean(best_records['loss'])
            ):
                patience = args.patience
                best_records = test_records
                save_path = f"{args.save_path}/best.pth"
                torch.save({
                    "epoch": epoch,
                    "args": vars(args),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, save_path)
                _logger.note(nd.utils.tag2ansi(f"Best model saved to [underline green]{save_path}[reset]"))
            else:
                patience -= 1
                # _logger.info(nd.utils.tag2ansi(
                #     f"Patience left: [brightred]{patience}/{args.patience}[reset] ("
                #     f"[bold underline orange]best Accuracy={best_records['accuracy']:.2%}[reset] "
                #     f"at epoch [#66CCFF]{best_records['epoch']}[reset]. "
                #     f"[#66CCFF]ADE={np.mean(best_records['ade']):.4f}, "
                #     f"[#66CCFF]FDE={np.mean(best_records['fde']):.4f}, "
                #     f"[#66CCFF]X_ERROR (normal)={np.nanmean(best_records['norm_err']):.4f}, "
                #     f"[#66CCFF]Y_ERROR (tangential)={np.nanmean(best_records['tan_err']):.4f}, "
                #     f"[#66CCFF]Collision-Ped={np.mean(best_records['collision_ped']) - (base := np.mean(best_records['collision_ped_base'])):.2%} (+{base:.2%}), "
                #     f"[#66CCFF]Collision-Veh={np.mean(best_records['collision_veh']) - (base := np.mean(best_records['collision_veh_base'])):.2%} (+{base:.2%}), "
                #     f"[#66CCFF]Collision-Map={np.mean(best_records['collision_map']) - (base := np.mean(best_records['collision_map_base'])):.2%} (+{base:.2%}), "
                #     f"[#66CCFF]Collision-Ped2={np.mean(best_records['collision_ped2']):.2%}, "
                #     f"[#66CCFF]Collision-Veh2={np.mean(best_records['collision_veh2']):.2%}, "
                #     f"[#66CCFF]Collision-Map2={np.mean(best_records['collision_map2']):.2%}, "
                #     f"[#66CCFF]AvgLen={np.mean(best_records['trajlen']):.4f}, "
                #     f"[#66CCFF]Loss={np.mean(best_records['loss']):.4f}, "
                #     f"[#66CCFF]PedNum={np.mean(best_records['ped_num']):.1f}, "
                #     f"[#66CCFF]VehNum={np.mean(best_records['veh_num']):.1f}), "
                #     f"[#66CCFF]RolloutTime={np.mean(best_records['rollout_time'])*1000:.2f}ms "
                #     f"([bold underline orange]FPS={1/np.mean(best_records['rollout_time']):.2f} Hz[reset])"
                # ))
            timer.add('save_best')

        # 打印用时
        allocated = torch.cuda.memory_allocated(args.device) / 1024 / 1024 / 1024
        reserved = torch.cuda.memory_reserved(args.device) / 1024 / 1024 / 1024
        peak = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024 / 1024
        _logger.info(nd.utils.tag2ansi(
            f"[pink][Epoch {epoch}/{args.epochs}] finished. "
            f"Time Usage={timer}, "
            f"CUDA ({args.device}) usage: allocated={allocated:.1f}GiB, peak={peak:.1f}GiB, reserved={reserved:.1f}GiB"
            # f"adjust reserved memory from {reserved_raw/1024:.1f}GiB to {reserved_new/1024:.1f}GiB"
            "[reset]"
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
                f"[brown]Adjust reserved memory from {reserved_raw/1024:.1f}GiB to {reserved_new/1024:.1f}GiB. [reset]"
            ))
        
        # 提前终止
        if 'patience' in locals() and patience <= 0:
            _logger.warning(nd.utils.tag2ansi(f"Early stopping at epoch [lightred]{epoch}/{args.epochs}[reset], "))
            break

    ## Log Best Result
    _logger.note(...)
    
    ## Load Best Model
    best_path = Path(args.save_path) / 'best.pth'
    checkpoint = torch.load(best_path, map_location=args.device)
    if checkpoint['epoch'] != best_records['epoch']:
        _logger.warning(nd.utils.tag2ansi(
            f"Best epoch in records.jsonl ({best_records['epoch']}) does not match that in best.pth ({checkpoint['epoch']})!"
        ))
    model.load_state_dict(checkpoint["model"])
    _logger.note(f'Load best model from epoch {best_records["epoch"]} ({best_path}) for final test.')

    ## Test
    torch.set_grad_enabled(False)
    model.eval()
    test_records = ...
    with open(f"{args.save_path}/records.jsonl", "a") as f:
        if test_records is not None:
            f.write(json.dumps(test_records) + "\n")

    ## Log Test Result
    w = np.array(test_records['sample_nums'], dtype=float)
    w /= w.sum()
    test_records['accuracy'] = 1 - np.sum(w * test_records['ade']) / np.sum(w * test_records['trajlen'])
    test_records['unweighted_accuracy'] = 1 - np.mean(test_records['ade']) / np.mean(test_records['trajlen'])
    _logger.note(...)

    _logger.note(f"Training finished. Re-run: {args.command}")


if __name__ == "__main__":
    parser = ArgumentParser()
    # 基础配置
    parser.add_argument("--name", type=str, default="train", help="实验任务名称，用于生成实验ID")
    parser.add_argument("--exp_name", type=str, default=None, help="手动指定实验名称（若指定则覆盖自动生成的名称）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备，可选 'cpu', 'cuda:0' 或 'auto'（自动选择显存充足的 GPU）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，固定以复现实验结果")
    parser.add_argument("--save_dir", type=str, default="./logs/train", help="日志和模型权重的保存根目录")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式（输出更多日志，不保存部分文件）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 的工作线程数（0 表示主线程）")
    parser.add_argument("--minimize_gpu", action="store_true", default=False, help="是否在每个 epoch 结束后尽可能释放显存以供其他进程使用")

    parser.add_argument('--n_samples', type=int, default=None, help="训练样本数量，默认为 None 表示无限生成样本")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=8, help="训练批次大小")
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

    ## Warm Unknown Args
    if unknown:
        _logger.warning(f"Unknown args: {unknown}")

    ## Set Seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    nd.utils.seed_all(args.seed)
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
