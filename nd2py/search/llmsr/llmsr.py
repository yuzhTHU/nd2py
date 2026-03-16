# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import re
import os
import ast
import time
import yaml
import json
import dotenv
import inspect
import logging
import requests
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from numpy.random import default_rng
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Dict, List, Tuple, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from ...utils import NamedTimer, ParallelTimer, render_markdown, render_python, tag2ansi
from .api import LLMAPI

dotenv.load_dotenv()

__all__ = ["LLMSR", "render_markdown", "render_python"]

_logger = logging.getLogger(__name__)


def _softmax(logprob: np.ndarray, tau: float = 1.0) -> np.ndarray:
    prob = np.exp((logprob - np.max(logprob)) / tau)
    prob /= np.sum(prob)
    return prob


class Individual:
    def __init__(self, program: str):
        self.program = program
        self.score = None

    def __rept__(self):
        return self.program

    def copy(self) -> "Individual":
        copy = Individual(self.program)
        copy.score = self.score
        return copy


class LLMSR(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        prompt: str,
        eval_program: callable,
        seed_program: callable,
        template: str = "{prompt}\n\n{eval_program}\n\n{seed_programs}",
        namespace: Dict[str, object] = {},
        n_islands: int = 10,
        n_iter: int = 1000,
        programs_per_prompt: int = 2,
        temperature_init: float = 0.1,
        temperature_period: int = 30_000,
        random_state: Optional[int] = None,
        log_per_iter: int = 1,
        log_per_sec: float = None,
        save_path: str = None,
        llm_provider: str = "SiliconFlow",
        llm_model: str = "Qwen3-8B",
    ):
        self.prompt = prompt
        self.eval_program = textwrap.dedent(inspect.getsource(eval_program))
        self.eval_name = eval_program.__name__
        self.seed_program = textwrap.dedent(inspect.getsource(seed_program))
        self.seed_name = seed_program.__name__
        self.template = template
        self.namespace = namespace
        self.n_islands = n_islands
        self.n_iter = n_iter
        self.random_state = random_state
        self.programs_per_prompt = programs_per_prompt
        self.temperature_init = temperature_init
        self.temperature_period = temperature_period
        self.log_per_iter = log_per_iter
        self.log_per_sec = log_per_sec
        self.save_path = save_path
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.api = LLMAPI.load(llm_provider, llm_model, self.programs_per_prompt)

        self.islands = []
        self.records = []
        self.best_model = None
        self._rng = default_rng(random_state)
        self.speed_timer = NamedTimer(unit="iter")
        self.token_counter = ParallelTimer(unit="token")
        self.price_counter = ParallelTimer(unit="$")

    def fit(self, data: np.ndarray | pd.DataFrame | Dict[str, np.ndarray]):
        """
        Args:
            data: (n_samples, n_dims)
        """
        if isinstance(data, np.ndarray):
            raise NotImplementedError(
                "data should be a DataFrame or a dict, not a numpy array. "
            )
        elif isinstance(data, pd.DataFrame):
            data = {col: data[col].values for i, col in enumerate(data.columns)}
        elif isinstance(data, dict):
            data = {k: np.asarray(v) for k, v in data.items()}
        else:
            raise ValueError(f"Unknown type: {type(data)}")

        prompt = self.generate_prompt([Individual(self.seed_program)])
        _logger.note(f"Initial Prompt:\n{render_markdown(prompt)}")
        if self.save_path:
            with open(Path(self.save_path) / "prompt_demo.md", "w", encoding="utf-8") as f:
                f.write(prompt)

        self.islands = self.init_islands(data)
        self.start_time = time.time()
        self.speed_timer.clear(reset_last_add_time=True)
        self.token_counter.clear(reset_last_add_time=True)
        self.price_counter.clear(reset_last_add_time=True)
        for n_iter in range(1, 1 + self.n_iter):
            # Evolve for an Iteration
            self.islands = self.evolve(self.islands, data, n_iter=n_iter)
            
            # Update best
            best = max(sum(self.islands, []), key=lambda x: x.score).copy()
            if best is not None and (self.best_model is None or best.score > self.best_model.score):
                self.best_model = best
                _logger.note(tag2ansi(f"Update Best Result with score [#66CCFF bold]{best.score}[reset]:\n{render_python(best.program)}"))
                if self.save_path:
                    with open(Path(self.save_path) / f"best_model_{n_iter}.py", "w") as f:
                        f.write(f"# Score={best.score}\n")
                        f.write(best.program)
            
            # Save Record & Print Log
            record = dict(
                iter=n_iter,
                time=self.speed_timer.time,
                score=best.score,
                length=len(best.program),
                program=best.program,
                population_size=len(sum(self.islands, [])),
            )
            if (
                (self.log_per_iter is not None and n_iter % self.log_per_iter == 0)
                or (self.log_per_sec is not None and (time.time() - getattr(self, '_last_log_time', 0)) > self.log_per_sec)
            ):
                log = {
                    'Iter': record["iter"],
                    'Score': f"{record['score']:.6f}",
                    'Program Length': record["length"],
                    'Population Size': record["population_size"],
                    'Time Usage': self.speed_timer.to_str(mode='time', mode_of_detail='pace', mode_of_percent='by_time'),
                    'Token Usage': self.token_counter.to_str(mode='count', mode_of_detail='speed', mode_of_percent='by_count'),
                    'Money Usage': self.price_counter.to_str(mode='count', mode_of_detail='speed', mode_of_percent='by_count'),
                    'Best Program': f"\n{render_python(record['program'])}\n"
                }  
                _logger.info(tag2ansi(" | ".join(f"[#66CCFF bold]{k}[reset]: {v}" for k, v in log.items())))
                self._last_log_time = time.time()

            record = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in record.items()
            }
            self.records.append(record)
            if self.save_path:
                with open(Path(self.save_path) / "records.jsonl", "a") as f:
                    f.write(json.dumps(record) + "\n")
            if best.score > -1e-6:
                _logger.info(tag2ansi(f"Early stopping at iter [#66CCFF bold]{n_iter}/{self.n_iter}[reset] with score [#66CCFF bold]{best.score}[reset]"))
                break
        _logger.note(tag2ansi(
            f"Finished Searching after [#66CCFF bold]{n_iter}/{self.n_iter}[reset] iterations. "
            f"Final Result with score [#66CCFF bold]{best.score}[reset]:\n{render_python(best.program)}"
        ))
        return self

    def init_islands(self, data) -> List[List[Individual]]:
        islands = []
        indiv = Individual(self.seed_program)
        self.set_score(indiv, data)
        for _ in range(self.n_islands):
            island = [indiv.copy()]
            islands.append(island)
        return islands

    def evolve(
        self, islands: List[List[Individual]], data, n_iter=None
    ) -> List[List[Individual]]:
        self.speed_timer.add('Other Stuff')
        
        island_idx, individuals = self.tournament(islands, data, n_iter=n_iter)
        self.speed_timer.add('Tournament')
        
        prompt = self.generate_prompt(individuals)
        self.speed_timer.add('Generate Prompt')
        
        children = self.generate_children(prompt)
        self.speed_timer.add('Generate Children')
        
        self.set_score(children, data)
        self.speed_timer.add('Set Score')
        
        islands[island_idx].extend(children)
        return islands

    def tournament(
        self, islands: List[List[Individual]], data, n_iter=None
    ) -> Tuple[int, List[Individual]]:
        # 随机采样 island
        island_idx = self._rng.choice(len(islands))
        island = islands[island_idx]

        # 将 island 中的个体按照 score 分为若干 cluster
        clusters = defaultdict(list)
        for indiv in island:
            clusters[indiv.score].append(indiv)
        scores = np.array(list(clusters.keys()))
        clusters = list(clusters.values())

        # 按照 Boltzmann selection procedure 对 cluster 进行采样，score 大者优先
        if self.programs_per_prompt < len(clusters):
            tau_c = self.temperature_init * (
                1 - (n_iter % self.temperature_period) / self.temperature_period
            )
            prob = _softmax(scores, tau=tau_c)
            if len(np.nonzero(prob)[0]) < self.programs_per_prompt:
                prob = (prob + 0.1) / (prob + 0.1).sum()
            clusters_idx = self._rng.choice(
                len(clusters), size=self.programs_per_prompt, p=prob, replace=False
            )
            clusters = [clusters[i] for i in clusters_idx]

        # 对 cluster 中个体进行采样，length 小者优先
        individuals = []
        for cluster in clusters:
            if len(cluster) > 1:
                lengths = np.array([len(indiv.program) for indiv in cluster])
                lengths = (lengths - lengths.min()) / (lengths.max() + 1e-6)
                prob = _softmax(lengths, tau=1.0)
                indiv = self._rng.choice(cluster, p=prob)
            else:
                indiv = cluster[0]
            individuals.append(indiv)
        return island_idx, individuals

    def generate_prompt(self, individuals: List[Individual]) -> str:
        materials = {}
        materials["prompt"] = self.prompt
        materials["eval_program"] = f"```python\n{self.eval_program}\n```"
        materials["seed_programs"] = ""
        for i, indiv in enumerate(sorted(individuals, key=lambda x: x.score)):
            lines = indiv.program.split("\n")
            lines[0] = lines[0].replace(self.seed_name, f"{self.seed_name}_v{i}")
            if i > 0:
                lines.insert(0, f"# Improved version of {self.seed_name}_v{i-1}:")
            elif len(individuals) > 1:
                lines.insert(0, f"# A suboptimal but still good version")
            else:
                lines.insert(0, f"# A good version")
            seed_program = "\n".join(lines)
            materials["seed_programs"] += f"```python\n{seed_program}\n```\n\n"
        return self.template.format(**materials)

    def generate_children(self, prompt: str) -> List[Individual]:
        children = []
        for program in (gen := self.api(prompt)):
            if (clear_program := self.clear(program)) is not None:
                children.append(Individual(clear_program))
        self.token_counter.add('total', gen.usage['token'].get('total', 0))
        self.token_counter.add('prompt', gen.usage['token'].get('prompt', 0))
        self.token_counter.add('answer', gen.usage['token'].get('answer', 0))
        self.token_counter.add('reason', gen.usage['token'].get('reason', 0))
        self.price_counter.add('total', gen.usage['price'].get('total', 0))
        self.price_counter.add('prompt', gen.usage['price'].get('prompt', 0))
        self.price_counter.add('answer', gen.usage['price'].get('answer', 0))
        self.price_counter.add('reason', gen.usage['price'].get('reason', 0))
        return children

    def clear(self, program: str) -> Optional[str]:
        lines = program.split("\n")
        # Find the first line of the function definition
        for start_idx, line in enumerate(lines):
            if line[:3] == "def":
                break
        else:
            _logger.warning(
                "No function definition found in the program:\n%s",
                render_markdown(program),
            )
            return None
        # Replace the first line with the seed program's first line
        lines[start_idx] = self.seed_program.split("\n", 1)[0]
        # Find the end of the function definition
        end_idx = len(lines)
        error_linenos = set()
        while end_idx - start_idx > 1:
            try:
                ast.parse("\n".join(lines[start_idx:end_idx]))
                break
            except SyntaxError as e:
                if e.lineno is None:
                    _logger.warning(
                        "Syntax error in the program:\n%s", render_markdown(program)
                    )
                    return None
                error_linenos.add(e.lineno)
                end_idx = start_idx + e.lineno - 1  # e.lineno 是从 1 开始的
        else:
            _logger.warning(
                "No valid function definition found in the program:\n%s",
                render_markdown(program),
            )
            return None
        # Ensure the function definition has a return statement
        if "return" not in "\n".join(lines[start_idx:end_idx]):
            _logger.warning(
                "No return statement found in the function definition:\n%s",
                render_markdown(program),
            )
            return None
        return "\n".join(lines[start_idx:end_idx])

    def run_evaluate(self, program: str, **kwargs):
        try:
            namespace = {}
            namespace.update(self.namespace)
            exec(program, namespace)
            exec(self.eval_program, namespace)
            score = namespace[self.eval_name](**kwargs)
        except ValueError as e:
            match = re.search(r"too many values to unpack \(expected (\d+)\)", str(e))
            if not match:
                raise e

            namespace = {}
            namespace.update(self.namespace)
            exec(program, namespace)
            exec(self.eval_program, namespace)
            score = namespace[self.eval_name](**kwargs, maxn_params=int(match.group(1)))
        return score

    # def set_score(self, individuals: Individual | List[Individual], data):
    #     if isinstance(individuals, Individual):
    #         individuals = [individuals]
    #     self.speed_timer.add("drop")
    #     for indiv in individuals:
    #         with ThreadPoolExecutor(max_workers=1) as executor:
    #             future = executor.submit(self.run_evaluate, indiv.program, **data)
    #             try:
    #                 indiv.score = future.result(timeout=120)  # 120秒超时
    #                 _logger.debug(
    #                     "Evaluated individual (score=%s)\n%s",
    #                     indiv.score,
    #                     render_python(indiv.program),
    #                 )
    #             except TimeoutError as e:
    #                 _logger.warning(
    #                     "Timeout when evaluating individual:\n%s",
    #                     render_python(indiv.program),
    #                 )
    #                 indiv.score = -np.inf
    #             except Exception as e:
    #                 _logger.warning(
    #                     "Error evaluating individual: %s\n%s",
    #                     e,
    #                     render_python(indiv.program),
    #                 )
    #                 indiv.score = -np.inf
    #     self.speed_timer.add("set_score")
    #     return individuals

    def set_score(self, individuals: Individual | List[Individual], data):
        if isinstance(individuals, Individual):
            individuals = [individuals]
        self.speed_timer.add("drop")
        for indiv in individuals:
            try:
                indiv.score = self.run_evaluate(indiv.program, **data)
                _logger.debug(
                    f"Evaluated individual (score={indiv.score})\n{render_python(indiv.program)}"
                )
            except Exception as e:
                indiv.score = -np.inf
                _logger.warning(
                    f"Error evaluating individual: {e}\n{render_python(indiv.program)}"
                )
        self.speed_timer.add("set_score")
        return individuals
