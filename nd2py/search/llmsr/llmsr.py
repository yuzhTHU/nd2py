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
from collections import defaultdict
from numpy.random import default_rng
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Dict, List, Tuple, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from ...utils import NamedTimer

dotenv.load_dotenv()

__all__ = ["LLMSR", "render_markdown", "render_python"]

_logger = logging.getLogger(__name__)


def _softmax(logprob: np.ndarray, tau: float = 1.0) -> np.ndarray:
    prob = np.exp((logprob - np.max(logprob)) / tau)
    prob /= np.sum(prob)
    return prob


def render_markdown(text: str, width=120, theme="staroffice") -> str:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax

    console = Console(record=True, width=width)
    with console.capture() as capture:
        matches = list(re.finditer(r"```(\w+)?\n(.*?)\n```", text, re.DOTALL))
        current = 0
        for match in matches:
            lang, code = match.groups()
            start, end = match.span()
            md = Markdown(text[current:start])
            code = Syntax(code, lang or "text", word_wrap=True, line_numbers=True, theme=theme)
            console.print(md)
            console.print(code)
            current = end
        if current < len(text):
            console.print(Markdown(text[current:]))
    return capture.get()


def render_python(text: str, width=120, highlight_lines=[], theme="staroffice") -> str:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax

    console = Console(record=True, width=width)
    with console.capture() as capture:
        code = Syntax(
            text,
            "python",
            theme=theme,
            word_wrap=True,
            line_numbers=True,
            highlight_lines=highlight_lines,
        )
        console.print(code)
    return capture.get()


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
        log_per_iter: int = float("inf"),
        log_per_sec: float = float("inf"),
        log_detailed_speed=False,
        save_path: str = None,
        model: str | Literal["manual"] = "Qwen3-8B",
        **kwargs,
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
        self.log_detailed_speed = log_detailed_speed
        self.save_path = save_path
        self.model = model

        self.islands = []
        self.records = []
        self.best_model = None
        self._rng = default_rng(random_state)
        self.speed_timer = NamedTimer(unit="iter", mode="pace")
        self.token_timer = NamedTimer(unit="token", mode="speed")
        if kwargs:
            _logger.warning(
                "Unknown args: %s", ", ".join(f"{k}={v}" for k, v in kwargs.items())
            )

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
        render_prompt = render_markdown(prompt)
        _logger.note("Demo Prompt:\n%s", render_prompt)
        # lines = render_prompt.splitlines()
        # for i in range(0, len(lines), 10):
        #     print("\n".join(lines[i : i + 10]))
        with open(f"./{self.save_path}/demo_prompt.md", "w", encoding="utf-8") as f:
            f.write(prompt)

        self.islands = self.init_islands(data)
        self.start_time = time.time()
        self.speed_timer.clear(reset=True)
        self.token_timer.clear(reset=True)
        for n_iter in range(1, 1 + self.n_iter):
            self.islands = self.evolve(self.islands, data, n_iter=n_iter)

            best = max(sum(self.islands, []), key=lambda x: x.score).copy()
            if self.save_path and best.program != self.best_model:
                with open(f"{self.save_path}/best_model_{n_iter}.py", "w") as f:
                    f.write(f"# Score={best.score}\n")
                    f.write(best.program)
            self.best_model = best.program

            record = dict(
                iter=n_iter,
                time=time.time() - self.start_time,
                score=best.score,
                length=len(best.program),
                program=best.program,
                population_size=len(sum(self.islands, [])),
            )
            if (
                not n_iter % self.log_per_iter
                or self.speed_timer.time > self.log_per_sec
            ):
                log = {}
                log["Iter"] = record["iter"]
                log["Score"] = f"{record['score']:.6f}"
                log["Length"] = record["length"]
                log["Population Size"] = record["population_size"]
                log["Best Model"] = f"\n{render_python(record['program'])}\n"
                if self.log_detailed_speed:
                    log["Speed"] = str(self.speed_timer)
                    log["Token Usage"] = str(self.token_timer)
                _logger.info(
                    " | ".join(f"\033[4m{k}\033[0m: {v}" for k, v in log.items())
                )
                self.speed_timer.clear()
                # self.token_timer.clear()  # Comment this for more accurate token usage estimation

            record = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in record.items()
            }
            self.records.append(record)
            if self.save_path:
                with open(f"{self.save_path}/records.jsonl", "a") as f:
                    f.write(json.dumps(record) + "\n")
            if best.score > -1e-6:
                _logger.info(
                    f"Early stopping at iter {iter} with score {best.score}:\n{render_python(best.program)}"
                )
                break
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
        island_idx, individuals = self.tournament(islands, data, n_iter=n_iter)
        prompt = self.generate_prompt(individuals)
        children = self.generate_children(prompt)
        self.set_score(children, data)
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
        self.speed_timer.add("drop")
        if self.model == "manual":
            programs = self._manual(prompt)
        else:
            programs = self._silicon_flow(prompt)
        self.speed_timer.add("request_llm")

        children = []
        for program in programs:
            clear_program = self.clear(program)
            if clear_program:
                children.append(Individual(clear_program))
        return children

    def _manual(self, prompt: str) -> str:
        import pyperclip
        import tiktoken

        # Copy the prompt to clipboard and save it to a file for manual input
        pyperclip.copy(prompt)
        prompt_path = f"{self.save_path}/manual_prompt.md"
        with open(prompt_path, "w") as f:
            f.write(prompt)
        _logger.note(
            f"Prompt copied to clipboard (and saved to {prompt_path}). Please paste it into the LLM interface and return the generated program."
        )
        programs = []
        for idx in range(1, self.programs_per_prompt + 1):
            # Wait for user input or clipboard content
            program = input(
                f"Please enter the generated program {idx}/{self.programs_per_prompt} (press Enter to use clipboard): "
            )
            if not program.strip():
                program = pyperclip.paste()
            if program is not None:
                programs.append(program)
            # Calculate token usage
            encoding = tiktoken.encoding_for_model("gpt-3.5")
            prompt_tokens = len(encoding.encode(prompt))
            answer_tokens = len(encoding.encode(program))
            total_tokens = prompt_tokens + answer_tokens
            self.token_timer.add("total", total_tokens, update_time=False)
            self.token_timer.add("prompt", prompt_tokens, update_time=False)
            self.token_timer.add("answer", answer_tokens)
        return programs

    def _silicon_flow(self, prompt: str) -> str:
        # Load the LLM configuration from the environment variable ${LLM_CONFIG_PATH} or default path
        path = os.environ.get("LLM_CONFIG_PATH", None)
        path = path or os.path.join(os.path.dirname(__file__), "llm_config.yaml")
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        if self.model not in config:
            raise ValueError(
                f"Model {self.model} not found in config file: {path} (available models: {', '.join(config.keys())})"
            )
        cfg = config[self.model]
        # Send the request to the LLM
        url = cfg["url"]
        payload = cfg["payload"] | {
            "messages": [{"role": "user", "content": prompt}],
            "n": self.programs_per_prompt,
        }
        headers = {k: os.path.expandvars(v) for k, v in cfg["headers"].items()}
        response = requests.request("POST", url, json=payload, headers=headers)
        if response.status_code != 200:
            _logger.error(
                "Error requesting LLM: %s\nResponse: %s",
                result.get("error", "Unknown error"),
                response.text,
            )
            return []
        result = response.json()
        usage = result["usage"]
        total_tokens = usage["total_tokens"]
        prompt_tokens = usage["prompt_tokens"]
        answer_tokens = usage["completion_tokens"]
        reason_tokens = usage["completion_tokens_details"].get("reasoning_tokens", 0)
        self.token_timer.add("total", total_tokens, update_time=False)
        self.token_timer.add("prompt", prompt_tokens, update_time=False)
        self.token_timer.add("answer", answer_tokens, update_time=False)
        self.token_timer.add("reason", reason_tokens)
        programs = [choice["message"]["content"] for choice in result["choices"]]
        return programs

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
