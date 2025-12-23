import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from poolenv import PoolEnv
from agent import Agent, BasicAgent, CEMAgent, NewAgent, SearchAgent, MCTSAgent, RLAgent
from utils import set_random_seed


class RandomAgent(Agent):
    def decision(self, balls=None, my_targets=None, table=None):
        return self._random_action()


def _make_opponent(name):
    if name == 'random':
        return RandomAgent()
    if name == 'new':
        return NewAgent()
    if name == 'basic':
        return BasicAgent()
    if name == 'search':
        return SearchAgent()
    if name == 'mcts':
        return MCTSAgent()
    if name == 'rl':
        return RLAgent()
    if name == 'basic_fast':
        opp = BasicAgent()
        opp.INITIAL_SEARCH = 5
        opp.OPT_SEARCH = 3
        return opp
    raise ValueError(f"unknown opponent: {name}")


def _play_match(agent_a, agent_b, n_games, seed, seed_enable, silent):
    if n_games % 4 != 0:
        raise ValueError("n_games must be a multiple of 4")

    set_random_seed(enable=seed_enable, seed=seed)

    env = PoolEnv()
    results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
    players = [agent_a, agent_b]
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']

    f = io.StringIO() if silent else None
    with (contextlib.redirect_stdout(f) if silent else contextlib.nullcontext()):
        with (contextlib.redirect_stderr(f) if silent else contextlib.nullcontext()):
            for i in range(n_games):
                env.reset(target_ball=target_ball_choice[i % 4])
                while True:
                    player = env.get_curr_player()
                    obs = env.get_observation(player)
                    if player == 'A':
                        action = players[i % 2].decision(*obs)
                    else:
                        action = players[(i + 1) % 2].decision(*obs)
                    env.take_shot(action)

                    done, info = env.get_done()
                    if done:
                        if info['winner'] == 'SAME':
                            results['SAME'] += 1
                        elif info['winner'] == 'A':
                            results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
                        else:
                            results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i + 1) % 2]] += 1
                        break
    results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
    results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5
    return results


def _fitness_from_results(results, n_games):
    return float(results['AGENT_B_SCORE']) / float(n_games)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--pop-size', type=int, default=18)
    parser.add_argument('--elite-frac', type=float, default=0.3)
    parser.add_argument('--eval-games', type=int, default=8)
    parser.add_argument('--eval-repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--opponent', choices=['random', 'new', 'basic', 'basic_fast', 'search', 'mcts', 'rl'], default='basic')
    parser.add_argument('--opponents', type=str, default=None)
    parser.add_argument('--seed-disable', action='store_true')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--out', type=str, default=os.path.join('eval', 'cem_agent.json'))
    args = parser.parse_args()

    if args.eval_games % 4 != 0:
        raise ValueError("--eval-games must be a multiple of 4")
    if args.pop_size <= 0 or args.iters <= 0:
        raise ValueError("--pop-size and --iters must be positive")
    if not (0.0 < args.elite_frac <= 1.0):
        raise ValueError("--elite-frac must be in (0, 1]")
    if args.eval_repeats <= 0:
        raise ValueError("--eval-repeats must be positive")

    if args.opponents is not None:
        opponent_names = [x.strip() for x in args.opponents.split(',') if x.strip()]
        if len(opponent_names) == 0:
            raise ValueError("--opponents is empty")
        for n in opponent_names:
            _make_opponent(n)
    else:
        opponent_names = [args.opponent]

    template = CEMAgent()
    bounds = template.param_bounds
    keys = list(bounds.keys())
    lows = np.array([bounds[k][0] for k in keys], dtype=np.float64)
    highs = np.array([bounds[k][1] for k in keys], dtype=np.float64)

    mean = (lows + highs) * 0.5
    std = (highs - lows) * 0.25
    min_std = (highs - lows) * 0.03

    elite_k = max(1, int(args.pop_size * args.elite_frac))
    best_params = template.params
    best_fitness = float('-inf')

    t0 = time.time()
    for it in range(args.iters):
        candidates = mean + std * np.random.randn(args.pop_size, len(keys))
        candidates = np.clip(candidates, lows, highs)

        fitnesses = []
        for j in range(args.pop_size):
            params = {k: float(candidates[j, idx]) for idx, k in enumerate(keys)}
            agent_b = CEMAgent(params=params)
            scores = []
            for opp_idx, opp_name in enumerate(opponent_names):
                for r in range(args.eval_repeats):
                    agent_a = _make_opponent(opp_name)
                    results = _play_match(
                        agent_a=agent_a,
                        agent_b=agent_b,
                        n_games=args.eval_games,
                        seed=args.seed + it * 10000 + opp_idx * 100 + r,
                        seed_enable=not args.seed_disable,
                        silent=args.silent,
                    )
                    scores.append(_fitness_from_results(results, args.eval_games))
            fit = float(np.mean(scores))
            fitnesses.append(fit)

            if fit > best_fitness:
                best_fitness = fit
                best_params = params

        fitnesses = np.array(fitnesses, dtype=np.float64)
        elite_idx = np.argsort(fitnesses)[-elite_k:]
        elite = candidates[elite_idx]

        mean = np.mean(elite, axis=0)
        std = np.maximum(np.std(elite, axis=0), min_std)

        elapsed = time.time() - t0
        print(f"iter={it+1}/{args.iters} best={best_fitness:.3f} gen_best={float(np.max(fitnesses)):.3f} elapsed={elapsed:.1f}s")

        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            'params': best_params,
            'fitness': best_fitness,
            'iter': it + 1,
            'keys': keys,
            'bounds': bounds,
            'eval_games': args.eval_games,
            'eval_repeats': args.eval_repeats,
            'opponent': args.opponent,
            'opponents': opponent_names,
            'seed': args.seed,
        }
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"checkpoint saved: {args.out}")


if __name__ == '__main__':
    main()
