import argparse
import contextlib
import io
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from poolenv import PoolEnv
from agent import Agent, BasicAgent, CEMAgent, NewAgent, MCTSAgent, RLAgent, SearchAgent
from utils import set_random_seed


class RandomAgent(Agent):
    def decision(self, balls=None, my_targets=None, table=None):
        return self._random_action()


def _load_params(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'params' in data and isinstance(data['params'], dict):
        return data['params']
    if isinstance(data, dict):
        return data
    raise ValueError('invalid checkpoint format')


def _make_opponent(name):
    if name == 'random':
        return RandomAgent()
    if name == 'new':
        return NewAgent()
    if name == 'basic':
        return BasicAgent()
    if name == 'basic_fast':
        opp = BasicAgent()
        opp.INITIAL_SEARCH = 5
        opp.OPT_SEARCH = 3
        return opp
    if name == 'mcts':
        return MCTSAgent()
    if name == 'rl':
        return RLAgent()
    if name == 'search':
        return SearchAgent()
    raise ValueError(f"unknown opponent: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=['cem', 'mcts', 'search', 'rl', 'basic'], default='cem', help="Choose the main agent to evaluate")
    parser.add_argument('--checkpoint', type=str, default='eval/cem_agent.json', help="Path to checkpoint (only for cem agent)")
    parser.add_argument('--n-games', type=int, default=120)
    parser.add_argument('--seed-enable', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--opponent', choices=['basic', 'basic_fast', 'new', 'random', 'mcts', 'rl', 'search'], default='basic')
    parser.add_argument('--silent', action='store_true')
    args = parser.parse_args()

    if args.n_games % 4 != 0:
        raise ValueError('--n-games must be a multiple of 4')

    params = None
    if args.agent == 'cem':
        try:
            params = _load_params(args.checkpoint)
            print(f"Loaded CEM params from {args.checkpoint}")
        except FileNotFoundError:
            print(f"Warning: Checkpoint {args.checkpoint} not found. Using default CEM params.")
            params = None

    def _make_agent(name, params):
        if name == 'cem':
            return CEMAgent(params=params)
        if name == 'mcts':
            return MCTSAgent()
        if name == 'rl':
            return RLAgent()
        if name == 'search':
            return SearchAgent()
        if name == 'basic':
            return BasicAgent()
        raise ValueError(f"Unknown agent: {name}")

    agent_b = _make_agent(args.agent, params)
    agent_a = _make_opponent(args.opponent)

    set_random_seed(enable=args.seed_enable, seed=args.seed)
    env = PoolEnv()
    results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
    players = [agent_a, agent_b]
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']

    f = io.StringIO() if args.silent else None
    with (contextlib.redirect_stdout(f) if args.silent else contextlib.nullcontext()):
        with (contextlib.redirect_stderr(f) if args.silent else contextlib.nullcontext()):
            for i in range(args.n_games):
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
    winrate = results['AGENT_B_SCORE'] / float(args.n_games)

    print(results)
    print(f"AGENT_B_WINRATE={winrate:.4f}")


if __name__ == '__main__':
    main()
