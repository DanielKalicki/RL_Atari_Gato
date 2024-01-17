import os
os.environ['WANDB_CONSOLE'] = 'off'

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer, \
    LazyTensorStorage, SamplerWithoutReplacement, TensorDictPrioritizedReplayBuffer
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal
from torchrl.objectives import ClipPPOLoss, A2CLoss
from torchrl.objectives.value import GAE

from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    EnvCreator,
    ExplorationType,
    GrayScale,
    GymEnv,
    NoopResetEnv,
    ParallelEnv,
    Resize,
    RewardClipping,
    RewardSum,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    OneHotCategorical,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
    NoisyLinear,
)
from collections import OrderedDict, defaultdict
from torchrl.data.replay_buffers import ReplayBuffer
from tqdm import tqdm
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-log", help="enable wandb logging", default=False)
args = parser.parse_args()

from torchrl.record.loggers import generate_exp_name, get_logger
from tensordict import TensorDict
import time
import math
import random
import torch.nn.functional as F

timestr = time.strftime("%Y%m%d-%H%M%S")

device = "cpu" if not torch.cuda.device_count() else "cuda"

class CReLU(nn.Module):
    def __init__(self, inplace=False):
        super(CReLU, self).__init__()

    def forward(self, x):
        x = torch.cat((x,-x),dim=-1)
        return F.relu(x)

def make_env(env_name, num_envs, test=False):
    def make_base_env(env_name, test=False):
        env = GymEnv(env_name, frame_skip=4, from_pixels=True, pixels_only=False, device=device, full_action_space=True)
        env = TransformedEnv(env)
        env.append_transform(NoopResetEnv(noops=30, random=True))
        if not test:
            env.append_transform(EndOfLifeTransform())
        return env

    if num_envs > 1:
        env = ParallelEnv(num_envs, EnvCreator(lambda: make_base_env(env_name, test)))
    else:
        env = make_base_env(env_name, test)
    env = TransformedEnv(env)
    env.append_transform(ToTensorImage())
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=4, dim=-3))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=4500))
    if not test:
        env.append_transform(RewardClipping(-1, 1))
    env.append_transform(DoubleToFloat())
    env.append_transform(VecNorm(in_keys=["pixels"]))
    return env

def main():
    # env_name = "BoxingNoFrameskip-v4"
    # env_name = "PongNoFrameskip-v4"
    # env_name = "FreewayNoFrameskip-v4"
    env_name = "EnduroNoFrameskip-v4"
    # env_name = "DoubleDunkNoFrameskip-v4"
    # env_name = "GopherNoFrameskip-v4"
    # env_name = "FishingDerbyNoFrameskip-v4"
    # env_name = "TennisNoFrameskip-v4"
    # env_name = "QbertNoFrameskip-v4"
    frame_skip = 4
    total_frames = 40_000_000 // frame_skip
    frames_per_batch = 4096*4 // frame_skip
    storage_frames_per_batch = frames_per_batch
    num_epochs = 10
    sub_batch_size = 1024*1
    max_grad_norm = 40

    # tr_input_dim = 2048
    tr_input_dim = 64*7*7

    if args.log:
        exp_name = generate_exp_name("PPO", f"Atari_{env_name}")
        logger = get_logger(
            "wandb", logger_name="ppo", experiment_name=exp_name, project='rl_atari_gato', entity='danielkalicki'
        )

    # test_env_pong = make_env("PongNoFrameskip-v4", 1, test=True)
    # test_env_boxing = make_env("BoxingNoFrameskip-v4", 1, test=True)
    # test_env_freeway = make_env("FreewayNoFrameskip-v4", 1, test=True)
    test_env_enduro = make_env("EnduroNoFrameskip-v4", 1, test=True)
    # test_env_doubledunk = make_env("DoubleDunkNoFrameskip-v4", 1, test=True)
    # test_env_gopher = make_env("GopherNoFrameskip-v4", 1, test=True)
    # test_env_fishing = make_env("FishingDerbyNoFrameskip-v4", 1, test=True)
    # test_env_tennis = make_env("TennisNoFrameskip-v4", 1, test=True)
    # test_env_qbert = make_env("QbertNoFrameskip-v4", 1, test=True)
    test_envs = [test_env_enduro]
    test_env = test_envs[-1]

    action_dim = test_env.action_spec.shape[0]

    # ------------------ TRANSFORMER ------------------
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)

    class Transformer(nn.Module):
        def __init__(self, tr_input_dim, action_dim):
            super().__init__()

            self.action_dim = action_dim

            self.state_enc_net = nn.Sequential(OrderedDict([
                ('ConvNet', ConvNet(
                    activation_class=torch.nn.ReLU,
                    num_cells=[32*2, 64*4, 64],
                    kernel_sizes=[8, 4, 3],
                    strides=[4, 2, 1],
                )),
            ])).to(device)

        def forward(self, x_1, x_2):
            if x_2 is None: # first action
                x_2 = torch.zeros((x_1.shape[0], self.action_dim), dtype=torch.int64).to(device)

            state_enc = self.state_enc_net(x_1) 

            return state_enc

    transformer_net = Transformer(tr_input_dim, action_dim).to(device)
    transformer_module = TensorDictModule(
        in_keys=["pixels", "action"],
        module = transformer_net,
        out_keys=["tr_out"]
    )

    # ------------------ ACTOR ------------------
    actor_net = nn.Sequential(OrderedDict([
        ('fc1',     nn.Linear(tr_input_dim, 512)),
        ('act1',    nn.ReLU()),
        ('fc2',     nn.Linear(512, action_dim)), # [B, Action]
    ])).to(device)
    actor_module = TensorDictModule(
        in_keys = ["tr_out"],
        module = actor_net,
        out_keys = ["logits"],
    )
    actor_module = TensorDictSequential(transformer_module, actor_module)

    rollout = test_env.rollout(3)
    print("rollout of three steps:", rollout)
    transformer_module(rollout)

    policy_module = ProbabilisticActor(
        in_keys = ["logits"],
        module = actor_module,
        spec = test_env.action_spec,
        return_log_prob = True,
        distribution_class = OneHotCategorical,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # ------------------ CRITIC ------------------
    value_net = nn.Sequential(OrderedDict([
        ('fc1',     nn.Linear(tr_input_dim, 512)),
        ('act1',    nn.ReLU()),
        ('fc2',     nn.Linear(512, 1)), # [B, ValueDim=1]
        ('act2',    nn.Tanh()),
    ])).to(device)
    value_module = TensorDictModule(
        in_keys = ["tr_out"],
        module = value_net,
        out_keys = ["state_value"],
    )
    value_module = TensorDictSequential(transformer_module, value_module)

    # ------------------ DATA COLLECTION ------------------
    collector = SyncDataCollector(
        create_env_fn=make_env(env_name, 8),
        policy=policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1
    )

    sampler = SamplerWithoutReplacement()
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(storage_frames_per_batch, device="cpu"),
        sampler=sampler,
        batch_size=sub_batch_size,
    )

    # ------------------ LOSS ------------------
    advantage_module = GAE(
        gamma=0.9, lmbda=0.95, value_network=value_module, average_gae=False
    )
    loss_module = ClipPPOLoss(
        actor=policy_module,
        critic=value_module,
        clip_epsilon=0.1,
        entropy_bonus=bool(1e-4),
        entropy_coef=0.01, #1e-3,
        critic_coef=1.0,
        separate_losses=False, # loss function is decreasing
        loss_critic_type="l2", # smooth_l1 - >based on https://arxiv.org/pdf/2011.14826v2.pdf hubbert loss is better than L2
        normalize_advantage=True, # improves performance on 2 tasks
    )

    advantage_module.set_keys(done="end-of-life", terminated="end-of-life")
    loss_module.set_keys(done="end-of-life", terminated="end-of-life")

    # ------------------ OPTIMIZER ------------------
    optim = torch.optim.Adam(loss_module.parameters(), lr=2.5e-4, weight_decay=0.0, eps=1e-6) # weight_decay=L2=5e-4
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_frames // frames_per_batch, 0.0)

    # ------------------ TRAIN LOOP ------------------
    collected_frames = 0
    sampling_start = time.time()
    pbar = tqdm(total=total_frames)
    losses = TensorDict({}, batch_size=[num_epochs, storage_frames_per_batch // sub_batch_size])

    for i, data in enumerate(collector):
        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch * frame_skip
        pbar.update(data.numel())

        # Get training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "terminated"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "terminated"]]
            log_info.update({
                "train/reward": episode_rewards.mean().item(), 
                "train/episode_length": episode_length.sum().item() / len(episode_length),
            })

        training_start = time.time()
        for j in range(num_epochs):
            # Compute GAE
            with torch.no_grad():
                data = advantage_module(data)
            data_reshape = data.reshape(-1)

            # Update the replay buffer
            replay_buffer.extend(data_reshape)

            for k, batch in enumerate(replay_buffer):
                # Get a data batch
                batch = batch.to(device)

                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, k] = loss.select("loss_critic", "loss_entropy", "loss_objective").detach()
                loss_sum = (loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"])

                # Backward pass
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(list(loss_module.parameters()), max_norm=max_grad_norm)

                # Update the networks
                optim.step()
                optim.zero_grad()

        # Get training losses and times
        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        log_info.update({
            "train/lr": optim.param_groups[0]["lr"],
            "train/sampling_time": sampling_time,
            "train/training_time": training_time,
        })

        # Get test rewards
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            # for test_env in [test_envs[-1]]:
            for test_env in test_envs:
                policy_module.eval()
                eval_start = time.time()
                test_rewards = []
                for _ in range(1):
                    td_test = test_env.rollout(policy=policy_module, auto_reset=True, auto_cast_to_device=True, break_when_any_done= True, max_steps=10_000)
                    reward = td_test["next", "episode_reward"][td_test["next", "done"]]
                    test_rewards.append(reward.cpu())
                del td_test
                test_rewards = torch.cat(test_rewards, 0).mean()
                print(test_rewards.mean())

                eval_time = time.time() - eval_start
                log_info.update({f"eval/{test_env.env_name}/reward": test_rewards.mean(),f"eval/{test_env.env_name}/time": eval_time,})
                
        if args.log:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

        if i%10 == 0:
            save_dict = {
                'epoch': i,
                'policy_module': policy_module.state_dict(),
                'value_module': value_module.state_dict()
            }
            torch.save(save_dict, f"./save/{env_name}_{timestr}")

        collector.update_policy_weights_()
        scheduler.step()
        sampling_start = time.time()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")

if __name__ == '__main__':
    main()