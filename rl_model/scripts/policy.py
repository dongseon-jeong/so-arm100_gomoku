from typing import Optional, Tuple, List

from stable_baselines3.ppo.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

import gymnasium as gym

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch as th
from torch.distributions import Categorical


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # 반드시 먼저 부모 초기화 호출
        super().__init__(*args, **kwargs)
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        self.features_extractor_class = base_FlattenExtractor

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            13*13,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def forward(self, obs_dict, deterministic=False):
        # obs = obs_dict["obs"].float().to(self.device)
        action_mask = obs_dict["action_mask"].bool().to(self.device)

        features = self.extract_features(obs_dict)  # obs는 Tensor
        
        latent_pi, latent_vf = self.mlp_extractor(features)

        logits = self.action_net(latent_pi)
        dist = MaskedCategorical(logits=logits, mask=action_mask)

        actions = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(latent_pi)
        return CategoricalDistribution(logits.shape[-1]), logits

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        
        try:

            # if isinstance(observation, dict):
            obs_tensor = {'obs': torch.FloatTensor(observation['obs']).to(self.device).unsqueeze(0),
                        'action_mask': torch.FloatTensor(observation['action_mask']).to(self.device).unsqueeze(0) }
            # else:
            #     raise TypeError("Expected observation to be a dict with 'obs' and 'action_mask'")
        except:
            print(observation)
        
        with torch.no_grad():
            actions, _, _ = self.forward(obs_tensor, deterministic=deterministic)
        

        return actions.cpu().numpy(), state

    def predict_values(self, obs):

        features = self.extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def extract_features(self, obs, features_extractor: Optional[BaseFeaturesExtractor] = None):
        
        # obs가 dict인 경우 'obs'만 추출
        if isinstance(obs, dict):
            obs = obs['obs']

        if self.share_features_extractor:
            return self.base_extract_features(obs, self.features_extractor)
        else:
            pi_features = self.base_extract_features(obs, self.pi_features_extractor)
            vf_features = self.base_extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

    def base_extract_features(self, obs, features_extractor):
        # self.observation_space가 Dict이라면 'obs' 서브공간 사용
        if isinstance(self.observation_space, gym.spaces.Dict):
            obs_space = self.observation_space.spaces['obs']
        else:
            obs_space = self.observation_space

        preprocessed_obs = self.base_preprocess_obs(obs, obs_space)

        return features_extractor(preprocessed_obs)

    def base_preprocess_obs(self, obs, observation_space):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # obs가 Box일 경우 그대로 float 처리
        if isinstance(observation_space, gym.spaces.Box):
            return obs.float()

        # obs가 Discrete일 경우 one-hot 인코딩
        elif isinstance(observation_space, gym.spaces.Discrete):
            return F.one_hot(obs.long(), num_classes=observation_space.n).float()

        # obs가 MultiDiscrete일 경우 각 차원을 one-hot 후 concat
        elif isinstance(observation_space, gym.spaces.MultiDiscrete):
            nvec = observation_space.nvec
            return th.cat(
                [
                    F.one_hot(obs_.long(), num_classes=int(nvec[idx])).float()
                    for idx, obs_ in enumerate(th.split(obs.long(), 1, dim=1))
                ],
                dim=-1,
            ).view(obs.shape[0], sum(nvec))

        else:
            raise NotImplementedError(f"Unsupported observation type: {type(observation_space)}")

    def evaluate_actions(self, obs_dict, actions):
        obs = obs_dict["obs"].float().to(self.device)
        action_mask = obs_dict["action_mask"].bool().to(self.device)

        features = self.extract_features(obs_dict)
        latent_pi, latent_vf = self.mlp_extractor(features)

        logits = self.action_net(latent_pi)
        dist = MaskedCategorical(logits=logits, mask=action_mask)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(latent_vf)

        return values, log_prob, entropy

class MaskedCategorical(Categorical):
    def __init__(self, logits, mask):
        logits = logits.clone()
        logits[~mask] = -1e8  # 매우 작은 값으로 설정해서 softmax에서 무시
        super().__init__(logits=logits)


class base_FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space: The observation space of the environment
    """

    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations['obs']
        return self.flatten(observations)