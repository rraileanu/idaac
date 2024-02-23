import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from multi_agent_rollout_buffer import MultiAgentRolloutBuffer

class PPO():
    """
    PPO
    """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update_with_multi_traj(self, rollouts: MultiAgentRolloutBuffer, batch_size=None,
                               epoch=None) -> tuple:
        """
        Update the network with multiple trajectories of accumulated data

        :param rollouts:
        :param batch_size: (int)
        :param epoch: (int) (Optional)
        :return:
        """

        loss_epoch = 0
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        num_updates = 0
        traj_len_epoch = 0

        total_steps = rollouts.flatten_and_augment(epoch=epoch, gif_steps=75)
        if total_steps == 0:
            return None, None, None, None, None

        # number of times data should be trained over per epoch
        for e in range(self.ppo_epoch):
            for rollout_batch in rollouts.get(total_steps, batch_size):
                obs_batch, actions_batch, old_action_log_probs_batch, value_preds_batch, \
                    adv_targ, return_batch = rollout_batch

                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                                     (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                             value_losses_clipped).mean()

                # Update actor-critic using both PPO Loss
                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                num_updates += 1
                traj_len_epoch += obs_batch.size(dim=0)

                loss_epoch += loss.item()
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return traj_len_epoch, loss_epoch, value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    '''
    Old fn to remove
    def update(self, rollouts):
        # takes everything but last return and everything but last value pred to compute adv across board
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                    old_action_log_probs_batch, adv_targ = sample

                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch)
                    
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                                value_losses_clipped).mean()
                
                # Update actor-critic using both PPO Loss
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                    dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)
                self.optimizer.step()  
                                
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
    '''
