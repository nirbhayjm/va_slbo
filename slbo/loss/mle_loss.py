# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import lunzi.nn as nn
import numpy as np
import tensorflow as tf
from lunzi import Tensor
from slbo.utils.normalizer import Normalizers


class MLELoss(nn.Module):
    op_train: Tensor
    op_grad_norm: Tensor
    _step: int
    _criterion: nn.Module
    _normalizers: Normalizers
    _model: nn.Module

    def __init__(
        self,
        model: nn.Module,
        normalizers: Normalizers,
        dim_state: int,
        dim_action: int,
        criterion: nn.Module,
        step=4,
    ):
        super().__init__()
        self._step = step
        self._criterion = criterion
        self._model = model
        self._normalizers = normalizers
        with self.scope:
            self.op_states = tf.placeholder(tf.float32, shape=[step, None, dim_state])
            self.op_actions = tf.placeholder(tf.float32, shape=[step, None, dim_action])
            self.op_rewards = tf.placeholder(tf.float32, shape=[step, None])
            self.op_masks = tf.placeholder(tf.float32, shape=[step, None])
            self.op_next_states_ = tf.placeholder(
                tf.float32, shape=[step, None, dim_state]
            )

        self.op_loss, self.op_ma_loss = self(
            self.op_states,
            self.op_actions,
            self.op_next_states_,
            self.op_rewards,
            self.op_masks,
        )

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        next_states_: Tensor,
        rewards: Tensor,
        masks: Tensor,
    ):
        if len(states.shape) == 3:
            states = states.reshape((-1, states.shape[2]))
            actions = actions.reshape((-1, actions.shape[2]))
            next_states_ = next_states_.reshape((-1, next_states_.shape[2]))

        cur_states = states
        ns_diffs = self._model(cur_states, actions)
        pred_ns = cur_states + ns_diffs
        loss = self._criterion(pred_ns, next_states_)
        va_loss = loss * 0
        return loss, va_loss

    @nn.make_method(fetch="loss")
    def get_loss(self, states, next_states_, actions, rewards, masks):
        pass

    def build_backward(self, lr: float, weight_decay: float, max_grad_norm=2.0):
        loss = self.op_loss.reduce_mean(name="Loss")

        optimizer = tf.train.AdamOptimizer(lr)
        params = self._model.parameters()
        regularization = weight_decay * tf.add_n(
            [tf.nn.l2_loss(t) for t in params], name="regularization"
        )

        grads_and_vars = optimizer.compute_gradients(
            loss + regularization, var_list=params
        )
        print([var.name for grad, var in grads_and_vars])
        clip_grads, op_grad_norm = tf.clip_by_global_norm(
            [grad for grad, _ in grads_and_vars], max_grad_norm
        )
        clip_grads_and_vars = [
            (grad, var) for grad, (_, var) in zip(clip_grads, grads_and_vars)
        ]
        self.op_train = optimizer.apply_gradients(clip_grads_and_vars)
        self.op_grad_norm = op_grad_norm
