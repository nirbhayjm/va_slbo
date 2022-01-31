import lunzi.nn as nn
import numpy as np
import tensorflow as tf
from lunzi import Tensor
from slbo.policies import BasePolicy
from slbo.utils.normalizer import Normalizers
from slbo.utils.runner import Runner


class ValueAwareLoss(nn.Module):
    op_train: Tensor
    op_grad_norm: Tensor
    _value_function: nn.Module
    _step: int
    _criterion: nn.Module
    _normalizers: Normalizers
    _model: nn.Module
    _va_norm: str

    def __init__(
        self,
        model: nn.Module,
        value_function: nn.Module,
        normalizers: Normalizers,
        dim_state: int,
        dim_action: int,
        criterion: nn.Module,
        va_loss_coeff: float,
        va_norm: str = "l1",
        virt_runner: Runner = None,
        step=4,
    ):
        super().__init__()
        self._step = step
        self._criterion = criterion
        self._model = model
        self._normalizers = normalizers

        self._value_function = value_function
        self._va_norm = va_norm
        self._virt_runner = virt_runner
        self._va_loss_coeff = va_loss_coeff

        with self.scope:
            self.op_states = tf.placeholder(tf.float32, shape=[step, None, dim_state])
            self.op_actions = tf.placeholder(tf.float32, shape=[step, None, dim_action])
            self.op_rewards = tf.placeholder(tf.float32, shape=[step, None])
            self.op_masks = tf.placeholder(tf.float32, shape=[step, None])
            self.op_next_states_ = tf.placeholder(
                tf.float32, shape=[step, None, dim_state]
            )

        (
            self.op_loss,
            self.op_va_loss,
        ) = self(
            self.op_states,
            self.op_actions,
            self.op_next_states_,
            self.op_rewards,
            self.op_masks,
        )
        self.op_value_pred = self.value_forward(self.op_states, self.op_masks)

    @nn.make_method(fetch="value_pred")
    def get_value(self, states, masks):
        pass

    def value_forward(self, states: Tensor, masks: Tensor):
        states = states.reshape((-1, states.shape[2]))
        v_pred = self._value_function(states)
        v_pred = v_pred.reshape((self._step, -1))
        return v_pred

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        next_states_: Tensor,
        rewards: Tensor,
        masks: Tensor,
    ):
        states = states.reshape((-1, states.shape[2]))
        actions = actions.reshape((-1, actions.shape[2]))
        next_states_ = next_states_.reshape((-1, next_states_.shape[2]))
        lhs_ns_value = self._value_function(next_states_)

        ns_sample = self._model(states, actions)
        rhs_ns_value = self._value_function(ns_sample)

        model_advantage = lhs_ns_value - rhs_ns_value
        # Shape (B,)

        if self._va_norm == "l2":
            va_loss_batch = model_advantage ** 2
        elif self._va_norm == "l1":
            va_loss_batch = tf.abs(model_advantage)
        else:
            raise ValueError(f"{self._va_norm}")

        va_loss_batch = (
            tf.reduce_sum(va_loss_batch.reshape((self._step, -1)), 0) / self._step
        )
        # Shape (B,)

        va_loss = tf.reduce_mean(va_loss_batch, axis=0)
        slbo_add_loss = va_loss * 0
        loss = self._va_loss_coeff * va_loss

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
