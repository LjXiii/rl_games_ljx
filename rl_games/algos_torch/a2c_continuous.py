from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
import torch 


class A2CAgent(a2c_common.ContinuousA2CBase):
    """Continuous PPO Agent

    The A2CAgent class inerits from the continuous asymmetric actor-critic class and makes modifications for PPO.

    """
    def __init__(self, base_name, params):
        """Initialise the algorithm with passed params

        Args:
            base_name (:obj:`str`): Name passed on to the observer and used for checkpoints etc.
            params (:obj `dict`): Algorithm parameters

        """

        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        build_config = {
            'actions_num' : self.actions_num, #动作数量
            'input_shape' : obs_shape, #输入形状
            'num_seqs' : self.num_actors * self.num_agents, #序列数量
            'value_size': self.env_info.get('value_size',1), #获得值的大小，不存在的情况下默认为1
            'normalize_value' : self.normalize_value, #是否对值归一化
            'normalize_input': self.normalize_input, #是否对输入归一化
        } #定义了一个名为 build_config 的字典，其中包含了一系列键值对，用于配置构建环境的参数
        
        self.model = self.network.build(build_config) #创建模型
        self.model.to(self.ppo_device) #将模型移动到指定的设备
        self.states = None
        self.init_rnn_from_model(self.model) #从模型中初始化循环神经网络（RNN）
        self.last_lr = float(self.last_lr) #将 self.last_lr 属性转换为浮点数类型，以确保学习率参数的正确类型
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound' 根据配置中的参数获取损失类型，如果配置中未指定，则默认为 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay) #使用 Adam 优化器来初始化模型的优化器，传递了模型的参数、学习率、epsilon 值和权重衰减参数

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length, #视野长度
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_length' : self.seq_length,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs, #最大周期数
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done #完成时是否将rnn置0
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn):
        state = self.get_full_state_weights() #获取模型的完整状态及权重信息
        torch_ext.save_checkpoint(fn, state) #将模型状态保存到指定的文件中

    def restore(self, fn, set_epoch=True):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch) #恢复模型状态

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_gradients(self, input_dict):
        """Compute gradients needed to step the networks of the algorithm.

        Core algo logic is defined here

        Args:
            input_dict (:obj:`dict`): Algo inputs as a dict.

        """
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions'] #旧的动作对数概率值
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma'] #旧的标准差
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0 #学习率
        curr_e_clip = self.e_clip #截断

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict) #调用了模型 self.model，并传递了batch_dict作为输入，得到了模型的输出结果res_dict
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            #从 res_dict 中提取了动作对数概率 action_log_probs、价值估计 values、熵 entropy、均值 mu 和标准差 sigma

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip) #计算损失函数

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value) #如果模型具有值损失（self.has_value_loss 为真），则计算评论家损失 c_loss
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu) #损失边界类型为 'regularisation'，则计算正则化损失
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu) #损失边界类型为 'bound'，则计算边界损失
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef #计算总损失
            
            if self.multi_gpu:
                self.optimizer.zero_grad() #如果使用多GPU，将优化器中的梯度清零，以准备进行后续的反向传播和梯度更新
            else: #如果使用单GPU，遍历模型的参数，并将每个参数的梯度设置为 None，以手动清零梯度
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward() #self.scaler.scale(loss)：将损失值 loss 缩放到合适的范围，以便在混合精度训练中使用 .backward()：执行反向传播计算梯度，根据损失值计算模型参数的梯度
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step() #将梯度截断并执行参数更新

        with torch.no_grad(): #表明接下来计算不用梯度
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl) #计算kl散度
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                #将 KL 散度与 rnn_masks 相乘，以过滤掉不需要考虑的部分，对结果进行求和，并除以 rnn_masks 中的元素数量，以获得平均 KL 散度

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch, #当前状态值的预测值
            'returns' : return_batch, #回报值
            'new_neglogp' : action_log_probs, #新动作的负对数概率
            'old_neglogp' : old_action_log_probs_batch, #旧动作的负对数概率
            'masks' : rnn_masks #RNN 掩码
        }, curr_e_clip, 0)      

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss) #mu.detach()分离均值，sigma.detach()分离标准差

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss


