# -*- coding: utf-8 -*-
"""Standalone PPO tutorial. Copy this file anywhere and run: python3 FILE.py --help.
Only NumPy, Matplotlib and Pillow are required; no project-local imports.
"""
import os
import tempfile
from pathlib import Path
# 小矩阵计算默认使用单线程；setdefault 保留用户显式配置，必须在导入 NumPy 前设置。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ppo-tutorial-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ppo-tutorial-cache"))
import argparse
import json
import numpy as np
import matplotlib
# 无窗口绘图，支持服务器和没有桌面的环境。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

STAGE = 5

# 状态列依次为 x、v、theta、omega；count 个环境批量推进。
class CartPole:
    """Uniform rod, explicit Euler, dt=0.02; Gymnasium-compatible boundaries."""
    def __init__(self, seed=0, count=1):
        self.rng = np.random.default_rng(seed)
        self.count = count
        self.state = np.zeros((count, 4))
        self.age = np.zeros(count, dtype=int)
        self.max_steps = 500
        self.reset(np.ones(count, dtype=bool))

    def reset(self, mask):
        self.state[mask] = self.rng.uniform(-0.05, 0.05, (int(mask.sum()), 4))
        self.age[mask] = 0

    def step(self, actions, continuous=False, external_force=0.):
        x, v, theta, omega = self.state.T.copy()
        force = np.asarray(actions, dtype=float) if continuous else np.where(np.asarray(actions) == 1, 10., -10.)
        if not np.isfinite(force).all() or np.any(np.abs(force) > 10):
            raise ValueError("Force must be finite and within [-10, 10] N")
        external_force = np.asarray(external_force, dtype=float)
        if not np.isfinite(external_force).all():
            raise ValueError("External force must be finite")
        # 仅控制力受 +/-10 N 限制；外力独立进入动力学，总力不再裁剪。
        force = force + external_force
        sin, cos = np.sin(theta), np.cos(theta)
        # 均匀杆动力学：M=1 kg、m=0.1 kg、质心距 l=0.5 m，以下依次计算耦合项和两种加速度。
        temp = (force + .05 * omega**2 * sin) / 1.1
        angular = (9.8 * sin - cos * temp) / (.5 * (4/3 - .1 * cos**2 / 1.1))
        acceleration = temp - .05 * angular * cos / 1.1
        # 显式 Euler：全部使用旧状态更新，dt=0.02 s。
        self.state = np.column_stack((x + .02*v, v + .02*acceleration,
                                      theta + .02*omega, omega + .02*angular))
        self.age += 1
        # 越界是真实失败；达到步数上限只是截断，两者对价值自举的处理不同。
        terminated = (np.abs(self.state[:, 0]) > 2.4) | (np.abs(self.state[:, 2]) > np.deg2rad(12))
        truncated = (self.age >= self.max_steps) & ~terminated
        return self.state.copy(), np.ones(self.count), terminated, truncated

# Actor 与 Critic 各用独立的 64-64 tanh 网络，a/v 前缀区分参数。
class Network:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.p = {}
        for prefix, output in [('a', 2), ('v', 1)]:
            for i, (n, m) in enumerate([(4,64), (64,64), (64,output)]):
                self.p[f'{prefix}w{i}'] = rng.normal(size=(n,m)) * np.sqrt(1/n) * (0.01 if prefix == 'a' and i == 2 else 1)
                self.p[f'{prefix}b{i}'] = np.zeros(m)
        self.m = {k:np.zeros_like(v) for k,v in self.p.items()}
        self.v = {k:np.zeros_like(v) for k,v in self.p.items()}
        self.iteration = 0

    def forward(self, states, prefix):
        # 按物理量典型尺度归一化；连续策略的第五维为上一推力。
        h0 = states / np.array([2.4, 2., .21, 2.])
        h1 = np.tanh(h0 @ self.p[prefix+'w0'] + self.p[prefix+'b0'])
        h2 = np.tanh(h1 @ self.p[prefix+'w1'] + self.p[prefix+'b1'])
        return h2 @ self.p[prefix+'w2'] + self.p[prefix+'b2'], (h0,h1,h2)

    def predict(self, states):
        logits, _ = self.forward(states,'a')
        # 减去行最大值后再做 Softmax，避免指数溢出。
        z = logits - logits.max(axis=1,keepdims=True)
        probs = np.exp(z); probs /= probs.sum(axis=1,keepdims=True)
        values, _ = self.forward(states,'v')
        return probs, values[:,0]

    # cache 保存各层输入；delta 是损失对当前层线性输出的导数。
    def backward(self, prefix, cache, delta, grads):
        for i in [2,1,0]:
            grads[prefix+f'w{i}'] = cache[i].T @ delta
            grads[prefix+f'b{i}'] = delta.sum(axis=0)
            if i:
                # 链式法则：乘权重转置，再乘 tanh 导数 1-h^2。
                delta = (delta @ self.p[prefix+f'w{i}'].T) * (1-cache[i]**2)

    def update(self, states, actions, oldlog, advantages, targets, lr):
        probs, values = self.predict(states)
        logp = np.log(probs + 1e-12)
        newlog = logp[np.arange(len(actions)),actions]
        ratio = np.exp(newlog-oldlog)
        # PPO 裁剪平坦区关闭该策略项梯度：正优势 ratio>1.2，负优势 ratio<0.8。
        active = ~(((advantages>0)&(ratio>1.2)) | ((advantages<0)&(ratio<.8)))
        entropy = -(probs*logp).sum(axis=1)
        # softmax 的 p-one_hot(a) 导数，随后按优势、概率比和裁剪掩码加权。
        delta = probs.copy(); delta[np.arange(len(actions)),actions] -= 1
        delta *= (advantages*ratio*active)[:,None]
        # 负熵项的 logits 梯度；最小化负熵鼓励探索。
        delta += .01 * probs * (logp + entropy[:,None])
        grads = {}
        self.backward('a',self.forward(states,'a')[1],delta/len(actions),grads)
        # 价值目标固定；0.5*(V-target)/批大小 对应本实现价值损失系数。
        self.backward('v',self.forward(states,'v')[1],(.5*(values-targets)/len(actions))[:,None],grads)
        # 先汇总全局梯度范数，再按上限 0.5 同比例裁剪，随后进行 Adam 更新。
        norm = np.sqrt(sum((g*g).sum() for g in grads.values()))
        self.iteration += 1
        for k,g in grads.items():
            g *= min(1., .5/(norm+1e-8))
            self.m[k] = .9*self.m[k]+.1*g
            self.v[k] = .999*self.v[k]+.001*g*g
            # Adam 使用偏差修正的一阶/二阶矩更新；iteration 统计优化步而非交互步。
            self.p[k] -= lr*(self.m[k]/(1-.9**self.iteration))/(np.sqrt(self.v[k]/(1-.999**self.iteration))+1e-8)
        return [float(entropy.mean()),float(np.mean((values-targets)**2)),
                float(np.mean((ratio-1)-(newlog-oldlog))),float(np.mean(np.abs(ratio-1)>.2)),float(norm)]

    def save(self, path):
        np.savez(path, **self.p)

    # 只恢复网络参数，不恢复 Adam 动量；名称、形状及有限值必须全部匹配。
    def load(self,path):
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(self.p):
                raise ValueError("Checkpoint parameter names do not match this policy")
            restored = {}
            for key, expected in self.p.items():
                value = data[key]
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid shape or non-finite parameter: {key}")
                restored[key] = value.copy()
        self.p = restored

# 数组形状为 [时间,环境]；价值目标使用未标准化优势加旧价值。
def gae(rewards, values, next_values, terminated, truncated, gamma=.99, lam=.95):
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1])
    for t in reversed(range(len(rewards))):
        # 真实失败禁止自举；时间截断仍使用重置前最终观测的价值。
        delta = rewards[t]+gamma*next_values[t]*(~terminated[t])-values[t]
        # 失败和截断都阻断 GAE 跨回合递推，防止把重置后的新回合连进来。
        running = delta+gamma*lam*(~(terminated[t]|truncated[t]))*running
        advantages[t] = running
    return advantages, advantages+values

# 将上一控制力加入观测，使力变化代价对应的控制历史对策略可见。
class ForceEnvironment(CartPole):
    def __init__(self, seed=0, count=1, shaped=True):
        self.previous = np.zeros(count)
        self.shaped = shaped
        super().__init__(seed, count)

    def reset(self, mask):
        super().reset(mask)
        self.previous[mask] = 0

    def observe(self):
        return np.column_stack((self.state, self.previous))

    def step(self, force, external_force=0.):
        _, reward, term, trunc = super().step(force, continuous=True, external_force=external_force)
        # 第二篇关闭形状化奖励；第三篇增加姿态、位置、用力和力变化代价。
        if self.shaped:
            reward -= .15*(self.state[:,2]/.21)**2 + .05*(self.state[:,0]/2.4)**2
            reward -= .01*(force/10)**2 + .05*((force-self.previous)/10)**2
        self.previous = np.asarray(force).copy()
        return self.observe(), reward, term, trunc

class GaussianNetwork(Network):
    def __init__(self, seed=0):
        super().__init__(seed)
        self.entropy_coef = .01
        rng=np.random.default_rng(seed+71)
        for prefix in ['a','v']:
            self.p[prefix+'w0']=rng.normal(size=(5,64))/np.sqrt(5)
        self.p['aw2']=self.p['aw2'][:,:1].copy()
        self.p['ab2']=np.zeros(1)
        # 全局可学习 log 标准差，不是随状态变化的网络输出；sigma=exp(logstd)。
        self.p['logstd']=np.array([-.7])
        self.m={k:np.zeros_like(v) for k,v in self.p.items()}
        self.v={k:np.zeros_like(v) for k,v in self.p.items()}

    def forward(self, states, prefix):
        # 按物理量典型尺度归一化；连续策略的第五维为上一推力。
        h0=states/np.array([2.4,2.,.21,2.,10.])
        h1=np.tanh(h0@self.p[prefix+'w0']+self.p[prefix+'b0'])
        h2=np.tanh(h1@self.p[prefix+'w1']+self.p[prefix+'b1'])
        return h2@self.p[prefix+'w2']+self.p[prefix+'b2'],(h0,h1,h2)

    def predict(self, states):
        return self.forward(states,'a')[0][:,0],self.forward(states,'v')[0][:,0]

    # F=10*tanh(z) 的对数密度必须减去变量变换的 log 雅可比。
    def logprob(self, z, mu):
        logstd=self.p['logstd'][0]
        # 用 logaddexp 稳定计算，避免 tanh 饱和时直接计算 log(1-tanh(z)^2)。
        log_jacobian=np.log(10.)+2*(np.log(2.)-z-np.logaddexp(0.,-2*z))
        return -.5*((z-mu)/np.exp(logstd))**2-logstd-.5*np.log(2*np.pi)-log_jacobian

    def actor_gradients(self, states, z, oldlog, advantages, entropy_noise):
        mu,_=self.predict(states); std=np.exp(self.p['logstd'][0])
        newlog=self.logprob(z,mu); ratio=np.exp(newlog-oldlog)
        # PPO 裁剪平坦区关闭该策略项梯度：正优势 ratio>1.2，负优势 ratio<0.8。
        active=~(((advantages>0)&(ratio>1.2))|((advantages<0)&(ratio<.8)))
        weight=-advantages*ratio*active
        dmu=weight*(z-mu)/std**2
        # dstd 实际是对 log(sigma) 的梯度，不能按对 sigma 的导数理解。
        dstd=weight*(((z-mu)/std)**2-1)
        # 重参数化采样估计有界动作分布的熵；后续梯度包含 tanh 雅可比的影响。
        entropy_z=mu+std*entropy_noise
        entropy=-self.logprob(entropy_z,mu)
        dmu += self.entropy_coef*2*np.tanh(entropy_z)
        dstd += self.entropy_coef*(-1+2*np.tanh(entropy_z)*std*entropy_noise)
        grads={}
        self.backward('a',self.forward(states,'a')[1],dmu[:,None]/len(z),grads)
        grads['logstd']=np.array([dstd.mean()])
        return grads, newlog, ratio, entropy

    def update(self, states, z, oldlog, advantages, targets, lr, rng):
        grads,newlog,ratio,entropy=self.actor_gradients(states,z,oldlog,advantages,rng.normal(size=len(z)))
        _,values=self.predict(states)
        # 价值目标固定；0.5*(V-target)/批大小 对应本实现价值损失系数。
        self.backward('v',self.forward(states,'v')[1],(.5*(values-targets)/len(z))[:,None],grads)
        # 先汇总全局梯度范数，再按上限 0.5 同比例裁剪，随后进行 Adam 更新。
        norm=np.sqrt(sum((g*g).sum() for g in grads.values())); self.iteration+=1
        for k,g in grads.items():
            g *= min(1.,.5/(norm+1e-8))
            self.m[k]=.9*self.m[k]+.1*g; self.v[k]=.999*self.v[k]+.001*g*g
            # Adam 使用偏差修正的一阶/二阶矩更新；iteration 统计优化步而非交互步。
            self.p[k]-=lr*(self.m[k]/(1-.9**self.iteration))/(np.sqrt(self.v[k]/(1-.999**self.iteration))+1e-8)
        # 限制探索尺度，避免标准差趋零或过大；推力边界仍由 tanh 保证。
        self.p['logstd'][:]=np.clip(self.p['logstd'],-4.,.5)
        return [float(entropy.mean()),float(np.mean((values-targets)**2)),
                float(np.mean((ratio-1)-(newlog-oldlog))),float(np.mean(np.abs(ratio-1)>.2)),float(norm)]

# 继承连续控制，以目标误差代替绝对位置观测，并增加位置和速度代价。
class PositionEnvironment(ForceEnvironment):
    def __init__(self,seed=0,count=1,target=0.,radius=.5,max_steps=1500):
        if abs(target)+radius>2.0: raise ValueError('Initial positions must stay inside +/-2 m')
        self.target=target; self.radius=radius
        super().__init__(seed,count,shaped=False)
        self.max_steps=max_steps

    def reset(self,mask):
        super().reset(mask)
        self.state[mask,0]=self.target+self.rng.uniform(-self.radius,self.radius,int(mask.sum()))

    def observe(self):
        observation=super().observe()
        observation[:,0]-=self.target
        return observation

    def step(self,force,external_force=0.):
        # 父类 step 会覆盖 previous，先保存旧力才能正确计算本步力变化。
        previous=self.previous.copy()
        obs,_,term,trunc=super().step(force,external_force=external_force)
        reward=1-.15*(self.state[:,2]/.21)**2-.01*(self.state[:,3]/2)**2
        # 有界位置代价避免远离目标时惩罚无限增大而诱导提前失败；同时保留姿态与平滑代价。
        reward-=.45*np.tanh((self.state[:,0]-self.target)/.5)**2
        reward-=.15*np.tanh(self.state[:,1]/.5)**2
        reward-=.01*(force/10)**2+.05*((force-previous)/10)**2
        return obs,reward,term,trunc

# 仅改变训练中的扰动分布，完整继承位置控制奖励。
class PushEnvironment(PositionEnvironment):
    def __init__(self,seed=0,count=8,**kwargs):
        self.push_rng=np.random.default_rng(seed+600)
        self.onset=np.zeros(count,dtype=int); self.duration=np.zeros(count,dtype=int)
        self.amplitude=np.zeros(count); self.strength=1.
        super().__init__(seed,count,**kwargs)

    def reset(self,mask):
        super().reset(mask); n=int(mask.sum())
        self.onset[mask]=self.push_rng.integers(80,350,n)
        # 脉冲持续 5~15 步，即 0.10~0.30 s；方向与幅值独立随机。
        self.duration[mask]=self.push_rng.integers(5,16,n)
        self.amplitude[mask]=self.push_rng.uniform(1.,self.strength,n)*self.push_rng.choice([-1.,1.],n)

    def step(self,force):
        external=self.amplitude*((self.age>=self.onset)&(self.age<self.onset+self.duration))
        return super().step(force,external_force=external)

# 相同种子和复制初态保证策略/外力条件可比；评估时不再更新网络。
def evaluate_pushes(net,seed=20000,episodes=30,forces=(0.,1.,-1.,3.,-3.,5.,-5.)):
    count=episodes*len(forces)
    env=PositionEnvironment(seed,count=count,radius=.5,max_steps=1500)
    initial=env.state[:episodes].copy()
    env.state[:]=np.tile(initial,(len(forces),1))
    magnitudes=np.repeat(forces,episodes); active=np.ones(count,dtype=bool)
    lengths=np.zeros(count,dtype=int); failed=np.zeros(count,dtype=bool)
    states=[]; controls=[]; disturbances=[]
    for t in range(1500):
        mu,_=net.predict(env.observe()); control=10*np.tanh(mu)
        # 固定测试协议：10.0 s 开始施力，持续 0.2 s，与训练随机脉冲分开。
        external=magnitudes if 500<=t<510 else np.zeros(count)
        env.step(control,external_force=external)
        term=(np.abs(env.state[:,0])>2.4)|(np.abs(env.state[:,2])>np.deg2rad(12))
        lengths[active]+=1; failed[active&term]=True
        # 已失败轨迹以 NaN 标记，不能把后续填零状态作为恢复或稳态样本。
        states.append(np.where(active[:,None],env.state,np.nan))
        controls.append(np.where(active,control,np.nan)); disturbances.append(external.copy())
        active&=~term
        env.state[~active]=0; env.previous[~active]=0
    states=np.array(states); controls=np.array(controls); disturbances=np.array(disturbances)
    records=[]; summaries=[]
    for i,force in enumerate(magnitudes):
        n=lengths[i]; data=states[:n,i]; recovery=None; upright=None
        held=(np.abs(data[:,0])<=.05)&(np.abs(data[:,1])<=.05)
        angle=np.abs(data[:,2])<=np.deg2rad(1.)
        # 位置恢复要求连续 2 s 同时满足 |位置误差|<=0.05 m、|速度|<=0.05 m/s。
        for start in range(510,n-99):
            if held[start:start+100].all(): recovery=(start+1)*.02-10.2; break
        for start in range(510,n-24):
            if angle[start:start+25].all(): upright=(start+1)*.02-10.2; break
        record={'force_n':float(force),'trial':int(i%episodes),'success':not bool(failed[i]),
                'length':int(n),'recovery_s':recovery,'upright_s':upright,
                'max_error_m':float(np.max(np.abs(data[500:,0]))) if n>500 else None,
                'max_angle_deg':float(np.rad2deg(np.max(np.abs(data[500:,2])))) if n>500 else None,
                'last5_error_m':float(np.sqrt(np.mean(data[-250:,0]**2))) if not failed[i] else None}
        records.append(record)
    for force in forces:
        group=[r for r in records if r['force_n']==force]
        # 恢复均值只统计成功恢复的幸存回合；必须同时报告存活率与有效回合数。
        valid=[r['recovery_s'] for r in group if r['success'] and r['recovery_s'] is not None]
        mean=lambda key:float(np.mean([r[key] for r in group if r[key] is not None])) if any(r[key] is not None for r in group) else None
        summaries.append({'force_n':force,'episodes':episodes,'survival_rate':float(np.mean([r['success'] for r in group])),
                          'recovered_survivors':len(valid),'recovery_s':float(np.mean(valid)) if valid else None,
                          'max_error_m':mean('max_error_m'),'max_angle_deg':mean('max_angle_deg'),
                          'last5_error_m':mean('last5_error_m')})
    return summaries,records

# 先比较存活率，再比较恢复代价；未恢复回合含失败都受到 20 s 惩罚。
def rank(summaries):
    survival=np.mean([r['survival_rate'] for r in summaries])
    cost=np.mean([(r['recovery_s'] or 0)*r['recovered_survivors']/r['episodes']+
                  20*(1-r['recovered_survivors']/r['episodes']) for r in summaries])
    return float(survival),-float(cost)

def train(seed,steps,out,initial=None):
    net=GaussianNetwork(seed)
    if initial is not None: net.load(initial)
    env=PushEnvironment(seed,count=8,radius=.5,max_steps=500)
    rng=np.random.default_rng(seed+40); history=[]; lengths=[]
    best=rank(evaluate_pushes(net,10000+seed,8,forces=(0.,3.,-3.))[0]); net.save(out/'policy.npz')
    for update in range(int(np.ceil(steps/2048))):
        # 课程上限在前半训练预算内从 1 N 增至 3 N；新幅度在回合重置时采样。
        env.strength=1+2*min(1.,update*2048/(steps*.5))
        states=[]; actions=[]; logs=[]; rewards=[]; values=[]; nextvalues=[]; terms=[]; truncs=[]
        # 每批 8 环境 x 256 步=2048 样本，先完成采样，再固定旧概率/优势做优化。
        for t in range(256):
            s=env.observe(); mu,v=net.predict(s); z=mu+np.exp(net.p['logstd'][0])*rng.normal(size=8)
            ns,r,term,trunc=env.step(10*np.tanh(z)); nv=net.predict(ns)[1]
            states.append(s); actions.append(z); logs.append(net.logprob(z,mu))
            rewards.append(r); values.append(v); nextvalues.append(nv); terms.append(term); truncs.append(trunc)
            lengths.extend(env.age[term|trunc].tolist()); env.reset(term|trunc)
        adv,targets=gae(*map(np.array,[rewards,values,nextvalues,terms,truncs]),gamma=.995)
        s=np.array(states).reshape(-1,5); z=np.array(actions).ravel(); oldlog=np.array(logs).ravel()
        # 只标准化 Actor 优势；价值目标保留原回报尺度，不能用标准化优势重算。
        adv=adv.ravel(); adv=(adv-adv.mean())/(adv.std()+1e-8); targets=targets.ravel()
        # 每批数据更新 4 轮，每轮打乱并拆成 64 样本小批量；随后重新采样。
        for epoch in range(4):
            order=rng.permutation(2048)
            for start in range(0,2048,64):
                ix=order[start:start+64]; net.update(s[ix],z[ix],oldlog[ix],adv[ix],targets[ix],1e-4,rng)
        row={'steps':(update+1)*2048,'strength_n':env.strength,'training_length':float(np.mean(lengths[-50:])) if lengths else 0.}
        if update%10==0 or update==int(np.ceil(steps/2048))-1:
            score,_=evaluate_pushes(net,10000+seed,8,forces=(0.,3.,-3.)); candidate=rank(score)
            row['validation']=score
            if candidate>=best: best=candidate; net.save(out/'policy.npz')
            print(f'steps={row["steps"]} validation rank={candidate} best={best}',flush=True)
        history.append(row)
    # 分别保存最终更新模型和验证最优模型；返回/评估采用验证最优，测试集不参与选择。
    net.save(out/'policy_final.npz'); net.load(out/'policy.npz')
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    return net

def rollout(force, checkpoint):
    net=GaussianNetwork(); net.load(checkpoint)
    env=PositionEnvironment(20000,radius=.5)
    rows=[]; post=[]
    for step in range(1500):
        control=float(10*np.tanh(net.predict(env.observe())[0][0]))
        disturbance=force if 500<=step<510 else 0.
        rows.append([*env.state[0].tolist(),control,disturbance])
        _,_,term,trunc=CartPole.step(env,np.array([control]),continuous=True,external_force=disturbance)
        # 回放直接调用基础动力学，需要手动同步上一推力，保持策略观测与训练一致。
        env.previous[:]=control
        post.append(env.state[0].copy())
        if term[0] or trunc[0]: break
    states=np.array(post); recovery=None
    held=(np.abs(states[:,0])<=.05)&(np.abs(states[:,1])<=.05)
    for start in range(510,len(states)-99):
        if held[start:start+100].all():
            recovery=(start+1)*.02-10.2; break
    metrics={'force_n':force,'duration_s':.2,'start_s':10.,'survived_30s':len(rows)==1500 and not bool(term[0]),
             'recovery_after_pulse_s':recovery,'max_position_error_after_pulse_m':float(np.abs(states[500:,0]).max()) if len(states)>500 else None,
             'max_angle_after_pulse_deg':float(np.rad2deg(np.abs(states[500:,2]).max())) if len(states)>500 else None}
    return rows,metrics

def render_push_animation(checkpoint,out,forces):
    out.mkdir(parents=True,exist_ok=True)
    forces=list(forces)
    if 3. not in forces: forces.append(3.)
    trials={}; metrics=[]
    for force in forces:
        rows,score=rollout(force,checkpoint); trials[str(force)]=rows; metrics.append(score)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (out/'trajectories.json').write_text(json.dumps(trials),encoding='utf-8')
    frames=[]
    # 优先从扰动前的 8 s 开始回放；若提前失败则从首帧展示，避免空 GIF。
    start_frame=400 if len(trials['3.0'])>400 else 0
    for i,row in enumerate(trials['3.0'][start_frame:1000:2]):
        im=frame(row[:5]); draw=ImageDraw.Draw(im)
        draw.text((15,40),f't={start_frame*.02+i*.04:.2f} s | disturbance={row[5]:+.1f} N',fill='#b43c32')
        draw.line((300,240,300,260),fill='#20906d',width=3)
        if row[5]:
            cx=300+row[0]*90
            draw.line((cx-90,185,cx-35,185),fill='#d74338',width=5)
            draw.polygon([(cx-35,185),(cx-45,178),(cx-45,192)],fill='#d74338')
        frames.append(im)
    frames[0].save(out/'pulse.gif',save_all=True,append_images=frames[1:],duration=40,loop=0)
    # 页面内嵌轨迹与指标，可直接离线打开；可见文字保持英文，避免中文字体依赖。
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Cart-Pole Disturbance Recovery</title>
<style>body{max-width:1000px;margin:25px auto;font-family:system-ui;background:#f3f6fa;color:#263747}canvas{background:white;width:100%;border-radius:12px}button,select,input{margin:8px;padding:8px}pre{white-space:pre-wrap}</style>
<h1>After Training: Disturbance and Recovery</h1><p>Position-control PPO receives an external force at t=10 s for 0.2 s. The green dashed line marks the 0 m target; red arrows show the external force. The policy uses its observations without retraining during the test.</p>
<select id="trial"><option value="3.0">Right +3 N</option><option value="5.0">Right +5 N</option><option value="1.0">Right +1 N</option><option value="-3.0">Left -3 N</option><option value="0.0">No disturbance</option></select>
<button id="play">Pause / Play</button><button id="jump">Jump Before Disturbance</button><button id="restart">Restart</button><span id="status"></span>
<canvas id="c" width="1000" height="360"></canvas><input id="seek" type="range" min="0" max="1499" value="400" style="width:95%"><canvas id="plot" width="1000" height="220"></canvas><pre id="metrics"></pre>
<script>const trials=DATA,metrics=METRICS;let index=trials[document.getElementById('trial').value].length>400?400:0,playing=true,last=0;const select=document.getElementById('trial'),seek=document.getElementById('seek'),ctx=document.getElementById('c').getContext('2d'),p=document.getElementById('plot').getContext('2d');
select.onchange=()=>{index=trials[select.value].length>400?400:0;playing=true};document.getElementById('play').onclick=()=>playing=!playing;document.getElementById('jump').onclick=()=>{index=450;playing=true};document.getElementById('restart').onclick=()=>{index=0;playing=true};seek.oninput=()=>{index=+seek.value;playing=false};
function draw(now){const rows=trials[select.value];index=Math.min(index,rows.length-1);const s=rows[index],x=500+s[0]*140,y=270;
ctx.clearRect(0,0,1000,360);ctx.strokeStyle='#20906d';ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(500,50);ctx.lineTo(500,330);ctx.stroke();ctx.setLineDash([]);
ctx.strokeStyle='#687780';ctx.beginPath();ctx.moveTo(100,298);ctx.lineTo(900,298);ctx.stroke();ctx.fillStyle='#2768ac';ctx.fillRect(x-40,y-20,80,40);
ctx.fillStyle='#263747';for(const dx of [-25,25]){ctx.beginPath();ctx.arc(x+dx,y+20,10,0,7);ctx.fill()}
ctx.strokeStyle='#ec9f28';ctx.lineWidth=8;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+170*Math.sin(s[2]),y-170*Math.cos(s[2]));ctx.stroke();
if(s[5]){const sign=Math.sign(s[5]);ctx.strokeStyle='#d74338';ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(x-sign*120,230);ctx.lineTo(x-sign*50,230);ctx.lineTo(x-sign*65,220);ctx.moveTo(x-sign*50,230);ctx.lineTo(x-sign*65,240);ctx.stroke()}
document.getElementById('status').textContent=`t=${(index*.02).toFixed(2)}s | x=${s[0].toFixed(3)}m | angle=${(s[2]*180/Math.PI).toFixed(2)} deg | Control=${s[4].toFixed(2)}N | Disturbance=${s[5].toFixed(1)}N`;seek.value=index;seek.max=rows.length-1;
p.clearRect(0,0,1000,220);p.fillStyle='#263747';p.fillText('Position error (+/-0.5 m) / Time 0-30 s',20,20);p.strokeStyle='#ddd';p.beginPath();p.moveTo(40,120);p.lineTo(980,120);p.stroke();p.strokeStyle='#20906d';p.beginPath();rows.forEach((r,i)=>{const xx=40+i/1499*940,yy=120-r[0]*170;i?p.lineTo(xx,yy):p.moveTo(xx,yy)});p.stroke();p.strokeStyle='#d74338';p.beginPath();p.moveTo(40+index/1499*940,30);p.lineTo(40+index/1499*940,210);p.stroke();
document.getElementById('metrics').textContent=JSON.stringify(metrics.find(m=>m.force_n===+select.value),null,2);
if(playing&&now-last>=20){if(index<rows.length-1)index++;else playing=false;last=now}requestAnimationFrame(draw)}requestAnimationFrame(draw);
</script></html>'''.replace('DATA',json.dumps(trials)).replace('METRICS',json.dumps(metrics))
    options=''.join(f'<option value="{force}">External force {force:+g} N</option>' for force in forces)
    start=html.index('<select id="trial">')+len('<select id="trial">')
    end=html.index('</select>',start)
    # 页面内嵌轨迹与指标，可直接离线打开；可见文字保持英文，避免中文字体依赖。
    html=html[:start]+options+html[end:]
    (out/'demo.html').write_text(html,encoding='utf-8')
    print(json.dumps(metrics,indent=2))


def frame(state):
    im=Image.new('RGB',(600,300),'#f3f6fa'); d=ImageDraw.Draw(im)
    x,_,theta,_,action=state
    cx=300+x*90; cy=210
    d.line((50,235,550,235),fill='#687780',width=3)
    d.line((84,225,84,245),fill='#e06a63',width=3); d.line((516,225,516,245),fill='#e06a63',width=3)
    d.rectangle((cx-28,cy-14,cx+28,cy+14),fill='#2768ac')
    d.ellipse((cx-23,cy+8,cx-9,cy+22),fill='#263747'); d.ellipse((cx+9,cy+8,cx+23,cy+22),fill='#263747')
    end=(cx+125*np.sin(theta),cy-125*np.cos(theta))
    d.line((cx,cy,*end),fill='#ec9f28',width=7); d.ellipse((end[0]-8,end[1]-8,end[0]+8,end[1]+8),fill='#de8320')
    d.text((15,15),f'x={x:+.2f} m    angle={np.rad2deg(theta):+.2f} deg    force={action:+.2f} N',fill='#263747')
    return im

def render(out,result,trajectory,random_trajectory):
    # 离散轨迹存的是 0/1 索引，绘图前转成 -10/+10 N；连续轨迹已经是物理力。
    if result.get("action_space") != "continuous":
        trajectory = [row[:4] + [10. if row[4] else -10.] for row in trajectory]
        random_trajectory = [row[:4] + [10. if row[4] else -10.] for row in random_trajectory]
    history=result['history']
    horizon=result.get('max_steps',500)
    baseline_label=result.get('baseline_label','Random')
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    if history:
        axs[0].plot([r['steps'] for r in history],[r['train_length'] for r in history],label='Training (last 50 episodes)')
        validation=[r for r in history if 'validation_length' in r]
        axs[0].plot([r['steps'] for r in validation],[r['validation_length'] for r in validation],label='Validation (10 episodes)')
    axs[0].axhline(result['random']['length'],ls='--',color='gray',label=baseline_label)
    axs[0].set(xlabel='Environment steps',ylabel='Episode length',ylim=(0,horizon*1.04)); axs[0].legend(fontsize=8); axs[0].grid(alpha=.2)
    axs[1].bar([baseline_label,'PPO'],[result['random']['length'],result['trained']['length']],color=['#a5abb3','#2878b5'])
    axs[1].set(ylabel=f"Mean length ({result['trained']['episodes']} held-out episodes)",ylim=(0,horizon*1.1))
    fig.savefig(out/'learning_curve.png',dpi=150); plt.close(fig)
    states=np.array(trajectory); t=np.arange(len(states))*.02
    fig,axs=plt.subplots(3,1,figsize=(10,7),layout='constrained',sharex=True)
    axs[0].plot(t,states[:,0]); axs[0].set(ylabel='Cart position (m)'); axs[0].axhline(2.4,ls='--',color='red'); axs[0].axhline(-2.4,ls='--',color='red')
    axs[1].plot(t,np.rad2deg(states[:,2])); axs[1].set(xlabel='Time (s)',ylabel='Pole angle (deg)'); axs[1].axhline(12,ls='--',color='red'); axs[1].axhline(-12,ls='--',color='red')
    axs[2].plot(t,states[:,4]); axs[2].set(xlabel='Time (s)',ylabel='Force (N)',ylim=(-11,11))
    fig.savefig(out/'trajectory.png',dpi=150); plt.close(fig)
    # 原步长 20 ms，每两步取一帧并显示 40 ms，保持实速播放。
    frames=[frame(s) for s in trajectory[::2]]
    frames[0].save(out/'ppo.gif',save_all=True,append_images=frames[1:],duration=40,loop=0)
    payload=json.dumps({'PPO':trajectory,baseline_label:random_trajectory})
    # 页面内嵌轨迹与指标，可直接离线打开；可见文字保持英文，避免中文字体依赖。
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Cart-Pole PPO Simulation</title>
<style>body{font-family:system-ui;max-width:1000px;margin:30px auto;background:#f3f6fa;color:#263747}canvas{background:white;width:100%;border-radius:12px}button,select{padding:8px;margin:8px}img{width:100%}pre{white-space:pre-wrap}</style>
<h1>Cart-Pole Control with Policy Gradients</h1><p>See the reward configuration below. Each step is 0.02 s, with a HORIZON-step observation limit. The PPO demonstration uses deterministic actions.</p>
<select id="policy"><option>PPO</option><option>BASELINE</option></select><button id="pause">Pause / Play</button><button id="restart">Restart</button><span id="info"></span><canvas id="c" width="1000" height="360"></canvas>
<h2>Training and Held-Out Evaluation</h2><img src="learning_curve.png"><img src="trajectory.png"><pre id="metrics"></pre>
<script>const data=PAYLOAD;const results=RESULTS;document.getElementById('metrics').textContent=JSON.stringify(results,null,2);
const canvas=document.getElementById('c'),ctx=canvas.getContext('2d');let index=0,playing=true,last=0;
const select=document.getElementById('policy');select.onchange=()=>{index=0;playing=true};document.getElementById('pause').onclick=()=>playing=!playing;document.getElementById('restart').onclick=()=>{index=0;playing=true};
function draw(now){const rows=data[select.value],s=rows[index];ctx.clearRect(0,0,1000,360);const x=500+s[0]*140,y=270;
ctx.strokeStyle='#687780';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(100,298);ctx.lineTo(900,298);ctx.stroke();
ctx.fillStyle='#2768ac';ctx.fillRect(x-40,y-20,80,40);ctx.fillStyle='#263747';for(const offset of [-25,25]){ctx.beginPath();ctx.arc(x+offset,y+20,10,0,Math.PI*2);ctx.fill()}
ctx.strokeStyle='#ec9f28';ctx.lineWidth=9;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+170*Math.sin(s[2]),y-170*Math.cos(s[2]));ctx.stroke();
document.getElementById('info').textContent=`${index+1}/${rows.length} steps | ${(index*.02).toFixed(2)} s | angle=${(s[2]*180/Math.PI).toFixed(2)} deg | x=${s[0].toFixed(2)} m | Force=${s[4].toFixed(2)} N`;
if(playing&&now-last>=20){if(index<rows.length-1)index++;else playing=false;last=now}requestAnimationFrame(draw)}requestAnimationFrame(draw);
</script></html>'''.replace('BASELINE',baseline_label).replace('HORIZON',str(horizon)).replace('PAYLOAD',payload).replace('RESULTS',json.dumps({'trained':result['trained'],baseline_label:result['random']},ensure_ascii=False))
    (out/'demo.html').write_text(html,encoding='utf-8')


def arguments(stage):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=200704 if stage==1 else 401408,
                        help='Training budget, rounded up to a batch of 2048')
    parser.add_argument('--episodes',type=int,default=30,help='Held-out evaluation episodes')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'outputs'/Path(__file__).stem)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--initial',type=Path,help='Optional compatible checkpoint for fine-tuning; otherwise train from scratch')
    group.add_argument('--evaluate',type=Path,help='Evaluate this checkpoint without training')
    if stage>=4:
        parser.add_argument('--target',type=float,default=0.)
    if stage==5:
        parser.add_argument('--forces',type=float,nargs='+',default=[0.,1.,-1.,3.,-3.,5.,-5.,8.,-8.,10.,-10.,15.,-15.,20.,-20.])
    args=parser.parse_args()
    if args.steps<=0 or args.episodes<=0: parser.error('--steps and --episodes must be positive')
    if args.seed<0: parser.error('--seed must be non-negative')
    for path in [args.initial,args.evaluate]:
        if path is not None and not path.is_file(): parser.error(f'Checkpoint not found: {path}')
    if stage>=4 and (not np.isfinite(args.target) or abs(args.target)>1.5):
        parser.error('--target must be finite and within [-1.5, 1.5] m')
    if stage==5:
        if args.target != 0: parser.error('The disturbance experiment uses target=0; other targets require separate evaluation')
        if not np.isfinite(args.forces).all(): parser.error('--forces must be finite')
        args.forces=list(dict.fromkeys(args.forces))
    args.output.mkdir(parents=True,exist_ok=True)
    return args


def record_run(args,stage,result):
    result.update(stage=stage,seed=args.seed,requested_steps=args.steps,
                  actual_training_steps=0 if args.evaluate else int(np.ceil(args.steps/2048))*2048,
                  initial_checkpoint=str(args.initial) if args.initial else None,
                  evaluated_checkpoint=str(args.evaluate) if args.evaluate else None,
                  training_mode='evaluation' if args.evaluate else ('fine_tune' if args.initial else 'from_scratch'))
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['history']},indent=2,allow_nan=False))
    print(f"Outputs: {args.output.resolve()}",flush=True)


def main():
    args=arguments(STAGE)
    baseline_net=GaussianNetwork(args.seed)
    if args.initial: baseline_net.load(args.initial)
    if args.evaluate:
        net=GaussianNetwork(args.seed); net.load(args.evaluate)
        selected=args.evaluate
    else:
        net=train(args.seed,args.steps,args.output,args.initial)
        selected=args.output/'policy.npz'
    new_summary,new_trials=evaluate_pushes(net,20000+args.seed,args.episodes,forces=tuple(args.forces))
    old_summary,old_trials=evaluate_pushes(baseline_net,20000+args.seed,args.episodes,forces=tuple(args.forces))
    result=dict(target_m=0.,reward='position_unchanged',trained=new_summary,baseline=old_summary,
                baseline_label='Initial checkpoint' if args.initial else 'Untrained baseline')
    record_run(args,STAGE,result)
    (args.output/'new_trials.json').write_text(json.dumps(new_trials,indent=2),encoding='utf-8')
    (args.output/'baseline_trials.json').write_text(json.dumps(old_trials,indent=2),encoding='utf-8')
    render_push_animation(selected,args.output/'animation',args.forces)


if __name__=='__main__':
    main()
