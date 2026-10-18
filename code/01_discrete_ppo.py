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

STAGE = 1

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

def evaluate(net, seed, episodes=30, random=False, record=False):
    env = CartPole(seed)
    rng = np.random.default_rng(seed+1)
    results, trajectory = [], []
    for episode in range(episodes):
        angles, positions, actions = [], [], []
        while True:
            state = env.state[0].copy()
            action = int(rng.integers(2)) if random else int(net.predict(env.state)[0][0].argmax())
            if record and episode == 0:
                trajectory.append([*state.tolist(), action])
            angles.append(state[2]); positions.append(state[0]); actions.append(action)
            _,_,term,trunc = env.step([action])
            if term[0] or trunc[0]:
                results.append({'length':int(env.age[0]),'success':bool(trunc[0]),
                                'angle_rms_deg':float(np.rad2deg(np.sqrt(np.mean(np.square(angles))))),
                                'max_abs_position':float(np.max(np.abs(positions))),
                                'switches':int(np.count_nonzero(np.diff(actions)))})
                env.reset(np.array([True])); break
    summary = {k:float(np.mean([r[k] for r in results])) for k in results[0]}
    summary['episodes'] = episodes
    summary['length_std'] = float(np.std([r['length'] for r in results]))
    return summary, trajectory

def train(seed, steps, out, initial=None):
    net = Network(seed)
    if initial is not None: net.load(initial)
    rng = np.random.default_rng(seed+20)
    env = CartPole(seed,8)
    history, lengths = [], []
    best = -1
    for update in range(max(1,int(np.ceil(steps/2048)))):
        states=[]; actions=[]; logs=[]; rewards=[]; values=[]; nextvalues=[]; terms=[]; truncs=[]
        # 每批 8 环境 x 256 步=2048 样本，先完成采样，再固定旧概率/优势做优化。
        for t in range(256):
            s = env.state.copy(); p,v = net.predict(s)
            a = (rng.random(8)>p[:,0]).astype(int)
            ns,r,term,trunc = env.step(a)
            # 必须在 reset 前求下一状态价值，尤其不能用新回合初态替代截断最终观测。
            nv = net.predict(ns)[1]  # final observation BEFORE resetting
            states.append(s); actions.append(a); logs.append(np.log(p[np.arange(8),a]+1e-12))
            rewards.append(r); values.append(v); nextvalues.append(nv); terms.append(term); truncs.append(trunc)
            lengths.extend(env.age[term|trunc].tolist()); env.reset(term|trunc)
        adv,targets = gae(*map(np.array,[rewards,values,nextvalues,terms,truncs]))
        s=np.array(states).reshape(-1,4); a=np.array(actions).ravel(); oldlog=np.array(logs).ravel()
        # 只标准化 Actor 优势；价值目标保留原回报尺度，不能用标准化优势重算。
        adv=adv.ravel(); adv=(adv-adv.mean())/(adv.std()+1e-8); targets=targets.ravel()
        metrics=[]
        # 每批数据更新 4 轮，每轮打乱并拆成 64 样本小批量；随后重新采样。
        for epoch in range(4):
            order=rng.permutation(2048)
            for start in range(0,2048,64):
                ix=order[start:start+64]
                metrics.append(net.update(s[ix],a[ix],oldlog[ix],adv[ix],targets[ix],3e-4))
        row={'steps':(update+1)*2048,'train_length':float(np.mean(lengths[-50:])) if lengths else 0}
        row.update(zip(['entropy','value_mse','approx_kl','clip_fraction','gradient_norm'],np.mean(metrics,axis=0).tolist()))
        if update%5 == 0 or update == int(np.ceil(steps/2048))-1:
            score,_=evaluate(net,10000+seed,10)
            row['validation_length']=score['length']
            if score['length']>=best:
                best=score['length']; net.save(out/'policy.npz')
            print(f"seed={seed} steps={row['steps']} training={row['train_length']:.1f} validation={score['length']:.1f}",flush=True)
        history.append(row)
    # 分别保存最终更新模型和验证最优模型；返回/评估采用验证最优，测试集不参与选择。
    net.save(out/'policy_final.npz'); net.load(out/'policy.npz')
    return net,history

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
    if args.evaluate:
        net=Network(args.seed); net.load(args.evaluate); history=[]
    else: net,history=train(args.seed,args.steps,args.output,args.initial)
    learned,trajectory=evaluate(net,20000+args.seed,args.episodes,record=True)
    baseline,random_trajectory=evaluate(net,20000+args.seed,args.episodes,random=True,record=True)
    result=dict(action_space='discrete',max_steps=500,history=history,trained=learned,random=baseline)
    record_run(args,STAGE,result)
    (args.output/'trajectory.json').write_text(json.dumps(trajectory),encoding='utf-8')
    render(args.output,result,trajectory,random_trajectory)


if __name__=='__main__':
    main()
