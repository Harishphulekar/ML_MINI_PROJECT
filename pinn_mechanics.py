"""PINN for mechanics: forward solve + inverse discovery on a damped oscillator.
   m x'' + c x' + k x = 0,  x(0)=1, x'(0)=0.   Run: python pinn_mechanics.py
"""
import json, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
M, C_TRUE, K_TRUE, T = 1.0, 0.4, 4.0, 10.0

def analytic(t, c=C_TRUE, k=K_TRUE, m=M):
    wd = np.sqrt(k/m - (c/(2*m))**2); a = c/(2*m)
    return np.exp(-a*t) * (np.cos(wd*t) + a/wd*np.sin(wd*t))

class Net(nn.Module):
    def __init__(s, w=64, d=4):
        super().__init__()
        L, i = [], 1
        for _ in range(d): L += [nn.Linear(i, w), nn.Tanh()]; i = w
        s.f = nn.Sequential(*L, nn.Linear(i, 1))
    def forward(s, t): return s.f(t / T)
def derivs(net, t):
    t = t.clone().requires_grad_(True); x = net(t)
    x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, torch.ones_like(x_t), create_graph=True)[0]
    return x, x_t, x_tt

def train(inverse, iters=6000, noise=0.02):
    net = Net()
    c = nn.Parameter(torch.tensor(1.0)); k = nn.Parameter(torch.tensor(1.0))  # bad init
    params = list(net.parameters()) + ([c, k] if inverse else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    tc = torch.rand(300, 1) * T                      # collocation points
    t0 = torch.zeros(1, 1)
    if inverse:                                      # noisy sensor data
        td = torch.linspace(0, T, 40).reshape(-1, 1)
        xd = torch.tensor(analytic(td.numpy()), dtype=torch.float32) + noise * torch.randn(40, 1)
    hist = []
    for it in range(iters):
        opt.zero_grad()
        _, _, x_tt = derivs(net, tc); x, x_t, _ = derivs(net, tc)
        cc, kk = (c, k) if inverse else (C_TRUE, K_TRUE)
        l_phys = ((M * x_tt + cc * x_t + kk * x) ** 2).mean()
        x0, v0, _ = derivs(net, t0)
        l_ic = (x0 - 1) ** 2 + v0 ** 2
        l_data = ((net(td) - xd) ** 2).mean() if inverse else torch.tensor(0.)
        loss = l_phys + 10 * l_ic.squeeze() + 10 * l_data
        loss.backward(); opt.step()
        if it % 500 == 0 or it == iters - 1:
            hist.append((it, loss.item(), c.item(), k.item()))
    return net, c.item(), k.item(), hist, (td, xd) if inverse else None

def evaluate(net):
    t = torch.linspace(0, T, 500).reshape(-1, 1)
    p = net(t).detach().numpy().ravel(); a = analytic(t.numpy().ravel())
    return t.numpy().ravel(), p, a, float(np.linalg.norm(p - a) / np.linalg.norm(a))

if __name__ == "__main__":
    res = {}
    net, *_ = train(False); t, p, a, err = evaluate(net); res["forward_rel_L2"] = err
    net2, c, k, hist, (td, xd) = train(True); t2, p2, a2, err2 = evaluate(net2)
    res.update(inverse_rel_L2=err2, c_est=c, k_est=k, c_true=C_TRUE, k_true=K_TRUE,
               c_err_pct=abs(c - C_TRUE) / C_TRUE * 100, k_err_pct=abs(k - K_TRUE) / K_TRUE * 100)
    json.dump(res, open("results.json", "w"), indent=2); 
    for key, val in res.items():
        print(f"{key}: {val}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(t, a, "k-", label="analytic"); ax[0].plot(t, p, "r--", label="PINN"); ax[0].set_title(f"Forward (rel L2={err:.2e})"); ax[0].legend()
    ax[1].plot(t2, a2, "k-", label="analytic"); ax[1].plot(t2, p2, "r--", label="PINN"); ax[1].scatter(td, xd, s=10, label="noisy data"); ax[1].set_title("Inverse fit"); ax[1].legend()
    h = np.array(hist); ax[2].plot(h[:, 0], h[:, 2], label="c"); ax[2].plot(h[:, 0], h[:, 3], label="k")
    ax[2].axhline(C_TRUE, ls=":", c="gray"); ax[2].axhline(K_TRUE, ls=":", c="gray"); ax[2].set_title("Parameter discovery"); ax[2].legend()
    for x in ax: x.set_xlabel("t / iteration")
    plt.tight_layout(); plt.savefig("results.png", dpi=150)
