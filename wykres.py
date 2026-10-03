import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import struktura as st

TF = {"1": ("1min", "M1"), "5": ("5min", "M5"), "15": ("15min", "M15"), "60": ("1h", "H1")}
KOLORY = {"PDH": "#e67e22", "PDL": "#e67e22", "dzis H": "#7f8c8d", "dzis L": "#7f8c8d",
          "Azja H": "#9b59b6", "Azja L": "#9b59b6", "VWAP": "#3498db",
          "tydzień H": "#95a5a6", "tydzień L": "#95a5a6", "EQH": "#e74c3c", "EQL": "#2ecc71"}


def wykres(m1, info, tf="5", n=120, sciezka="cache/wykres.png"):
    regula, nazwa = TF.get(str(tf), TF["5"])
    d = st.zamkniete(m1, regula).tail(n)
    x = np.arange(len(d))
    cena = m1.close.iloc[-1]
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=110)
    bg, fg = "#131722", "#d1d4dc"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for i, (_, r) in enumerate(d.iterrows()):
        kol = "#26a69a" if r.close >= r.open else "#ef5350"
        ax.plot([i, i], [r.low, r.high], color=kol, lw=0.8, zorder=2)
        ax.add_patch(plt.Rectangle((i - 0.35, min(r.open, r.close)), 0.7,
                                   max(abs(r.close - r.open), 1e-6), color=kol, zorder=3))
    lo, hi = d.low.min(), d.high.max()
    pad = (hi - lo) * 0.08
    ax.set_xlim(-1, len(d) + 14)
    ax.set_ylim(lo - pad, hi + pad)
    # poziomy (tylko w zasięgu wykresu), etykiety rozsunięte, żeby się nie nakładały
    poz = sorted([(v, k) for k, v in info["L"].items() if lo - pad <= v <= hi + pad])
    gap = (hi - lo) * 0.035
    ypos = []
    for v, k in poz:
        y = v if not ypos else max(v, ypos[-1] + gap)
        ypos.append(y)
    for (v, k), y in zip(poz, ypos):
        kol = KOLORY.get(k, "#bdc3c7")
        ax.axhline(v, color=kol, lw=0.9, ls="--" if k == "VWAP" else ":", alpha=0.85, zorder=1)
        ax.text(len(d) + 0.5, y, f"{k} {v:.1f}", color=kol, fontsize=8, va="center")
    # swingi i ostatnie zdarzenie struktury danego TF
    t = next((t for t in info["tfy"] if t["tf"] == nazwa), None)
    if t is not None:
        for s, mark, kol, kol_y in [(t["sh_all"], "v", "#f1c40f", 1), (t["sl_all"], "^", "#f1c40f", -1)]:
            s = s[s.index.isin(d.index)]
            for ts, v in s.items():
                ax.scatter(d.index.get_loc(ts), v + kol_y * (hi - lo) * 0.012, marker=mark,
                           s=22, color=kol, zorder=4)
        if t["t_ev"] is not None:
            ts = pd.Timestamp(t["t_ev"])
            if ts in d.index:
                i = d.index.get_loc(ts)
                ax.axvline(i, color="#f39c12", lw=0.8, ls="-.", alpha=0.8)
                ax.text(i, hi + pad * 0.5, t["bos"], color="#f39c12", fontsize=9, ha="center")
    ax.axhline(cena, color="#ffffff", lw=0.8, alpha=0.7)
    ax.text(len(d) + 0.5, cena, f"{cena:.2f}", color="#000", fontsize=8, va="center",
            bbox=dict(facecolor="#ffffff", edgecolor="none", pad=1.5))
    krok = max(len(d) // 8, 1)
    ax.set_xticks(x[::krok])
    ax.set_xticklabels([ts.tz_convert("Europe/Warsaw").strftime("%d.%m %H:%M") for ts in d.index[::krok]],
                       color=fg, fontsize=8)
    ax.tick_params(axis="y", colors=fg, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#363a45")
    ax.grid(color="#363a45", lw=0.4, alpha=0.5)
    uk = t["uklad"] if t else ""
    ax.set_title(f"XAUUSD {nazwa} | {uk} | czas PL | kontekst, NIE sygnał", color=fg, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(sciezka, facecolor=bg)
    plt.close(fig)
    return sciezka
