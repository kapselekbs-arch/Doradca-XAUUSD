import numpy as np, pandas as pd, requests

SW = 5
ROLLOVER_NY = (16 * 60 + 55, 18 * 60 + 5)   # 16:55-18:05 czasu NY (DST liczy się samo)


def atr_f(d, n=14):
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi_f(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    s = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / s)


def zamkniete(m1, regula):
    """Świece danego TF, tylko w pełni zamknięte."""
    d = m1.resample(regula, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    koniec = m1.index[-1] + pd.Timedelta(minutes=1)
    return d[d.index + pd.Timedelta(regula) <= koniec]


def tf_struktura(m1, regula, nazwa, sw=SW):
    d = zamkniete(m1, regula)
    a, c = atr_f(d), d.close
    okno = 2 * sw + 1
    ph = d.high.where(d.high == d.high.rolling(okno, center=True).max())
    pl = d.low.where(d.low == d.low.rolling(okno, center=True).min())
    sh, sl = ph.dropna(), pl.dropna()          # swingi potwierdzone (sw świec później)
    e20, e50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    lvl_h, lvl_l = ph.shift(sw).ffill(), pl.shift(sw).ffill()
    up = (c > lvl_h) & (c.shift() <= lvl_h)
    dn = (c < lvl_l) & (c.shift() >= lvl_l)
    ev = pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=d.index)
    zd = ev[ev != 0]
    bos, wiek, t_ev = "brak zdarzeń", None, None
    if len(zd):
        k = int(zd.iloc[-1])
        choch = len(zd) > 1 and int(zd.iloc[-2]) != k
        bos = ("CHoCH " if choch else "BOS ") + ("↑" if k > 0 else "↓")
        wiek = len(d) - 1 - d.index.get_loc(zd.index[-1])
        t_ev = str(zd.index[-1])
    uklad = "MIESZANY"
    if len(sh) >= 2 and len(sl) >= 2:
        hh, hl = sh.iloc[-1] > sh.iloc[-2], sl.iloc[-1] > sl.iloc[-2]
        uklad = "WZROSTOWY (HH+HL)" if hh and hl else ("SPADKOWY (LH+LL)" if not hh and not hl else "MIESZANY")
    cena, at = c.iloc[-1], a.iloc[-1]
    return dict(tf=nazwa, atr=at, trend=(e20.iloc[-1] - e50.iloc[-1]) / at, rsi=rsi_f(c).iloc[-1],
                uklad=uklad, bos=bos, wiek=wiek, t_ev=t_ev,
                sh=sh.iloc[-1] if len(sh) else np.nan, sl=sl.iloc[-1] if len(sl) else np.nan,
                sh_all=sh, sl_all=sl, cena=cena)


def poziomy(m1, atr15):
    cena = m1.close.iloc[-1]
    ny = m1.index.tz_convert("America/New_York")
    dz = (ny + pd.Timedelta(hours=7)).normalize()
    g = m1.groupby(dz)
    dni = list(g.groups.keys())
    L = {}
    if len(dni) >= 2:
        p = g.get_group(dni[-2])
        L["PDH"], L["PDL"] = p.high.max(), p.low.min()
    dzis = g.get_group(dni[-1])
    L["dzis H"], L["dzis L"] = dzis.high.max(), dzis.low.min()
    utc = m1.index.tz_convert("UTC")
    dzien = utc.normalize()
    az = m1[(utc.hour < 7) & (dzien == dzien[-1])]
    if len(az):
        L["Azja H"], L["Azja L"] = az.high.max(), az.low.min()
    tp = (dzis.high + dzis.low + dzis.close) / 3
    if dzis.volume.sum() > 0:
        L["VWAP"] = (tp * dzis.volume).sum() / dzis.volume.sum()
    tydz = m1[m1.index >= (utc[-1].normalize() - pd.Timedelta(days=utc[-1].weekday()))]
    L["tydzień H"], L["tydzień L"] = tydz.high.max(), tydz.low.min()
    adr = (g.high.max() - g.low.min()).iloc[:-1].tail(14).mean()
    return L, (L["dzis H"] - L["dzis L"]) / adr if adr else np.nan, cena


def plynnosc(sh, sl, cena, atr, tol=0.15, n=8):
    """Równe szczyty/dołki (EQH/EQL) z ostatnich n swingów M15."""
    out = {}
    h = sh.tail(n).values
    for i in range(len(h)):
        for j in range(i):
            if abs(h[i] - h[j]) <= tol * atr and h[i] > cena:
                out["EQH"] = min(out.get("EQH", 1e18), (h[i] + h[j]) / 2)
    l = sl.tail(n).values
    for i in range(len(l)):
        for j in range(i):
            if abs(l[i] - l[j]) <= tol * atr and l[i] < cena:
                out["EQL"] = max(out.get("EQL", -1e18), (l[i] + l[j]) / 2)
    return out


def sesja(m1):
    t = m1.index[-1]
    ny = t.tz_convert("America/New_York")
    min_ny = ny.hour * 60 + ny.minute
    if ROLLOVER_NY[0] <= min_ny < ROLLOVER_NY[1]:
        return "ROLLOVER (spread zwykle rośnie)"
    h = t.tz_convert("UTC").hour
    return ("Azja" if h < 7 else "Londyn" if h < 13 else "Londyn+NY" if h < 16 else "NY" if h < 21 else "poza sesją")


def kalendarz(godz=12):
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                         timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        teraz = pd.Timestamp.now(tz="UTC")
        wyn = []
        for e in r.json():
            if e.get("country") != "USD" or e.get("impact") != "High":
                continue
            t = pd.Timestamp(e["date"]).tz_convert("UTC")
            if teraz - pd.Timedelta(minutes=30) <= t <= teraz + pd.Timedelta(hours=godz):
                wyn.append((t, e.get("title", "")))
        return sorted(wyn), None
    except Exception as ex:
        return None, f"kalendarz niedostępny ({str(ex)[:40]})"


def karta_struktury(m1):
    tfy = [tf_struktura(m1, r, n) for r, n in
           [("1min", "M1"), ("5min", "M5"), ("15min", "M15"), ("1h", "H1")]]
    m15 = tfy[2]
    L, adr_uz, cena = poziomy(m1, m15["atr"])
    L.update(plynnosc(m15["sh_all"], m15["sl_all"], cena, m15["atr"]))
    w = ["--- STRUKTURA (kontekst, NIE sygnał) ---"]
    for t in tfy:
        wk = f" {t['wiek']} św. temu" if t["wiek"] is not None else ""
        w.append(f"{t['tf']}: {t['uklad']} | trend {t['trend']:+.1f} ATR | RSI {t['rsi']:.0f} | "
                 f"{t['bos']}{wk}")
        w.append(f"    swing H {t['sh']:.2f} ({(t['sh'] - cena) / t['atr']:+.1f} ATR) | "
                 f"L {t['sl']:.2f} ({(t['sl'] - cena) / t['atr']:+.1f} ATR)")
    kier = [t["uklad"][:3] for t in tfy]
    w.append("Zgodność TF: " + ("WSZYSTKIE ZGODNE" if len(set(kier)) == 1 else "TF się różnią - słaba czytelność"))
    w.append("")
    w.append(f"--- POZIOMY (odległość w ATR M15 = {m15['atr']:.2f} USD) ---")
    for k, v in sorted(L.items(), key=lambda x: -x[1]):
        w.append(f"{k:10s} {v:9.2f}  ({(v - cena) / m15['atr']:+.1f} ATR)")
    w.append(f"Dzienny zakres wykorzystany: {adr_uz * 100:.0f}% ADR" if adr_uz == adr_uz else "")
    w.append(f"Sesja: {sesja(m1)}")
    ev, blad = kalendarz()
    w.append("")
    if blad:
        w.append("Makro: " + blad)
    elif not ev:
        w.append("Makro: brak ważnych publikacji USD w ciągu 12 h")
    else:
        for t, nazwa in ev:
            min_do = (t - pd.Timestamp.now(tz="UTC")).total_seconds() / 60
            w.append(f"MAKRO USD: {nazwa} za {min_do:.0f} min" + ("  >>> NIE OTWIERAJ POZYCJI" if -15 <= min_do <= 60 else ""))
    return "\n".join(w), dict(tfy=tfy, L=L, cena=cena, atr15=m15["atr"], news=ev)


def alerty(info, poprz):
    """Zwraca (lista alertów, nowy stan). poprz: dict ze stanu poprzedniego przebiegu."""
    msg, st = [], {}
    for t in info["tfy"]:
        if t["tf"] in ("M15", "H1"):
            st["ev_" + t["tf"]] = t["t_ev"]
            if poprz and poprz.get("ev_" + t["tf"]) not in (None, t["t_ev"]):
                msg.append(f"{t['tf']}: nowe {t['bos']}")
    cena, atr = info["cena"], info["atr15"]
    bliskie = sorted([k for k, v in info["L"].items() if k != "VWAP" and abs(v - cena) <= 0.3 * atr])
    st["blisko"] = ",".join(bliskie)
    nowe = set(bliskie) - set((poprz or {}).get("blisko", "").split(","))
    if nowe and poprz is not None:
        msg.append("Cena przy poziomie: " + ", ".join(sorted(nowe)))
    ev = info["news"] or []
    pil = [n for t, n in ev if 0 <= (t - pd.Timestamp.now(tz="UTC")).total_seconds() / 60 <= 60]
    st["news"] = ",".join(pil)
    if pil and set(pil) - set((poprz or {}).get("news", "").split(",")):
        msg.append("Za <60 min publikacja USD: " + ", ".join(pil))
    return msg, st
