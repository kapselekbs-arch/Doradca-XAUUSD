import numpy as np, pandas as pd, joblib
from datetime import datetime, timezone
from lightgbm import LGBMRegressor

SW = 5
SPREAD = 0.15

def atr_f(d, n=14):
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def rsi_f(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    s = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / s)

def cechy_tf(m1, regula, nazwa):
    d = m1.resample(regula, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    a = atr_f(d)
    f = pd.DataFrame(index=d.index)
    e20 = d.close.ewm(span=20, adjust=False).mean()
    e50 = d.close.ewm(span=50, adjust=False).mean()
    f["atr"] = a
    f["trend"] = (e20 - e50) / a
    f["dyst_ema20"] = (d.close - e20) / a
    f["rsi"] = rsi_f(d.close)
    okno = 2 * SW + 1
    ph = d.high.where(d.high == d.high.rolling(okno, center=True).max())
    pl = d.low.where(d.low == d.low.rolling(okno, center=True).min())
    sh = ph.shift(SW).ffill()
    sl = pl.shift(SW).ffill()
    f["dyst_swing_h"] = (sh - d.close) / a
    f["dyst_swing_l"] = (d.close - sl) / a
    f["bos_gora"] = (d.close > sh).astype(float)
    f["bos_dol"] = (d.close < sl).astype(float)
    f["zakres_pozycja"] = (d.close - d.low.rolling(50).min()) / (
        d.high.rolling(50).max() - d.low.rolling(50).min())
    f = f.shift(1)
    f.columns = [f"{nazwa}_{k}" for k in f.columns]
    return f

def ostatnie_zdarzenie(ev, okno):
    n = len(ev)
    ar = np.arange(n)
    idx = pd.Series(np.where(ev != 0, ar, np.nan)).ffill().values
    kier = pd.Series(np.where(ev != 0, ev, np.nan)).ffill().values
    wiek = ar - idx
    return np.where(wiek <= okno, kier, 0.0), wiek

def cechy_smc(m1, regula, nazwa, fvg_min=0.2):
    d = m1.resample(regula, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    a = atr_f(d)
    okno = 2 * SW + 1
    ph = d.high.where(d.high == d.high.rolling(okno, center=True).max())
    pl = d.low.where(d.low == d.low.rolling(okno, center=True).min())
    swH = ph.shift(SW).ffill()
    swL = pl.shift(SW).ffill()
    bear = (d.high > swH) & (d.high.shift() <= swH) & (d.close < swH)
    bull = (d.low < swL) & (d.low.shift() >= swL) & (d.close > swL)
    ev = np.where(bull, 1, np.where(bear, -1, 0))
    sw, sw_wiek = ostatnie_zdarzenie(ev, 10)
    bf = (d.low > d.high.shift(2)) & ((d.low - d.high.shift(2)) > a * fvg_min)
    sf = (d.high < d.low.shift(2)) & ((d.low.shift(2) - d.high) > a * fvg_min)
    ev2 = np.where(bf, 1, np.where(sf, -1, 0))
    fv, fv_wiek = ostatnie_zdarzenie(ev2, 30)
    er = (d.close - d.close.shift(10)).abs() / d.close.diff().abs().rolling(10).sum()
    sma = d.close.rolling(20).mean()
    bb = (d.close - sma) / (2 * d.close.rolling(20).std())
    f = pd.DataFrame({"sweep": sw, "sweep_wiek": np.minimum(sw_wiek, 50),
                      "fvg": fv, "fvg_wiek": np.minimum(fv_wiek, 50),
                      "er": er.values, "bb": bb.values}, index=d.index)
    if regula == "15min":
        f["tydzien_poz"] = ((d.close - d.low.rolling(480).min()) /
                            (d.high.rolling(480).max() - d.low.rolling(480).min())).values
    f = f.shift(1)
    f.columns = [f"{nazwa}_{k}" for k in f.columns]
    return f

def dodatkowe(m1, indeks):
    m1 = m1[["open", "high", "low", "close", "volume"]].astype(float)
    f = pd.DataFrame(index=m1.index)
    for regula, nazwa in [("5min", "M5"), ("15min", "M15")]:
        f = f.join(cechy_smc(m1, regula, nazwa).reindex(m1.index, method="ffill"))
    atr15 = atr_f(m1.resample("15min", label="left", closed="left").agg(
        {"high": "max", "low": "min", "close": "last"}).dropna()).shift(1)
    atr15 = atr15.reindex(m1.index, method="ffill")
    ny = m1.index.tz_convert("America/New_York")
    dz = (ny + pd.Timedelta(hours=7)).normalize()
    tp = (m1.high + m1.low + m1.close) / 3
    pv = (tp * m1.volume).groupby(dz).cumsum()
    vv = m1.volume.groupby(dz).cumsum()
    vwap = pv / vv
    f["vwap_dyst"] = ((m1.close - vwap) / (3 * atr15)).clip(-1, 1)
    utc = m1.index.tz_convert("UTC")
    dzien = pd.Series(utc.date, index=m1.index)
    w_azji = pd.Series(utc.hour < 7, index=m1.index)
    ah = m1.high.where(w_azji).groupby(dzien).cummax().groupby(dzien).ffill()
    al = m1.low.where(w_azji).groupby(dzien).cummin().groupby(dzien).ffill()
    f["azja_h_dyst"] = (ah - m1.close) / atr15
    f["azja_l_dyst"] = (m1.close - al) / atr15
    f["azja_wybicie"] = np.where(~w_azji & (m1.close > ah), 1.0,
                                 np.where(~w_azji & (m1.close < al), -1.0, 0.0))
    g = m1.groupby(dz)
    adr = (g.high.max() - g.low.min()).shift(1).rolling(14).mean().reindex(dz).values
    dzis = m1.high.groupby(dz).cummax() - m1.low.groupby(dz).cummin()
    f["adr_zuzyty"] = dzis / adr
    zn = np.sign(m1.close - m1.open) * m1.volume
    f["delta_15"] = zn.rolling(15).sum() / m1.volume.rolling(15).sum()
    return f.reindex(indeks)

def zakres_przyszly(m1, indeks, minuty=60):
    h = m1.high.astype(float); l = m1.low.astype(float)
    hmax = h[::-1].rolling(minuty).max()[::-1].shift(-1)
    lmin = l[::-1].rolling(minuty).min()[::-1].shift(-1)
    rng = hmax - lmin
    t = pd.Series(m1.index, index=m1.index)
    rng = rng.where((t.shift(-minuty) - t) <= pd.Timedelta(minutes=minuty + 30))
    return rng.reindex(indeks)

def cechy_biezace(m1):
    m1 = m1[["open", "high", "low", "close", "volume"]].astype(float)
    ohlc = m1[["open", "high", "low", "close"]]
    f = ohlc.copy()
    for regula, nazwa in [("5min", "M5"), ("15min", "M15"), ("1h", "H1")]:
        f = f.join(cechy_tf(ohlc, regula, nazwa).reindex(m1.index, method="ffill"))
    f["M1_atr"] = atr_f(ohlc)
    f["M1_zmiennosc_wzgl"] = f.M1_atr / f.M15_atr
    f["M1_rsi"] = rsi_f(m1.close)
    ny = m1.index.tz_convert("America/New_York")
    dz = (ny + pd.Timedelta(hours=7)).normalize()
    g = m1.groupby(dz)
    pdh = g.high.max().shift(1).reindex(dz).values
    pdl = g.low.min().shift(1).reindex(dz).values
    f["dyst_PDH"] = (pdh - m1.close) / f.M15_atr
    f["dyst_PDL"] = (m1.close - pdl) / f.M15_atr
    wa = m1.index.tz_convert("Europe/Warsaw")
    f["godz"] = wa.hour + wa.minute / 60
    f["dzien_tyg"] = wa.dayofweek
    return f.join(dodatkowe(m1, f.index))

def przygotuj(ds):
    X = ds.drop(columns=["open", "high", "low", "close", "R_long", "R_short"],
                errors="ignore").copy()
    for k in [c for c in X.columns if c.endswith("_atr")]:
        X[k] = X[k] / ds.close * 1e4
    return X

def model_vol():
    return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                         min_child_samples=800, subsample=0.7, subsample_freq=1,
                         colsample_bytree=0.7, reg_lambda=10, verbose=-1, n_jobs=-1)

def trenuj(ds2, m1, sciezka):
    zr = zakres_przyszly(m1, ds2.index, 60)
    ok = zr.notna()
    d = ds2[ok]
    y = np.log(zr[ok] / d.M15_atr)
    X = przygotuj(d)
    model = model_vol().fit(X, y)
    joblib.dump({"model": model, "kolumny": list(X.columns),
                 "trening_do": str(d.index[-1])}, sciezka)
    return len(X)

def pobierz_biezace(dni=45):
    import dukascopy_python as dk
    from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD as XAU
    teraz = datetime.now(timezone.utc).replace(tzinfo=None)
    start = teraz - pd.Timedelta(days=dni)
    d = dk.fetch(XAU, dk.INTERVAL_MIN_1, dk.OFFER_SIDE_BID, start, teraz)
    return d

def karta(m1_live, paczka, kapital=1000.0, ryzyko_proc=1.0, sl_usd=3.0):
    F = cechy_biezace(m1_live)
    X = przygotuj(F).reindex(columns=paczka["kolumny"])
    ok = X.notna().all(axis=1)
    if ok.sum() < 500:
        return "Za mało danych do obliczeń (potrzeba ok. 45 dni M1)."
    X = X[ok]
    pred = pd.Series(paczka["model"].predict(X), index=X.index)
    zakres = np.exp(pred) * F.M15_atr.reindex(X.index)
    ost = X.index[-1]
    ostatnia = m1_live.index[-1]
    teraz = pd.Timestamp.now(tz="UTC")
    wiek_min = (teraz - ostatnia).total_seconds() / 60
    hist = zakres[zakres.index >= ost - pd.Timedelta(days=20)].iloc[::5]
    pct = (hist < zakres.iloc[-1]).mean() * 100
    dni_okna = max((hist.index[-1] - hist.index[0]).days, 1)
    z = zakres.iloc[-1]
    stan = "CICHO" if pct < 25 else ("GŁOŚNO" if pct > 75 else "NORMALNIE")
    cena = m1_live.close.iloc[-1]
    atr5 = F.M5_atr.reindex(X.index).iloc[-1]
    atr15 = F.M15_atr.reindex(X.index).iloc[-1]
    ryzyko_usd = kapital * ryzyko_proc / 100
    loty = ryzyko_usd / (sl_usd * 100)
    w = []
    w.append("=" * 44)
    w.append(" KARTA DORADCY ZMIENNOŚCI: XAUUSD")
    w.append("=" * 44)
    w.append(f"Ostatnia świeca: {ostatnia.tz_convert('Europe/Warsaw'):%Y-%m-%d %H:%M} (czas PL)")
    if wiek_min > 30:
        w.append(f"UWAGA: dane sprzed {wiek_min/60:.1f} h (rynek zamknięty lub opóźnienie).")
    w.append(f"Cena (bid): {cena:.2f}")
    w.append("")
    w.append(f"Prognoza zakresu ceny na 60 min: {z:.1f} USD")
    w.append(f"Stan zmienności: {stan} ({pct:.0f}. percentyl z {dni_okna} dni)")
    w.append(f"Spread {SPREAD:.2f} USD to {SPREAD / z * 100:.1f}% prognozowanego zakresu")
    w.append(f"ATR: M5 {atr5:.2f} | M15 {atr15:.2f} USD")
    w.append("")
    w.append("--- Rozmiar pozycji (1 lot = 100 oz, sprawdź u brokera) ---")
    w.append(f"Kapitał {kapital:.0f}, ryzyko {ryzyko_proc}% = {ryzyko_usd:.2f} USD")
    w.append(f"SL {sl_usd:.2f} USD -> pozycja {loty:.3f} lota")
    w.append(f"SL to {sl_usd / z * 100:.0f}% prognozowanego zakresu 60 min; spread to {SPREAD / sl_usd:.2f} R")
    if sl_usd < 0.25 * z:
        w.append("Uwaga: SL bardzo ciasny wobec zmienności, duże ryzyko wybicia szumem.")
    elif sl_usd > z:
        w.append("Uwaga: SL szerszy niż cały prognozowany zakres godziny.")
    w.append("")
    w.append("Kierunek: BRAK zwalidowanej przewagi (testy 2005-2026).")
    w.append("Zdecyduj sam, ten doradca ocenia warunki i ryzyko.")
    return "\n".join(w)
