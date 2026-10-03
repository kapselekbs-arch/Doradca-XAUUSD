import os, sys, re, json, html, requests, joblib
import numpy as np, pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_core as ac, struktura as st, wykres as wk

TOKEN = os.environ["TELEGRAM_TOKEN"].strip()
CHAT = os.environ["TELEGRAM_CHAT_ID"].strip()

# ---- USTAWIENIA ----
KAPITAL, RYZYKO_PROC, SL_USD = 1000, 1.0, 3.0
GODZ_ALERTOW_PL = (8, 22)      # alerty tylko w tych godzinach (czas PL); /karta działa zawsze
CACHE = "cache"
M1P, STP = f"{CACHE}/m1.pkl", f"{CACHE}/stan.json"


def api(metoda, **kw):
    try:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/{metoda}", timeout=25, **kw)
        if not r.ok:
            print("Telegram błąd:", r.status_code, r.text[:100])
        return r
    except Exception as e:
        print("Telegram wyjątek:", type(e).__name__)   # bez treści (mogłaby zawierać adres z tokenem)
        return None


def wyslij(tekst):
    for i in range(0, len(tekst), 3800):
        api("sendMessage", data={"chat_id": CHAT, "parse_mode": "HTML",
                                 "text": "<pre>" + html.escape(tekst[i:i + 3800]) + "</pre>"})


def parsuj(tekst):
    """Polecenie przekazane z Cloudflare: /karta, /wykres [1|5|15|60]."""
    t = tekst.strip().lower().split()
    if t and t[0].startswith("/karta"):
        return True, []
    if t and t[0].startswith("/wykres"):
        return False, [t[1] if len(t) > 1 and t[1] in wk.TF else "5"]
    return False, []


def komendy(stan):
    """Czyta polecenia z Telegrama: /karta, /wykres [1|5|15|60]."""
    r = api("getUpdates", data={"offset": stan.get("offset", 0), "timeout": 0})
    karta, wykresy = False, []
    if r is not None and r.ok:
        for u in r.json().get("result", []):
            stan["offset"] = u["update_id"] + 1
            m = u.get("message") or {}
            if str(m.get("chat", {}).get("id")) != CHAT:
                continue
            t = (m.get("text") or "").strip().lower().split()
            if t and t[0].startswith("/karta"):
                karta = True
            elif t and t[0].startswith("/wykres"):
                wykresy.append(t[1] if len(t) > 1 and t[1] in wk.TF else "5")
    return karta, wykresy


def wyslij_wykres(m1, info, tf="5", podpis=""):
    try:
        sciezka = wk.wykres(m1, info, tf, sciezka=f"{CACHE}/wykres.png")
        with open(sciezka, "rb") as f:
            api("sendPhoto", data={"chat_id": CHAT, "caption": podpis[:900]}, files={"photo": f})
    except Exception as e:
        print("Wykres błąd:", type(e).__name__, str(e)[:120])


def dane_m1():
    import dukascopy_python as dk
    from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD as XAU
    os.makedirs(CACHE, exist_ok=True)
    teraz = datetime.now(timezone.utc).replace(tzinfo=None)
    m1 = pd.read_pickle(M1P) if os.path.exists(M1P) else None
    start = (m1.index[-1].tz_convert("UTC").tz_localize(None) - pd.Timedelta(minutes=10)
             if m1 is not None else teraz - pd.Timedelta(days=45))
    nowe = dk.fetch(XAU, dk.INTERVAL_MIN_1, dk.OFFER_SIDE_BID, start, teraz)
    if nowe is not None and len(nowe):
        m1 = nowe if m1 is None else pd.concat([m1, nowe])
    if m1 is None or not len(m1):
        raise RuntimeError("brak danych M1")
    m1 = m1[~m1.index.duplicated(keep="last")].sort_index()
    m1 = m1[m1.index >= m1.index[-1] - pd.Timedelta(days=45)]
    m1.to_pickle(M1P)
    return m1


def main():
    os.makedirs(CACHE, exist_ok=True)
    stan = json.load(open(STP)) if os.path.exists(STP) else {}
    kom = os.environ.get("KOMENDA", "").strip()
    if kom:                                  # polecenie z Cloudflare (natychmiast)
        zadanie, wykresy = parsuj(kom)
    elif os.environ.get("TRYB_WEBHOOK") == "1":
        zadanie, wykresy = False, []         # webhook aktywny: getUpdates nie działa
    else:
        zadanie, wykresy = komendy(stan)
    try:
        m1 = dane_m1()
        paczka = joblib.load("model_zmiennosc.joblib")
        wiek = (pd.Timestamp.now(tz="UTC") - m1.index[-1]).total_seconds() / 60
        if wiek > 30 and not (zadanie or wykresy):
            print(f"Dane sprzed {wiek / 60:.1f} h - rynek zamknięty, pomijam.")
            return
        k_vol = ac.karta(m1, paczka, KAPITAL, RYZYKO_PROC, SL_USD)
        k_str, info = st.karta_struktury(m1)
        pelna = k_vol + "\n\n" + k_str
        for tf in wykresy:
            wyslij_wykres(m1, info, tf, f"XAUUSD, wykres M{tf if tf != '60' else '60 (H1)'}")
        if zadanie:
            wyslij(pelna)
            wyslij_wykres(m1, info, "5", "XAUUSD M5 - poziomy i struktura")
        elif wykresy:
            pass
        else:
            vol = (re.search(r"Stan zmienności: (\w+)", k_vol) or [None, None])[1]
            lista, nowy = st.alerty(info, stan.get("alerty"))
            zakres = (re.search(r"60 min: ([\d.]+) USD", k_vol) or [None, "?"])[1]
            godz = pd.Timestamp.now(tz="Europe/Warsaw").hour
            zmiana = vol != stan.get("vol")
            if GODZ_ALERTOW_PL[0] <= godz < GODZ_ALERTOW_PL[1]:
                if zmiana:
                    wyslij(("START: " if not stan.get("vol") else "ZMIANA ZMIENNOŚCI: ") + str(vol) + "\n\n" + pelna)
                elif lista:
                    wyslij("\n".join(lista) + f"\n\nCena {info['cena']:.2f} | zmienność {vol} | "
                                              f"zakres 60 min ~{zakres} USD")
                    wyslij_wykres(m1, info, "5", "XAUUSD M5 - poziomy i struktura")
                stan["vol"], stan["alerty"] = vol, nowy
            else:
                stan["alerty"] = nowy
    except Exception as e:
        print("BŁĄD:", type(e).__name__, str(e)[:200])
        raise
    finally:
        json.dump(stan, open(STP, "w"))


if __name__ == "__main__":
    main()
