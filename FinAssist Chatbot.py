from __future__ import annotations

import os
import re
from typing import List, Dict, Optional, TypedDict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import gradio as gr
from dotenv import load_dotenv

# LangChain / LangGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from pathlib import Path as _Path
from PIL import Image

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY not set. On Windows PowerShell: setx GOOGLE_API_KEY \"<YOUR_KEY>\" "
        "then open a new terminal."
    )

SEED_DOCS: List[Dict[str, str]] = [
    {
        "id": "faq_001",
        "title": "Bütçe Planlaması Nedir?",
        "text": (
            "Bütçe planlaması, gelir ve giderlerinizi belirli bir dönemde planlayıp izleyerek "
            "tasarruf ve harcamalarınızı dengeleme sürecidir. Temel adımlar: hedef belirleme, "
            "gelir/ gider kalemlerini listeleme, tasarruf payı ayırma ve düzenli izleme."
        ),
        "lang": "tr",
    },
    {
        "id": "faq_002",
        "title": "Kredi Kartı Faizi Nasıl Hesaplanır?",
        "text": (
            "Kredi kartı faizinde dönem borcu ödenmezse, kalan tutara günlük bazda faiz işler. "
            "Basit yaklaşım: aylık faiz oranını 30'a bölerek günlük oran elde edilir ve gün sayısı ile çarpılır. "
            "Gerçek hesaplama bankaya göre değişir; ek ücretler ve gecikme faizi olabilir."
        ),
        "lang": "tr",
    },
    {
        "id": "faq_003",
        "title": "ETF vs Mutual Fund",
        "text": (
            "An ETF (exchange-traded fund) is a basket of securities that trades on an exchange like a stock, "
            "typically with intraday liquidity and generally lower expense ratios. Mutual funds price once per day "
            "at NAV and may have higher fees. Tax treatment and minimums differ by jurisdiction."
        ),
        "lang": "en",
    },
    {
        "id": "faq_004",
        "title": "Acil Durum Fonu",
        "text": (
            "Acil durum fonu genellikle 3-6 aylık zorunlu giderleri kapsayacak büyüklükte olmalıdır. "
            "Likiditesi yüksek, düşük riskli enstrümanlarda tutulması önerilir."
        ),
        "lang": "tr",
    },
    {
        "id": "faq_005",
        "title": "Risk ve Getiri",
        "text": (
            "Genel ilke: Beklenen getiri arttıkça risk de artar. Çeşitlendirme (diversification) tekil riskleri "
            "azaltabilir ancak sistematik riski tamamen ortadan kaldırmaz."
        ),
        "lang": "tr",
    },
]


EMBEDDING_MODEL_NAME = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
_embedder: Optional[SentenceTransformer] = None
_index: Optional[faiss.IndexFlatIP] = None
_id_to_doc: Dict[int, Dict[str, str]] = {}

def build_vectorstore(docs: List[Dict[str, str]]) -> None:
    global _embedder, _index, _id_to_doc
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [d["text"] for d in docs]
    vectors = _embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = vectors.shape[1]
    _index = faiss.IndexFlatIP(dim)  # cosine similarity (embeddings normalized)
    _index.add(vectors)
    _id_to_doc = {i: docs[i] for i in range(len(docs))}

build_vectorstore(SEED_DOCS)


class CalcResult(TypedDict):
    title: str
    details: str

def to_percent(p: float) -> str:
    return f"{p*100:.4g}%"

def simple_interest(principal: float, annual_rate: float, years: float) -> CalcResult:
    interest = principal * annual_rate * years
    total = principal + interest
    details = (
        f"Basit Faiz\n"
        f"Anapara: {principal:,.2f}\nYıllık Oran: {to_percent(annual_rate)}\nSüre: {years} yıl\n"
        f"Toplam Faiz: {interest:,.2f}\nVade Sonu Tutar: {total:,.2f}"
    )
    return {"title": "Basit Faiz Hesabı", "details": details}

def compound_interest(principal: float, annual_rate: float, years: float, compounds_per_year: int = 12) -> CalcResult:
    A = principal * (1.0 + annual_rate / compounds_per_year) ** (compounds_per_year * years)
    interest = A - principal
    details = (
        f"Bileşik Faiz\n"
        f"Anapara: {principal:,.2f}\nYıllık Oran: {to_percent(annual_rate)}\nSüre: {years} yıl\n"
        f"Bileşikleme: {compounds_per_year}/yıl\nToplam Faiz: {interest:,.2f}\nVade Sonu Tutar: {A:,.2f}"
    )
    return {"title": "Bileşik Faiz Hesabı", "details": details}

def annuity_payment(principal: float, annual_rate: float, years: float, payments_per_year: int = 12) -> CalcResult:
    r = annual_rate / payments_per_year
    n = int(round(years * payments_per_year))
    if r == 0:
        pmt = principal / max(n, 1)
    else:
        pmt = (r * principal) / (1 - (1 + r) ** (-n))
    details = (
        f"Kredi/Anüite Taksit Hesabı\n"
        f"Kredi Tutarı: {principal:,.2f}\nYıllık Oran: {to_percent(annual_rate)}\nVade: {years} yıl\n"
        f"Ödeme Sıklığı: {payments_per_year}/yıl\nAylık/Taksit: {pmt:,.2f}"
    )
    return {"title": "Aylık Taksit (Anüite)", "details": details}

def credit_card_daily_interest(balance: float, monthly_rate: float, days: int) -> CalcResult:
    daily_rate = monthly_rate / 30.0
    interest = balance * daily_rate * days
    total = balance + interest
    details = (
        f"Kredi Kartı Faiz Yaklaşımı\n"
        f"Bakiye: {balance:,.2f}\nAylık Oran: {to_percent(monthly_rate)}\nGün: {days}\n"
        f"Yaklaşık Faiz: {interest:,.2f}\nYeni Bakiye (yaklaşık): {total:,.2f}\n"
        f"Not: Bankaya göre gerçek hesaplama ve ek ücretler değişebilir."
    )
    return {"title": "Kredi Kartı Faiz Yaklaşımı", "details": details}

# Intent detection (very simple)
CALC_KEYWORDS = {
    "compound": ["compound", "bileşik"],
    "simple": ["simple", "basit"],
    "annuity": ["annuity", "taksit", "kredi", "mortgage"],
    "cc_interest": ["credit card", "kredi kart", "kart faizi"],
}

def detect_calc_intent(text: str) -> Optional[str]:
    t = text.lower()
    for intent, keys in CALC_KEYWORDS.items():
        if any(k in t for k in keys):
            return intent
    if ("faiz" in t or "interest" in t) and re.search(r"[0-9]", t):
        return "compound"  # default
    return None


NUM = r"(?:(?:\d+[\.,]?\d*)|(?:\d*[\.,]?\d+))"

def parse_money(text: str) -> Optional[float]:
    m = re.search(rf"({NUM})\s*(?:tl|try|₺|usd|eur|dolar|euro)?", text.lower())
    if not m:
        return None
    return float(m.group(1).replace(",", "."))

def parse_rate(text: str) -> Optional[float]:
    t = text.lower()
    m = re.search(rf"({NUM})\s*%", t)
    if m:
        return float(m.group(1).replace(",", ".")) / 100.0
    m2 = re.search(rf"oran\s*({NUM})", t)
    if m2:
        val = float(m2.group(1).replace(",", "."))
        return val if val < 1 else val / 100.0
    return None

def parse_years(text: str) -> Optional[float]:
    t = text.lower()
    m = re.search(rf"({NUM})\s*(yıl|yr|year|yil)", t)
    if m:
        return float(m.group(1).replace(",", "."))
    m2 = re.search(rf"({NUM})\s*(ay|month)", t)
    if m2:
        return float(m2.group(1).replace(",", ".")) / 12.0
    return None

def parse_days(text: str) -> Optional[int]:
    m = re.search(rf"({NUM})\s*(gün|gun|day)", text.lower())
    if m:
        return int(float(m.group(1).replace(",", ".")))
    return None


def retrieve_context(query: str, k: int = 4) -> List[Dict[str, str]]:
    if _embedder is None or _index is None:
        return []
    qv = _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = _index.search(qv, k)
    results = []
    for idx in I[0]:
        if idx == -1:
            continue
        results.append(_id_to_doc[int(idx)])
    return results


class GraphState(TypedDict):
    messages: List[BaseMessage]
    context: str
    calc_result: Optional[str]

SAFETY_SYSTEM_PROMPT = (
    "You are FinAssist, a bilingual (Turkish + English) financial education assistant.\n"
    "- Provide clear, practical explanations.\n"
    "- Be concise, structure answers with bullets/tables where helpful.\n"
    "- Use TRY amounts when user is in Türkiye unless otherwise specified.\n"
    "- IMPORTANT: Do NOT provide personalized investment advice.\n"
    "  You may explain concepts, formulas, and general considerations.\n"
    "- If the user asks for specific buy/sell/hold recommendations, include a gentle\n"
    "  disclaimer and suggest consulting a licensed advisor.\n"
)

def calc_agent(state: GraphState) -> GraphState:
    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user:
        return state
    text = last_user.content
    intent = detect_calc_intent(text)
    if not intent:
        return state

    principal = parse_money(text)
    rate = parse_rate(text)
    years = parse_years(text)

    result: Optional[CalcResult] = None
    try:
        if intent == "compound" and all(v is not None for v in [principal, rate, years]):
            result = compound_interest(principal, rate, years, compounds_per_year=12)
        elif intent == "simple" and all(v is not None for v in [principal, rate, years]):
            result = simple_interest(principal, rate, years)
        elif intent == "annuity" and all(v is not None for v in [principal, rate, years]):
            result = annuity_payment(principal, rate, years, payments_per_year=12)
        elif intent == "cc_interest":
            balance = principal
            monthly_rate = rate if (rate is not None) else None
            days = parse_days(text) or 30
            if balance is not None and monthly_rate is not None:
                result = credit_card_daily_interest(balance, monthly_rate, days)
    except Exception as e:
        result = {"title": "Hesaplama Hatası", "details": f"{e}"}

    if result is None:
        state["calc_result"] = (
            "Hesaplama için eksik bilgi var. Lütfen belirtin:\n"
            "- Tutar (ör. 10.000 TL)\n- Oran (ör. %24 yıllık)\n- Süre (ör. 3 yıl / 36 ay)\n"
        )
        return state

    state["calc_result"] = f"{result['title']}\n{result['details']}"
    return state

def retrieve_agent(state: GraphState) -> GraphState:
    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user:
        return state
    hits = retrieve_context(last_user.content, k=4)
    context_blocks = [f"[{h['title']}]\n{h['text']}" for h in hits]
    state["context"] = "\n\n".join(context_blocks)
    return state

# Build the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=GOOGLE_API_KEY,
    temperature=0.4,
)

def answer_agent(state: GraphState) -> GraphState:
    msgs: List[BaseMessage] = [SystemMessage(content=SAFETY_SYSTEM_PROMPT)]
    if state.get("context"):
        msgs.append(SystemMessage(content=f"Kısa bağlamsal bilgiler (RAG):\n{state['context']}"))
    if state.get("calc_result"):
        msgs.append(SystemMessage(content=f"Hesaplama çıktısı (varsa kullanıcıya özetle):\n{state['calc_result']}"))
    msgs.extend(state["messages"])
    msgs.append(SystemMessage(content="Cevabın sonunda şu uyarıyı ekle: 'Bilgilendirme amaçlıdır, yatırım tavsiyesi değildir.'"))
    response = llm.invoke(msgs)
    state["messages"].append(AIMessage(content=response.content))
    return state

def router(state: GraphState) -> str:
    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user:
        return "retrieve"
    return "calc" if detect_calc_intent(last_user.content) else "retrieve"

# Graph assembly
workflow = StateGraph(GraphState)
workflow.add_node("calc", calc_agent)
workflow.add_node("retrieve", retrieve_agent)
workflow.add_node("answer", answer_agent)
workflow.add_node("router", lambda s: s)

workflow.add_edge(START, "router")
workflow.add_conditional_edges("router", router, {"calc": "calc", "retrieve": "retrieve"})
workflow.add_edge("calc", "answer")
workflow.add_edge("retrieve", "answer")
workflow.add_edge("answer", END)
app = workflow.compile()

FORECAST_FOLDER = r"C:\Users\Emirkan\Desktop\FinAssist Chatbot\bist30_august_forecasts"

ORDERED_COMPANY_NAMES = [
    "Akbank TAS Stock Price History_august_forecast",
    "Anadolu Efes Malt Stock Price History_august_forecast",
    "Aselsan Stock Price History_august_forecast",
    "Astor Enerji AS Stock Price History_august_forecast",
    "BIM Magazalar Stock Price History_august_forecast",
    "Cimsa Stock Price History_august_forecast",
    "Emlak Konut GYO Stock Price History_august_forecast",
    "ENKA Stock Price History_august_forecast",
    "Erdemir Stock Price History_august_forecast",
    "Ford Otosan Stock Price History_august_forecast",
    "Garanti Bank Stock Price History_august_forecast",
    "Gubretas Stock Price History_august_forecast",
    "Kardemir D Stock Price History_august_forecast",
    "Koc Holding Stock Price History_august_forecast",
    "Koza Altin Stock Price History_august_forecast",
    "Migros Stock Price History_august_forecast",
    "Pegasus Stock Price History_august_forecast",
    "Petkim Stock Price History_august_forecast",
    "Sabanci Holding Stock Price History_august_forecast",
    "SASA Polyester Stock Price History_august_forecast",
    "Sisecam Stock Price History_august_forecast",
    "TAV Havalimanlar Stock Price History_august_forecast",
    "THY Stock Price History_august_forecast",
    "Tofas Stock Price History_august_forecast",
    "Tupras Turkiye Stock Price History_august_forecast",
    "Turk Telekom Stock Price History_august_forecast",
    "Turkcell Stock Price History_august_forecast",
    "Turkiye Is Bankasi C Stock Price History_august_forecast",
    "Ulker Biskuvi Stock Price History_august_forecast",
    "Yapi ve Kredi Bankasi Stock Price History_august_forecast",
]

def _list_pngs() -> list[_Path]:
    folder = _Path(FORECAST_FOLDER)
    try:
        return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".png"]
    except FileNotFoundError:
        return []

def _normalize(s: str) -> str:
    tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    s = s.translate(tr_map).lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def robust_scan_companies() -> list[str]:
    files = _list_pngs()
    by_stem = {f.stem: f for f in files}
    ordered = [n for n in ORDERED_COMPANY_NAMES if n in by_stem]
    extras = sorted([s for s in by_stem.keys() if s not in set(ORDERED_COMPANY_NAMES)])
    return ordered + extras

def robust_forecast_image_path(company: str) -> str | None:
    files = _list_pngs()  # re-scan each time
    by_stem_lower = {f.stem.lower(): f for f in files}
    f = by_stem_lower.get(company.lower())
    if not f:
        want = _normalize(company)
        for cand in files:
            if _normalize(cand.stem) == want:
                f = cand
                break
    return f.as_posix() if f else None

def _read_image_as_array(path: str) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        return np.array(im)


THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")

INTRO_MSG = (
    "Merhaba! Ben **FinAssist**. Genel finans konularında bilgi verebilirim (TR/EN). \n\n"
    "Örnekler: \n"
    "• 'Bileşik faizle 10.000 TL, %24 yıllık, 3 yıl sonra ne olur?'\n"
    "• 'Kredi kartı faiz oranı nasıl hesaplanır?'\n"
    "• 'ETF nedir, yatırım fonundan farkı?'\n\n"
    "**Not:** Bilgilendirme amaçlıdır, yatırım tavsiyesi değildir."
)

SUGGESTED_QUESTIONS = [
    ["Bütçe planlaması nasıl yapılır?"],
    ["Bileşik faiz ile 50.000 TL %30 yıllık, 2 yıl sonra ne olur?"],
    ["Kredi kartı faiz oranı nasıl hesaplanır?"],
    ["ETF nedir ve mutual fund'dan farkı nedir?"],
]

def gradio_fn(message: str, history: List[List[str]]) -> str:
    msgs: List[BaseMessage] = []
    for user_text, bot_text in history:
        if user_text:
            msgs.append(HumanMessage(content=user_text))
        if bot_text:
            msgs.append(AIMessage(content=bot_text))
    msgs.append(HumanMessage(content=message))
    state: GraphState = {"messages": msgs, "context": "", "calc_result": None}
    out: GraphState = app.invoke(state)
    ai_text = next((m.content for m in reversed(out["messages"]) if isinstance(m, AIMessage)), "")
    return ai_text

with gr.Blocks(theme=THEME, css=".disclaimer{font-size:12px;opacity:0.75}") as demo:
    gr.Markdown("# 💬 FinAssist — Finans Sohbet Asistanı (TR/EN)")

    with gr.Tabs():
        
        with gr.TabItem("Sohbet"):
            gr.Markdown(INTRO_MSG)

            chatbot = gr.Chatbot(
                show_copy_button=True,
                show_label=True,
                height=520,
                layout="bubble",
                avatar_images=[None, None],
                type="tuples",  # explicit for older Gradio
            )

            with gr.Row():
                msg = gr.Textbox(placeholder="Mesajınızı yazın…", lines=1)
                send_btn = gr.Button("Gönder", variant="primary", scale=0)

            gr.Examples(SUGGESTED_QUESTIONS, inputs=[msg])
            clear_btn = gr.Button("Temizle")

            def respond(message, chat_history):
                chat_history = chat_history or []
                chat_history = [tuple(pair) for pair in chat_history]
                reply = gradio_fn(message, chat_history)
                chat_history = chat_history + [(message, reply)]
                return "", chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            send_btn.click(respond, [msg, chatbot], [msg, chatbot])
            clear_btn.click(lambda: [], None, chatbot, queue=False)

            gr.HTML("<div class='disclaimer'>Bu bir eğitim amaçlı demodur ve yatırım tavsiyesi değildir.</div>")

        
        with gr.TabItem("BIST30 Forecasts (Aug 2025)"):
            gr.Markdown("Aşağıdan bir şirket seçin ve **Göster**'e basın. Grafikler yerel klasörden okunur.")

            choices = robust_scan_companies()
            dd_company = gr.Dropdown(
                choices,
                label="Şirket",
                value=choices[0] if choices else None,
            )
            show_btn = gr.Button("Göster", variant="primary")

            forecast_img = gr.Image(label="Tahmin Grafiği", interactive=False)
            status_md = gr.Markdown("")

            def show_forecast(company):
                try:
                    p = robust_forecast_image_path(company)
                    if p:
                        arr = _read_image_as_array(p)
                        # IMPORTANT: return raw value (no component.update for your Gradio version)
                        return arr, ""
                    return None, f"⚠️ Görsel bulunamadı: {company}"
                except Exception as e:
                    return None, f"⚠️ Hata: {e}"

            show_btn.click(show_forecast, inputs=[dd_company], outputs=[forecast_img, status_md])

if __name__ == "__main__":
    print("Starting FinAssist (Gradio) …")
    print(f"[Forecasts] Using folder: {FORECAST_FOLDER}")
    demo.queue().launch()