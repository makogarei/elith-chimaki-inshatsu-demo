import re
from pathlib import Path
from typing import List, Optional

import altair as alt
import pandas as pd
import streamlit as st
from openai import OpenAI
from streamlit_markdown import st_markdown

try:
    from weasyprint import HTML  # type: ignore
except Exception:  # pragma: no cover
    HTML = None  # optional dependency for PDF生成


st.set_page_config(
    page_title="千巻印刷産業株式会社様デモ",
    page_icon="📈",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datas"
DEFAULT_MODEL = "gpt-5.1"  # gpt-5系を指定する場合はサイドバーで上書きしてください
PRIMARY = "#0ea5e9"
ACCENT = "#f97316"
SURFACE = "#ffffff"
MUTED = "#475569"
DEFAULT_LP_URL = "https://www.elith.ai/"
DEFAULT_REPORT_MODEL = "o3-deep-research-2025-06-26"  # Deep Research系モデルを指定


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --primary: {PRIMARY};
            --accent: {ACCENT};
            --surface: {SURFACE};
            --muted: {MUTED};
        }}
        html, body, [class*="css"] {{
            font-family: 'Noto Sans JP', 'Inter', system-ui, -apple-system, sans-serif;
            background: #f8fafc;
            color: #0f172a;
        }}
        .block-container {{
            padding-top: 56px;
            padding-bottom: 40px;
        }}
        .hero {{
            padding: 24px 26px;
            border-radius: 14px;
            background: linear-gradient(120deg, rgba(14,165,233,0.12), rgba(249,115,22,0.10));
            border: 1px solid rgba(148,163,184,0.25);
            margin-bottom: 20px;
        }}
        .hero-title {{
            font-size: 28px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 4px;
        }}
        .hero-sub {{
            color: #334155;
            font-size: 15px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(14,165,233,0.12);
            color: #0f172a;
            font-size: 12px;
            margin-right: 6px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px;
            margin: 6px 0 12px 0;
        }}
        .metric-card {{
            padding: 14px;
            border-radius: 12px;
            background: #ffffff;
            border: 1px solid rgba(148,163,184,0.25);
            box-shadow: 0 10px 30px rgba(15,23,42,0.05);
        }}
        .metric-title {{ color: #475569; font-size: 14px; margin-bottom: 6px; letter-spacing: 0.2px; }}
        .metric-value {{ color: #0f172a; font-size: 24px; font-weight: 700; }}
        .metric-desc {{ color: var(--muted); font-size: 12px; }}
        .pill {{
            padding: 4px 10px;
            background: rgba(14,165,233,0.12);
            color: #0f172a;
            border-radius: 999px;
            font-size: 12px;
        }}
        /* Tabs */
        .stTabs [role="tablist"] {{
            gap: 8px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
        }}
        .stTabs [role="tab"] {{
            padding: 10px 14px;
            border-radius: 10px 10px 0 0;
            background: #e2e8f0;
            color: #475569;
            border: 1px solid transparent;
            font-weight: 600;
        }}
        .stTabs [role="tab"][aria-selected="true"] {{
            background: #e0f2fe;
            color: #0f172a;
            border: 1px solid #bae6fd;
        }}
        /* Expander as panel */
        [data-testid="stExpander"] > details {{
            border: 1px solid #e2e8f0;
            background: #f1f5f9;
            border-radius: 14px;
        }}
        [data-testid="stExpander"] summary {{
            color: #0f172a;
            font-weight: 700;
        }}
        /* Dataframe tweaks */
        [data-testid="stDataFrame"] div[data-testid="stVerticalBlock"] {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_api_key() -> Optional[str]:
    """secrets.toml からAPI Keyを取得。"""
    return st.secrets.get("openai_api_key") or st.secrets.get("OPENAI_API_KEY")


def parse_int(value: Optional[str]) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except ValueError:
        return None


def parse_percent(value: Optional[str]) -> Optional[float]:
    if pd.isna(value):
        return None
    text = str(value).replace("%", "").replace(",", "").strip()
    try:
        return float(text) / 100.0
    except ValueError:
        return None


def format_seconds(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    minutes, seconds = divmod(int(value), 60)
    return f"{minutes}m {seconds:02d}s"


def markdown_to_html(md_text: str) -> str:
    # 簡易なMarkdown→HTML変換（見出しと箇条書きのみ）
    lines = md_text.splitlines()
    html_lines = []
    for line in lines:
        if line.startswith("### "):
            html_lines.append(f"<h3>{line[4:].strip()}</h3>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:].strip()}</h2>")
        elif line.startswith("# "):
            html_lines.append(f"<h1>{line[2:].strip()}</h1>")
        elif line.startswith("- "):
            # 簡易リスト
            if not html_lines or not html_lines[-1].startswith("<ul>"):
                html_lines.append("<ul>")
            html_lines.append(f"<li>{line[2:].strip()}</li>")
        else:
            # リスト閉じ
            if html_lines and html_lines[-1].startswith("<li>") and "</ul>" not in html_lines[-1]:
                html_lines.append("</ul>")
            html_lines.append(f"<p>{line}</p>")
    if html_lines and html_lines[-1].startswith("<li>") and "</ul>" not in html_lines[-1]:
        html_lines.append("</ul>")
    return "\n".join(html_lines)


def tune_chart(chart: alt.Chart) -> alt.Chart:
    """Altair共通スタイル適用。"""
    return (
        chart.configure_axis(labelFontSize=12, titleFontSize=12, gridColor="#e2e8f0")
        .configure_legend(labelFontSize=12, titleFontSize=12)
        .configure_view(strokeWidth=0)
    )


def html_to_pdf_bytes(html: str) -> Optional[bytes]:
    if HTML is None:
        return None
    return HTML(string=html).write_pdf()


def parse_duration_to_seconds(text: Optional[str]) -> Optional[int]:
    if pd.isna(text):
        return None
    minutes = 0
    seconds = 0
    text = str(text)
    min_match = re.search(r"(\d+)m", text)
    sec_match = re.search(r"(\d+)s", text)
    if min_match:
        minutes = int(min_match.group(1))
    if sec_match:
        seconds = int(sec_match.group(1))
    return minutes * 60 + seconds


def load_csv(uploaded_file, fallback_name: str) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    fallback_path = DATA_DIR / fallback_name
    if fallback_path.exists():
        return pd.read_csv(fallback_path)
    return pd.DataFrame()


def clean_traffic(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["日付"], errors="coerce")
    temp["pageviews"] = temp["ページ閲覧数"].apply(parse_int)
    temp["sessions"] = temp["サイトセッション数"].apply(parse_int)
    temp["unique_visitors"] = temp["ユニーク訪問者数"].apply(parse_int)
    temp["bounce_rate"] = temp["不達率"].apply(parse_percent)
    temp["avg_session_seconds"] = temp["平均セッション時間"].apply(
        parse_duration_to_seconds
    )
    return temp.dropna(subset=["date"])


def clean_conversion(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    temp = df.copy()
    temp.rename(
        columns={"トラフィックカテゴリー": "traffic_category", "アクセス元": "source"},
        inplace=True,
    )
    temp["sessions"] = temp["サイトセッション"].apply(parse_int)
    temp["pageviews"] = temp["ページ閲覧数"].apply(parse_int)
    temp["unique_visitors"] = temp["ユニーク訪問者"].apply(parse_int)
    return temp


def clean_clicks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    temp = df.copy()
    temp.rename(
        columns={
            "ボタンのテキスト": "button_text",
            "ボタンタイプ": "button_type",
            "ページパス": "page_path",
            "リンク先アイテム": "link_item",
            "リンク詳細": "link_detail",
        },
        inplace=True,
    )
    temp["visitors"] = temp["ユニーク訪問者数"].apply(parse_int)
    temp["clicks"] = temp["ユニーククリック数"].apply(parse_int)
    temp["click_rate"] = temp["クリック率"].apply(parse_percent)
    return temp


def summarize_traffic(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    total_sessions = df["sessions"].sum(skipna=True)
    total_pageviews = df["pageviews"].sum(skipna=True)
    total_visitors = df["unique_visitors"].sum(skipna=True)

    weighted_time = (
        (df["avg_session_seconds"] * df["sessions"]).sum(skipna=True)
        / total_sessions
        if total_sessions
        else None
    )
    weighted_bounce = (
        (df["bounce_rate"] * df["sessions"]).sum(skipna=True) / total_sessions
        if total_sessions
        else None
    )

    return {
        "total_sessions": int(total_sessions) if pd.notna(total_sessions) else None,
        "total_pageviews": int(total_pageviews) if pd.notna(total_pageviews) else None,
        "total_visitors": int(total_visitors) if pd.notna(total_visitors) else None,
        "avg_session_seconds": weighted_time,
        "bounce_rate": weighted_bounce,
    }


def build_page_click_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby("page_path")
        .agg(
            visitors=("visitors", "sum"),
            clicks=("clicks", "sum"),
            avg_click_rate=("click_rate", "mean"),
        )
        .reset_index()
    )
    grouped["click_rate_calc"] = grouped["clicks"] / grouped["visitors"].replace(
        {0: pd.NA}
    )
    return grouped.sort_values("clicks", ascending=False)


def behavioral_highlights(traffic_df: pd.DataFrame, page_click_df: pd.DataFrame) -> List[str]:
    highlights: List[str] = []
    if not traffic_df.empty:
        latest = traffic_df.sort_values("date").tail(3)
        if not latest.empty:
            avg_sessions = latest["sessions"].mean()
            delta = latest["sessions"].pct_change().mean()
            if pd.notna(delta):
                trend = "増加" if delta > 0 else "減少"
                highlights.append(
                    f"直近3日間のセッション平均は約{avg_sessions:.0f}。前日比平均は{delta*100:+.1f}%で{trend}傾向。"
                )
    if not page_click_df.empty:
        top_pages = page_click_df.head(3)
        names = ", ".join(top_pages["page_path"].tolist())
        highlights.append(f"クリックが多いページ: {names}")
        low_engagement = (
            page_click_df[
                (page_click_df["click_rate_calc"] < page_click_df["click_rate_calc"].median())
                & (page_click_df["visitors"] > 50)
            ]
            .sort_values("click_rate_calc")
            .head(3)
        )
        if not low_engagement.empty:
            low_list = ", ".join(low_engagement["page_path"].tolist())
            highlights.append(f"クリック率が低めのページ: {low_list}")
    return highlights


def build_ai_prompt(
    summary: dict,
    conversion_df: pd.DataFrame,
    page_click_df: pd.DataFrame,
    extra_context: str = "",
) -> str:
    lines = [
        "以下はLP行動ログのサマリーです。LP改善のための具体的な提案を3〜5点、日本語で短く箇条書きしてください。",
        "",
    ]
    if summary:
        lines.append(
            f"- 集計期間のセッション合計: {summary.get('total_sessions')} / ページビュー合計: {summary.get('total_pageviews')} / ユーザー合計: {summary.get('total_visitors')}"
        )
        if summary.get("avg_session_seconds") is not None:
            lines.append(
                f"- 平均セッション時間(加重平均): {summary['avg_session_seconds']:.1f}秒"
            )
        if summary.get("bounce_rate") is not None:
            lines.append(f"- 推定直帰率(加重平均): {summary['bounce_rate']*100:.1f}%")
    if not conversion_df.empty:
        top_sources = (
            conversion_df.sort_values("sessions", ascending=False)
            .head(5)[["traffic_category", "source", "sessions"]]
            .to_dict("records")
        )
        lines.append("- 主要流入元(セッション順):")
        for src in top_sources:
            lines.append(
                f"  - {src['traffic_category']} / {src['source']}: {src['sessions']}セッション"
            )
    if not page_click_df.empty:
        top_pages = page_click_df.head(5).to_dict("records")
        lines.append("- ページ別の主なクリック状況:")
        for row in top_pages:
            cr = row["click_rate_calc"] * 100 if pd.notna(row["click_rate_calc"]) else None
            lines.append(
                f"  - {row['page_path']}: 訪問 {row['visitors']} / クリック {row['clicks']} / クリック率 {cr:.1f}%"
            )
    if extra_context.strip():
        lines.append(f"- 補足情報: {extra_context.strip()}")
    lines.append(
        "流入改善、CVR向上、直帰率低減につながる具体的な施策を提示してください。各施策には理由を一言添えてください。"
    )
    return "\n".join(lines)


def call_openai_deep_research(
    api_key: str, model: str, prompt: str, max_tokens: int = 2000
) -> str:
    """
    Deep Research API向けとしていたが、ここでは汎用のGPT-5系(chat.completions)で
    詳細レポートを生成する。max_tokensはモデルの制約に合わせて調整してください。
    """
    client = OpenAI(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": "あなたは市場調査とLP最適化に長けたリサーチャーです。情報量多めで具体的な日本語レポートをMarkdownで返してください。",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            max_completion_tokens=max_tokens,
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "max_completion_tokens" in msg:
            # max_completion_tokens 未対応モデルなら、トークン指定なしで再試行
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5,
            )
        elif "max_tokens" in msg:
            # max_tokens 未対応(= max_completion_tokens 必須)なら、再度 max_completion_tokens で試行
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5,
                max_completion_tokens=max_tokens,
            )
        else:
            # その他のエラーはそのまま上げる
            raise
    return response.choices[0].message.content


def build_report_prompt(
    summary: dict,
    conversion_df: pd.DataFrame,
    page_click_df: pd.DataFrame,
    goal_text: str,
) -> str:
    lines = [
        "あなたはLP最適化とCROのエキスパートです。以下のログ要約を基に、A4 1枚程度の日本語レポートを作成してください。",
        "読者はマーケ/事業責任者を想定し、箇条書き中心だが情報量多めで、数字と仮説を明示してください。",
        "",
        "必須構成:",
        "1) 期間サマリー（セッション/ PV / UU / 平均セッション時間 / 直帰率）",
        "2) ページごとの簡易集計（訪問数・平均滞在時間・クリック数の傾向。主要ページを列挙）",
        "3) ユーザー行動の特徴（よく見られているページ、直帰が多そうなページなどのインサイト）",
        "4) 流入インサイト（主要流入元のシェアと質の仮説、改善余地）",
        "5) LP改善につながる示唆・施策案（3〜5点、優先度・理由・期待効果・必要なアセット/改修・簡易実装案）",
        "6) 計測と実験アイデア（トラッキング追加、A/B案、主要KPIと副指標）",
        "紙面はA4 1枚相当で、箇条書きを主体にしつつ具体的に。必ず数字を入れてください。",
    ]
    if goal_text.strip():
        lines.append(f"- LPのゴール/補足: {goal_text.strip()}")
    if summary:
        lines.append(
            f"- 期間サマリー: セッション {summary.get('total_sessions')} / PV {summary.get('total_pageviews')} / UU {summary.get('total_visitors')} / 平均セッション {format_seconds(summary.get('avg_session_seconds'))} / 直帰率 {summary.get('bounce_rate')*100:.1f}%"
            if summary.get("bounce_rate") is not None
            else f"- 期間サマリー: セッション {summary.get('total_sessions')} / PV {summary.get('total_pageviews')} / UU {summary.get('total_visitors')}"
        )
    if not conversion_df.empty:
        top_sources = (
            conversion_df.sort_values("sessions", ascending=False)
            .head(5)[["traffic_category", "source", "sessions"]]
            .to_dict("records")
        )
        lines.append("- 流入上位 (セッション順): " + "; ".join([f"{r['traffic_category']} / {r['source']} : {r['sessions']}" for r in top_sources]))
    if not page_click_df.empty:
        top_pages = page_click_df.head(5).to_dict("records")
        lines.append("- クリック上位ページ: " + "; ".join([f"{r['page_path']} (訪問 {r['visitors']}, クリック {r['clicks']})" for r in top_pages]))
    lines.append("A4 1ページに収まるように簡潔にまとめてください。")
    return "\n".join(lines)


def main() -> None:
    inject_styles()
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">LP行動ログアナライザー</div>
            <div style="margin-top:6px;">
                <span class="badge">Streamlit</span>
                <span class="badge">Altair</span>
                <span class="badge">OpenAI</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("設定")
        model = st.text_input("モデル名", value=DEFAULT_MODEL)
        secret_hint = st.secrets.get("openai_api_key") or st.secrets.get("OPENAI_API_KEY")
        if secret_hint:
            st.success("secrets.toml の openai_api_key を検出しました。")
        else:
            st.warning("secrets.toml に openai_api_key が設定されていません。設定すると改善案が生成できます。")
        goal_text = st.text_area("ログ解析目的", value="採用の応募率を改善したい")
        st.divider()
        st.subheader("データ差し替え")
        traffic_upload = st.file_uploader("トラフィックレポート CSV", type=["csv"])
        conversion_upload = st.file_uploader("流入元レポート CSV", type=["csv"])
        clicks_upload = st.file_uploader("ボタンクリック CSV", type=["csv"])
        st.caption("未指定時は datas/ 配下のサンプルを利用します。")

    api_key = get_api_key()

    raw_traffic = load_csv(traffic_upload, "トラフィックレポート_2025-11-06-2025-12-06.csv")
    raw_conversion = load_csv(conversion_upload, "conversion_table_api_2025-11-06-2025-12-06.csv")
    raw_clicks = load_csv(clicks_upload, "button_clicks_table_api_2025-11-06-2025-12-06.csv")

    traffic_df = clean_traffic(raw_traffic)
    conversion_df = clean_conversion(raw_conversion)
    clicks_df = clean_clicks(raw_clicks)
    page_click_summary = build_page_click_summary(clicks_df)

    date_min = traffic_df["date"].min().date() if not traffic_df.empty else None
    date_max = traffic_df["date"].max().date() if not traffic_df.empty else None
    with st.expander("フィルター", expanded=True):
        colf1, colf2, colf3, colf4 = st.columns(4)
        if date_min and date_max:
            date_range = colf1.date_input(
                "分析期間",
                (date_min, date_max),
                min_value=date_min,
                max_value=date_max,
            )
        else:
            date_range = ()
        page_keyword = colf2.text_input("ページパスで絞り込み", placeholder="/project, /company ...")
        max_vis = int(page_click_summary["visitors"].max()) if not page_click_summary.empty else 100
        max_vis = max(max_vis, 50)
        min_visitors = colf3.slider("訪問数の下限 (CTA集計)", 0, max_vis, min(50, max_vis), step=10)
        top_n = colf4.slider("可視化する上位件数", 5, 50, 15, step=5)

        traffic_view = traffic_df.copy()
        if isinstance(date_range, tuple) and len(date_range) == 2 and date_range[0] and date_range[1]:
            start, end = date_range
            traffic_view = traffic_view[
                (traffic_view["date"].dt.date >= start) & (traffic_view["date"].dt.date <= end)
            ]
        clicks_view = page_click_summary.copy()
        if page_keyword:
            key = page_keyword.lower()
            clicks_view = clicks_view[clicks_view["page_path"].str.lower().str.contains(key)]
        clicks_view = clicks_view[clicks_view["visitors"] >= min_visitors]
        clicks_for_chart = clicks_view.head(top_n)

        summary = summarize_traffic(traffic_view)

        metric_cards = [
            {
                "label": "セッション合計",
                "value": f"{summary.get('total_sessions', 0):,}",
                "desc": "期間内の総セッション",
            },
            {
                "label": "ページビュー合計",
                "value": f"{summary.get('total_pageviews', 0):,}",
                "desc": "閲覧ページ総数",
            },
            {
                "label": "訪問ユーザー合計",
                "value": f"{summary.get('total_visitors', 0):,}",
                "desc": "ユニーク訪問",
            },
            {
                "label": "平均セッション時間",
                "value": format_seconds(summary.get("avg_session_seconds")),
                "desc": "加重平均",
            },
        ]
        bounce_rate = summary.get("bounce_rate")
        if bounce_rate is not None:
            metric_cards.append(
                {
                    "label": "推定直帰率",
                    "value": f"{bounce_rate*100:.1f}%",
                    "desc": "期間加重平均",
                }
            )
        cards_html = '<div class="metric-grid">' + "".join(
            [
                f"<div class='metric-card'><div class='metric-title'>{c['label']}</div>"
                f"<div class='metric-value'>{c['value']}</div>"
                f"<div class='metric-desc'>{c['desc']}</div></div>"
                for c in metric_cards
            ]
        ) + "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

    tabs = st.tabs(
        ["概要", "行動/CTA", "流入元", "改善案 (AI)", "生データ", "LPプレビュー"]
    )

    with tabs[0]:
        st.subheader("スナップショット")
        highlights = behavioral_highlights(traffic_view, clicks_view)
        if highlights:
            st.write("行動特徴")
            for txt in highlights:
                st.write(f"- {txt}")
        else:
            st.info("特徴を抽出するためのデータが足りませんでした。")

        col_a, col_b = st.columns(2)
        if not clicks_view.empty:
            col_a.write("クリックが多いページ (上位5)")
            top_pages = clicks_view.head(5)[["page_path", "visitors", "clicks", "click_rate_calc"]]
            col_a.dataframe(
                top_pages.rename(
                    columns={
                        "page_path": "ページ",
                        "visitors": "訪問",
                        "clicks": "クリック",
                        "click_rate_calc": "クリック率",
                    }
                ),
                use_container_width=True,
            )
            col_b.write("クリック率が低いページ (訪問数フィルタ適用)")
            low_pages = (
                clicks_view.sort_values("click_rate_calc", ascending=True)
                .head(5)[["page_path", "visitors", "clicks", "click_rate_calc"]]
            )
            col_b.dataframe(
                low_pages.rename(
                    columns={
                        "page_path": "ページ",
                        "visitors": "訪問",
                        "clicks": "クリック",
                        "click_rate_calc": "クリック率",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.info("クリックデータがありません。")

    with tabs[1]:
        st.subheader("行動トレンド")
        if traffic_view.empty:
            st.warning("トラフィックデータがありません。")
        else:
            trend_df = traffic_view[["date", "pageviews", "sessions", "unique_visitors"]]
            melt = trend_df.melt("date", var_name="metric", value_name="value")
            chart = tune_chart(
                alt.Chart(melt)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="日付"),
                    y=alt.Y("value:Q", title="件数"),
                    color=alt.Color("metric:N", title="指標"),
                    tooltip=["date:T", "metric:N", "value:Q"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

            engagement = traffic_view[["date", "avg_session_seconds", "bounce_rate"]].melt(
                "date", var_name="metric", value_name="value"
            )
            engagement["value_display"] = engagement.apply(
                lambda r: r["value"] * 100 if r["metric"] == "bounce_rate" else r["value"],
                axis=1,
            )
            engagement_chart = tune_chart(
                alt.Chart(engagement)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="日付"),
                    y=alt.Y("value_display:Q", title="値"),
                    color=alt.Color("metric:N", title="指標"),
                    tooltip=["date:T", "metric:N", "value_display:Q"],
                )
                .properties(height=260)
            )
            st.caption("滞在時間は秒、直帰率は%として表示")
            st.altair_chart(engagement_chart, use_container_width=True)

        st.subheader("CTA / クリック分布")
        if clicks_for_chart.empty:
            st.warning("CTAデータがありません。フィルタを調整してください。")
        else:
            scatter = tune_chart(
                alt.Chart(clicks_for_chart)
                .mark_circle(size=120)
                .encode(
                    x=alt.X("visitors:Q", title="訪問数"),
                    y=alt.Y("click_rate_calc:Q", title="クリック率", axis=alt.Axis(format=".0%")),
                    size=alt.Size("clicks:Q", title="クリック数"),
                    color=alt.Color("click_rate_calc:Q", title="クリック率", scale=alt.Scale(scheme="blues")),
                    tooltip=[
                        "page_path",
                        "visitors",
                        "clicks",
                        alt.Tooltip("click_rate_calc:Q", title="クリック率", format=".1%"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(scatter, use_container_width=True)
            st.dataframe(
                clicks_view[
                    ["page_path", "visitors", "clicks", "click_rate_calc", "avg_click_rate"]
                ].rename(
                    columns={
                        "page_path": "ページ",
                        "visitors": "訪問数",
                        "clicks": "クリック数",
                        "click_rate_calc": "クリック率(算出)",
                        "avg_click_rate": "クリック率(平均)",
                    }
                ),
                use_container_width=True,
                height=420,
            )

    with tabs[2]:
        st.subheader("流入元")
        if conversion_df.empty:
            st.warning("流入元データがありません。")
        else:
            conv = conversion_df.copy()
            conv["session_share"] = conv["sessions"] / conv["sessions"].sum()
            top_conv = conv.sort_values("sessions", ascending=False).head(12)
            conv_chart = tune_chart(
                alt.Chart(top_conv)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("sessions:Q", title="セッション数"),
                    y=alt.Y("source:N", sort="-x", title="流入元"),
                    color=alt.Color("traffic_category:N", title="カテゴリー"),
                    tooltip=[
                        "traffic_category",
                        "source",
                        "sessions",
                        "unique_visitors",
                        alt.Tooltip("session_share:Q", format=".1%", title="シェア"),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(conv_chart, use_container_width=True)
            st.dataframe(
                top_conv[
                    ["traffic_category", "source", "sessions", "pageviews", "unique_visitors", "session_share"]
                ].rename(
                    columns={
                        "traffic_category": "カテゴリ",
                        "source": "ソース",
                        "sessions": "セッション",
                        "pageviews": "PV",
                        "unique_visitors": "訪問者",
                        "session_share": "シェア",
                    }
                ),
                use_container_width=True,
                height=400,
            )

    with tabs[3]:
        st.subheader("レポート生成 (OpenAI)")
        st.markdown("GPT-5系モデルでA4 1ページ相当の詳細レポートを生成します。")
        report_model = st.text_input("レポート生成モデル", value=DEFAULT_MODEL)
        if "report_md" not in st.session_state:
            st.session_state.report_md = ""
        if "report_pdf" not in st.session_state:
            st.session_state.report_pdf = None
        if "report_html" not in st.session_state:
            st.session_state.report_html = ""

        if api_key:
            if st.button("レポート生成 (1ページ)", type="primary"):
                report_prompt = build_report_prompt(
                    summary, conversion_df, clicks_view, goal_text or ""
                )
                with st.spinner("GPT-5系モデルでレポート生成中..."):
                    try:
                        md = call_openai_deep_research(
                            api_key, report_model or model, report_prompt
                        )
                        st.session_state.report_md = md
                        html = markdown_to_html(md)
                        st.session_state.report_html = html
                        st.session_state.report_pdf = html_to_pdf_bytes(html)
                        st.success("レポート生成が完了しました。")
                    except Exception as exc:
                        st.error(f"レポート生成でエラー: {exc}")

            if st.session_state.report_md:
                st.markdown("##### レポートプレビュー")
                st.text(st.session_state.report_md)
                # if st.session_state.report_pdf:
                #     st.download_button(
                #         "PDFをダウンロード",
                #         data=st.session_state.report_pdf,
                #         file_name="lp_report.pdf",
                #         mime="application/pdf",
                #     )
                # else:
                #     st.info(
                #         "PDF生成には weasyprint が必要です。`pip install weasyprint` を実行後、再度お試しください。"
                #     )
        else:
            st.info("レポート生成には OpenAI API Key (.streamlit/secrets.toml) を設定してください。")

    with tabs[4]:
        st.subheader("入力データの確認")
        st.write("トラフィックサマリ")
        st.dataframe(traffic_df, use_container_width=True)
        st.write("流入元データ")
        st.dataframe(conversion_df, use_container_width=True)
        st.write("クリックログ")
        st.dataframe(clicks_df, use_container_width=True, height=320)

    with tabs[5]:
        st.subheader("LPプレビュー (iframe)")
        st.caption(
            "X-Frame-Options や CSP でブロックされている場合は表示されません。その際はブラウザで直接開いてください。"
        )
        lp_url = st.text_input("URL", value=DEFAULT_LP_URL)
        height = st.slider("高さ(px)", 400, 1400, 900, step=50)
        if lp_url:
            try:
                st.components.v1.iframe(lp_url, height=height, scrolling=True)
            except Exception as exc:
                st.error(f"iframe 埋め込みに失敗しました: {exc}")


if __name__ == "__main__":
    main()
