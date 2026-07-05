"""
UI Components & Design Tokens — 日系簡約風
和風ミニマル · 淡色系 · Noto Sans TC
"""
import html as _html
import streamlit as st

# ─── Design Tokens ──────────────────────────────────────────────
P = dict(
    bg="#F8F7F5", surface="#FFFFFF", card="#FDFCFA",
    border="#E8E4DE", border2="#D4CFC8",
    ink="#2A2A2A", ink2="#505050", muted="#9A9490",
    primary="#4E7FB0", accent="#8B7BA8",
    high="#C4645A", medium="#C49A4A", low="#5B9E73",
    tag_bg="#F2F0EC", mbg="#EEF4FB", mtxt="#3D6B96",
    landlord="#4E7FB0", tenant="#5B9E73", admin="#8B7BA8",
)

# Risk color map
RC = {"高風險": P["high"], "中風險": P["medium"], "低風險": P["low"]}
# Room type color map
RTC = {
    "整棟出租": P["primary"], "私人套房": P["accent"],
    "共用套房": P["medium"], "飯店客房": P["low"],
}
# Room type translation
ROOM_JP = {
    "Entire home/apt": "整棟出租", "Private room": "私人套房",
    "Shared room": "共用套房", "Hotel room": "飯店客房",
}
# Feature name translation
FEAT_ZH = {
    "availability_365":              "年度可訂天數",
    "number_of_reviews":             "評論總數",
    "number_of_reviews_ltm":         "近12月評論數",
    "reviews_per_month":             "月均評論數",
    "price":                         "每晚價格",
    "calculated_host_listings_count": "房東房源數",
    "minimum_nights":                "最少入住晚數",
    "rt_Entire home/apt":            "房型：整棟",
    "rt_Shared room":                "房型：共用",
    "rt_Private room":               "房型：私人",
    "rt_Hotel room":                 "房型：飯店",
    "review_scores_rating":          "評分",
    "accommodates":                  "可住人數",
    "bedrooms":                      "臥室數",
    "beds":                          "床數",
    "bathrooms_count":               "衛浴數",
}


def inject_css():
    """Inject the Japanese minimalist CSS theme."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');
    html,body,.stApp{{background:{P['bg']};color:{P['ink']};
      font-family:'Noto Sans TC',sans-serif;}}
    section[data-testid="stSidebar"]{{background:{P['surface']};
      border-right:1px solid {P['border']};}}
    [data-testid="stMetric"]{{background:{P['surface']};border:1px solid {P['border']};
      border-radius:12px;padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,.03);
      transition:box-shadow .2s ease;}}
    [data-testid="stMetric"]:hover{{box-shadow:0 3px 12px rgba(0,0,0,.07);}}
    [data-testid="stMetricLabel"]{{color:{P['muted']} !important;
      font-size:.72rem !important;letter-spacing:.08em;text-transform:uppercase;}}
    [data-testid="stMetricValue"]{{color:{P['ink']} !important;
      font-size:1.3rem !important;font-weight:700;
      white-space:normal !important;overflow:visible !important;
      text-overflow:clip !important;line-height:1.25;}}
    [data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{{
      white-space:normal !important;overflow:visible !important;
      text-overflow:clip !important;}}
    [data-testid="stMetricValue"] *{{white-space:normal !important;
      overflow:visible !important;text-overflow:clip !important;}}
    .stTabs [data-baseweb="tab-list"]{{background:transparent;
      border-bottom:2px solid {P['border']};gap:0;padding:0;}}
    .stTabs [data-baseweb="tab"]{{color:{P['muted']};border-radius:0;
      padding:9px 20px;border-bottom:2px solid transparent;margin-bottom:-2px;
      font-size:.85rem;font-weight:500;transition:all .2s ease;}}
    .stTabs [aria-selected="true"]{{color:{P['primary']} !important;
      border-bottom:2px solid {P['primary']} !important;background:transparent !important;}}
    section[data-testid="stSidebar"] label{{color:{P['ink2']} !important;font-size:.80rem;}}
    .sec{{font-size:.71rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
      color:{P['muted']};margin:18px 0 3px;padding-bottom:7px;
      border-bottom:1px solid {P['border']};}}
    .mb{{display:inline-flex;align-items:center;gap:4px;
      background:{P['mbg']};border:1px solid #C8DCF0;border-radius:5px;
      padding:3px 10px;font-size:.70rem;font-weight:600;color:{P['mtxt']};
      letter-spacing:.03em;margin-bottom:7px;}}
    .mhigh{{background:#FEF2F0;border:1px solid #F0B8B4;color:#A03028;}}
    .note{{background:{P['tag_bg']};border-left:3px solid {P['primary']};
      padding:9px 14px;border-radius:0 6px 6px 0;
      font-size:.79rem;color:{P['ink2']};margin:8px 0;}}
    hr{{border:none;border-top:1px solid {P['border']} !important;margin:14px 0;}}
    ::-webkit-scrollbar{{width:4px;}}
    ::-webkit-scrollbar-thumb{{background:{P['border2']};border-radius:2px;}}
    .portal-card{{
      background:{P['surface']};
      border:1px solid {P['border']};
      border-radius:16px;
      padding:36px 28px;
      text-align:center;
      transition:all .3s cubic-bezier(.4,0,.2,1);
      cursor:pointer;
      box-shadow:0 1px 4px rgba(0,0,0,.03);
    }}
    .portal-card:hover{{
      box-shadow:0 8px 30px rgba(0,0,0,.08);
      transform:translateY(-4px);
      border-color:{P['primary']};
    }}
    .portal-icon{{font-size:2.8rem;margin-bottom:12px;}}
    .portal-title{{font-size:1.15rem;font-weight:700;color:{P['ink']};margin-bottom:6px;}}
    .portal-desc{{font-size:.8rem;color:{P['muted']};line-height:1.5;}}
    .risk-badge{{
      display:inline-block;padding:3px 12px;border-radius:20px;
      font-size:.72rem;font-weight:700;letter-spacing:.04em;
    }}
    .risk-high{{background:#FDECEA;color:{P['high']};}}
    .risk-medium{{background:#FDF5E4;color:#A07A20;}}
    .risk-low{{background:#EAF5EE;color:#3D7A55;}}
    .sentiment-pos{{color:{P['low']};font-weight:600;}}
    .sentiment-neg{{color:{P['high']};font-weight:600;}}
    .sentiment-neu{{color:{P['muted']};font-weight:500;}}
    .stat-card{{
      background:linear-gradient(135deg,{P['surface']},{P['tag_bg']});
      border:1px solid {P['border']};border-radius:12px;
      padding:16px 20px;text-align:center;
    }}
    .stat-value{{font-size:1.6rem;font-weight:700;color:{P['ink']};}}
    .stat-label{{font-size:.72rem;color:{P['muted']};letter-spacing:.06em;margin-top:4px;}}
    .rv-wrap{{position:relative;display:inline-block;cursor:help;
      color:{P['primary']};font-weight:600;
      border-bottom:1px dashed {P['primary']};}}
    .rv-wrap .rv-tip{{visibility:hidden;opacity:0;position:absolute;z-index:9999;
      left:0;top:150%;width:330px;max-height:300px;overflow-y:auto;
      background:{P['surface']};border:1px solid {P['border2']};border-radius:10px;
      padding:10px 13px;box-shadow:0 10px 34px rgba(0,0,0,.18);
      transition:opacity .15s ease;text-align:left;white-space:normal;
      font-size:.72rem;line-height:1.55;color:{P['ink2']};font-weight:400;}}
    .rv-wrap:hover .rv-tip{{visibility:visible;opacity:1;}}
    .rv-tip-h{{font-size:.64rem;font-weight:700;color:{P['muted']};
      letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;}}
    .rv-item{{padding:6px 0;border-bottom:1px dashed {P['border']};}}
    .rv-item:last-child{{border-bottom:none;}}
    .hero{{position:relative;display:flex;flex-wrap:wrap;border-radius:16px;
      overflow:hidden;border:1px solid {P['border']};
      box-shadow:0 6px 26px rgba(0,0,0,.08);margin:4px 0 8px;}}
    .hero-half{{position:relative;flex:1 1 320px;min-height:196px;
      padding:22px 30px 0;overflow:hidden;}}
    .hero-txt{{position:relative;z-index:2;}}
    .hero-l{{background:linear-gradient(180deg,#FBFAF8,#EFEDE9);}}
    .hero-r{{background:linear-gradient(160deg,#7ED6E8 0%,#BFE08A 52%,#F7D774 100%);}}
    .hero-tag{{display:inline-block;font-size:.66rem;font-weight:700;
      letter-spacing:.2em;color:{P['muted']};margin-bottom:8px;}}
    .hero-r .hero-tag{{color:#256048;}}
    .hero-half h2{{font-size:1.55rem;line-height:1.2;font-weight:800;
      color:{P['ink']};margin:0 0 8px;letter-spacing:-.5px;}}
    .hero-half p{{font-size:.76rem;line-height:1.6;color:{P['muted']};margin:0;
      max-width:92%;}}
    .hero-r p{{color:#20543E;}}
    .hero-cta{{margin-top:14px;display:inline-block;font-size:.78rem;
      font-weight:700;color:{P['landlord']};}}
    .hero-r .hero-cta{{color:#1C4A36;}}
    .hero-sky{{position:absolute;left:0;bottom:0;width:100%;height:104px;
      z-index:1;display:block;}}
    .hero-seam{{position:absolute;top:-4%;left:calc(50% - 7px);width:14px;
      height:108%;background:{P['surface']};transform:rotate(4deg);z-index:3;
      box-shadow:0 0 12px rgba(0,0,0,.06);}}
    @media(max-width:660px){{.hero-half{{flex:1 1 100%;min-height:166px;}}
      .hero-seam{{display:none;}}}}
    [data-testid="stSidebarNav"]{{display:none;}}
    .block-container,[data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"]{{padding-top:1.6rem !important;}}
    section[data-testid="stSidebar"] [data-testid="stPageLink"] a{{
      border-radius:8px;padding:6px 10px;font-size:.86rem;}}
    section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover{{
      background:{P['tag_bg']};}}
    </style>
    """, unsafe_allow_html=True)


# ─── UI helper functions ────────────────────────────────────────
def sec(t):
    """Section header."""
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)


def mb(text, warning=False):
    """Method badge."""
    cls = "mb mhigh" if warning else "mb"
    st.markdown(f'<span class="{cls}">📐 {text}</span>', unsafe_allow_html=True)


def note(t):
    """Info note block."""
    st.markdown(f'<div class="note">{t}</div>', unsafe_allow_html=True)


def risk_badge(level):
    """Return HTML for a risk level badge."""
    cls_map = {"高風險": "risk-high", "中風險": "risk-medium", "低風險": "risk-low"}
    cls = cls_map.get(level, "risk-medium")
    return f'<span class="risk-badge {cls}">{level}</span>'


def stat_card(value, label, color=None):
    """Render a stat card."""
    c = color or P["ink"]
    st.markdown(f'''
    <div class="stat-card">
      <div class="stat-value" style="color:{c};">{value}</div>
      <div class="stat-label">{label}</div>
    </div>''', unsafe_allow_html=True)


def html_table(df_in, fmt=None, cell_fn=None, height=360, wrap=False, scroll=True):
    """Render a styled HTML table.

    wrap=True   lets cell text wrap so every column fits the width
                (no bottom/horizontal scrollbar).
    scroll=False renders the full table with no inner scroll container
                (the dialog/page provides its own vertical scroll).
    """
    fmt = fmt or {}
    cell_fn = cell_fn or {}
    ws = "normal" if wrap else "nowrap"
    wb = "break-word" if wrap else "normal"
    th = (f"background:{P['tag_bg']};color:{P['muted']};font-size:.70rem;"
          f"letter-spacing:.07em;text-transform:uppercase;padding:8px 13px;"
          f"border-bottom:2px solid {P['border2']};white-space:{ws};"
          f"text-align:left;position:sticky;top:0;z-index:1;")
    td0 = (f"padding:7px 13px;font-size:.80rem;color:{P['ink']};"
           f"border-bottom:1px solid {P['border']};white-space:{ws};"
           f"word-break:{wb};vertical-align:top;")
    hdr = "".join(f'<th style="{th}">{c}</th>' for c in df_in.columns)
    rows = []
    import pandas as pd
    for i, (_, row) in enumerate(df_in.iterrows()):
        bg = P["surface"] if i % 2 == 0 else P["tag_bg"]
        cells = []
        for col in df_in.columns:
            v = row[col]
            disp = ("–" if pd.isna(v)
                    else (fmt[col].format(v) if col in fmt and pd.notna(v)
                          else str(v)))
            css = f"background:{bg};"
            if col in cell_fn:
                try:
                    css += cell_fn[col](v)
                except Exception:
                    pass
            cells.append(f'<td style="{td0}{css}">{disp}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    container = (f'overflow:auto;max-height:{height}px;' if scroll
                 else 'overflow:visible;')
    tstyle = ("width:100%;border-collapse:collapse;"
              + ("table-layout:fixed;" if wrap else ""))
    st.markdown(
        f'<div style="{container}border:1px solid {P["border"]};'
        f'border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.03);">'
        f'<table style="{tstyle}">'
        f'<thead><tr>{hdr}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
        unsafe_allow_html=True)


# ─── Plotly chart theme ─────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=P["ink2"], family="Noto Sans TC,sans-serif", size=11),
    margin=dict(l=46, r=16, t=28, b=34),
    legend=dict(
        bgcolor="rgba(255,255,255,.8)", bordercolor=P["border"],
        borderwidth=1, font=dict(color=P["ink2"]),
    ),
    xaxis=dict(
        gridcolor=P["border"], linecolor=P["border"], zeroline=False,
        tickfont=dict(color=P["muted"]),
    ),
    yaxis=dict(
        gridcolor=P["border"], linecolor=P["border"], zeroline=False,
        tickfont=dict(color=P["muted"]),
    ),
)


def apply_theme(fig, h=None, legend=True):
    """Apply the Japanese minimalist theme to a Plotly figure."""
    kw = dict(**PLOTLY_LAYOUT)
    if h:
        kw["height"] = h
    if not legend:
        kw["showlegend"] = False
    return fig.update_layout(**kw)


def review_hover_html(count, snippets, label=None):
    """
    Return an inline HTML span that reveals recent reviews on hover.
    `snippets` is a list of plain-text review strings.
    """
    label = label or f"💬 {count} 則評論"
    if not snippets:
        return (f'<span style="color:#9A9490;font-size:.78rem;">{label}'
                f'（尚無評論內容）</span>')
    items = "".join(
        f'<div class="rv-item">{_html.escape(str(s))}</div>' for s in snippets)
    return (f'<span class="rv-wrap">{label}'
            f'<span class="rv-tip"><div class="rv-tip-h">最新評論預覽</div>'
            f'{items}</span></span>')


def sidebar_nav():
    """Custom sidebar navigation: 回首頁 + 三入口 (replaces default 'app' entry)."""
    links = [
        ("app.py", "🏯 回首頁"),
        ("pages/1_🏠_房東入口.py", "🏠 房東入口"),
        ("pages/2_🔍_租客入口.py", "🔍 租客入口"),
        ("pages/3_📊_後台分析.py", "📊 後台分析"),
    ]
    if hasattr(st, "page_link"):
        for path, label in links:
            st.page_link(path, label=label)
    else:
        for path, label in links:
            if st.button(label, key=f"nav_{path}", use_container_width=True):
                st.switch_page(path)
    st.divider()
