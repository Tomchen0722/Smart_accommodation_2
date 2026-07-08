"""
房源圖片分析 — Image quality / clarity analysis.

判斷房源照片「清晰 / 尚可 / 模糊」，並用 SHAP（線性、精確解）解釋各影像特徵
對判斷的貢獻。核心清晰度指標為 Laplacian 變異數（經典失焦偵測法）。

設計考量（詳見 房源圖片分析說明.md）：
  • Laplacian 變異數 + 解析度 + 亮度 + 對比 + 邊緣密度 → 影像品質特徵向量。
  • 以可解釋的線性模型輸出清晰機率，SHAP 貢獻可分解每項特徵的推力。
  • CLIP 為「可選」語意判斷：若環境安裝了 torch/open_clip 才啟用；
    Streamlit Community Cloud 免費層記憶體有限，預設以輕量指標運作、結果穩定。
"""
import io
import re
import urllib.request

import numpy as np

# 特徵名稱（給 SHAP 圖表用）
FEAT_NAMES = ["清晰度 (Laplacian)", "解析度", "亮度適中", "對比", "邊緣密度"]

# 可解釋線性清晰度模型（權重越大代表越推向「清晰」）
_W0 = -3.0
_W = np.array([3.0, 0.8, 0.9, 0.7, 1.0])        # 對應 FEAT_NAMES
_BASE = np.array([0.5, 0.7, 0.85, 0.6, 0.5])    # SHAP 參考基準（中性圖片）


def fetch_image(url, timeout=8, max_side=900):
    """下載圖片並回傳 (PIL.Image RGB)；失敗回傳 None。"""
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        req = urllib.request.Request(
            str(url), headers={"User-Agent": "Mozilla/5.0 (SmartAccommodation)"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if max(img.size) > max_side:                # 限制尺寸以節省運算
            img.thumbnail((max_side, max_side))
        return img
    except Exception:
        return None


def _laplacian(gray):
    """3x3 Laplacian（純 numpy），回傳卷積結果。"""
    g = gray
    return (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
            - 4 * g[1:-1, 1:-1])


def extract_features(img):
    """回傳 (raw_dict, x_vector) — x_vector 為標準化後的 5 維特徵。"""
    rgb = np.asarray(img, dtype=float)
    h, w = rgb.shape[:2]
    gray = rgb.mean(axis=2)

    lap = _laplacian(gray)
    lap_var = float(lap.var())
    brightness = float(gray.mean()) / 255.0
    contrast = float(gray.std()) / 128.0
    edge_frac = float((np.abs(lap) > 12).mean())
    megapixels = (w * h) / 1e6

    raw = {
        "width": int(w), "height": int(h), "megapixels": round(megapixels, 2),
        "laplacian_var": round(lap_var, 1),
        "brightness": round(brightness, 3), "contrast": round(contrast, 3),
        "edge_density": round(edge_frac, 4),
    }
    x = np.array([
        np.clip(lap_var / 500.0, 0, 1.5),                    # 清晰度
        np.clip(megapixels / 0.8, 0, 1.5),                   # 解析度
        np.clip(1 - abs(brightness - 0.55) / 0.45, 0, 1.2),  # 亮度適中
        np.clip(contrast, 0, 1.2),                           # 對比
        np.clip(edge_frac * 10, 0, 1.2),                     # 邊緣密度
    ])
    return raw, x


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def classify(x):
    """回傳 (清晰機率 0-1, 標籤)。"""
    logit = _W0 + float(_W @ x)
    prob = float(_sigmoid(logit))
    if prob >= 0.60:
        label = "清晰"
    elif prob >= 0.40:
        label = "尚可"
    else:
        label = "模糊"
    return prob, label


def shap_contributions(x):
    """
    線性模型的精確 SHAP 值：φ_i = w_i · (x_i − baseline_i)。
    回傳 [(特徵名, 貢獻)]，正值＝推向清晰、負值＝推向模糊。
    """
    phi = _W * (np.asarray(x) - _BASE)
    return list(zip(FEAT_NAMES, [float(v) for v in phi]))


def clip_clarity(img):
    """
    可選：若環境安裝 torch + open_clip，用 CLIP 比對「清晰照片 vs 模糊照片」
    語意相似度。未安裝則回傳 None（App 改用輕量指標）。
    """
    try:
        import torch
        import open_clip
    except Exception:
        return None
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        prompts = ["a sharp, clear, well-lit photo of a room",
                   "a blurry, low-quality, dark photo"]
        with torch.no_grad():
            im = preprocess(img).unsqueeze(0)
            txt = tokenizer(prompts)
            imf = model.encode_image(im)
            tf = model.encode_text(txt)
            imf /= imf.norm(dim=-1, keepdim=True)
            tf /= tf.norm(dim=-1, keepdim=True)
            sims = (100.0 * imf @ tf.T).softmax(dim=-1)[0]
        return {"clear": float(sims[0]), "blurry": float(sims[1])}
    except Exception:
        return None


def analyze(url):
    """完整分析單一房源照片。回傳 dict（ok=False 表示無法下載）。"""
    img = fetch_image(url)
    if img is None:
        return {"ok": False}
    raw, x = extract_features(img)
    prob, label = classify(x)
    return {
        "ok": True, "prob": prob, "label": label,
        "raw": raw, "x": x.tolist(),
        "shap": shap_contributions(x),
        "clip": clip_clarity(img),
        "size": img.size,
    }


# ─── 通知信（示範用假資料） ──────────────────────────────────────
def fake_host_email(host_name, host_id):
    """房源無 email 欄位時，產生示範用假信箱。"""
    slug = re.sub(r"[^a-zA-Z0-9]", "", str(host_name or "host")).lower() or "host"
    return f"{slug}{int(host_id) % 1000 if str(host_id).isdigit() else ''}@host.demo"


def compose_email(listing_name, host_name, email, label, prob):
    """組出通知信內容（主旨、內文）。"""
    subject = "【智慧旅宿平台】您的房源照片建議重新上傳"
    body = (
        f"親愛的房東 {host_name} 您好，\n\n"
        f"系統偵測到您的房源「{listing_name}」封面照片清晰度偏低"
        f"（判定：{label}，清晰機率 {prob*100:.0f}%）。\n"
        f"模糊或昏暗的照片會顯著降低點閱與預訂率，建議您重新上傳一張"
        f"光線充足、對焦清晰的照片。\n\n"
        f"— 智慧旅宿平台 AI 圖片分析（本信為系統示範自動通知）"
    )
    return subject, body, email


def listing_photos(listing):
    """回傳該房源可用的照片 URL 清單（目前資料集每房源僅一張封面）。"""
    urls = []
    for col in ("picture_url",):
        u = str(listing.get(col, "") or "").strip()
        if u.startswith("http") and u not in urls:
            urls.append(u)
    return urls


# ─── 設施英文→繁中（供首頁詳情視窗使用；與租客頁一致）──────────────
_AMEN_ZH = [
    ("ac - split type ductless", "空調（分離式）"),
    ("heating - split type ductless", "暖氣（分離式）"),
    ("split type ductless", "分離式空調"),
    ("hot water kettle", "熱水壺"), ("hot water", "熱水"),
    ("hair dryer", "吹風機"), ("mini fridge", "小冰箱"),
    ("refrigerator", "冰箱"), ("freezer", "冷凍庫"),
    ("air conditioning", "空調"), ("ceiling fan", "吊扇"),
    ("portable fans", "電風扇"), ("heating", "暖氣"),
    ("smoke alarm", "煙霧偵測器"), ("carbon monoxide alarm", "一氧化碳偵測器"),
    ("fire extinguisher", "滅火器"), ("first aid kit", "急救箱"),
    ("exterior security cameras", "室外監視器"), ("window guards", "窗戶防護"),
    ("outlet covers", "插座保護蓋"), ("lock on bedroom door", "臥室門鎖"),
    ("lockbox", "密碼鎖箱"), ("keypad", "電子密碼鎖"),
    ("self check-in", "自助入住"), ("private entrance", "獨立入口"),
    ("hangers", "衣架"), ("shampoo", "洗髮精"), ("conditioner", "潤髮乳"),
    ("shower gel", "沐浴乳"), ("body soap", "肥皂"), ("essentials", "生活必需品"),
    ("bed linens", "床單"), ("extra pillows and blankets", "備用枕頭與棉被"),
    ("room-darkening shades", "遮光窗簾"), ("clothing storage", "衣物收納"),
    ("dishwasher", "洗碗機"), ("washer", "洗衣機"), ("dryer", "烘衣機"),
    ("drying rack", "曬衣架"), ("iron", "熨斗"), ("hdtv", "高畫質電視"),
    ("tv", "電視"), ("dedicated workspace", "專屬工作區"),
    ("ethernet connection", "有線網路"), ("dishes and silverware", "餐具"),
    ("cooking basics", "基本烹飪用品"), ("rice maker", "電鍋"),
    ("bread maker", "麵包機"), ("microwave", "微波爐"),
    ("gas stove", "瓦斯爐"), ("stove", "爐具"), ("oven", "烤箱"),
    ("dining table", "餐桌"), ("wine glasses", "酒杯"), ("kitchen", "廚房"),
    ("bathtub", "浴缸"), ("baby bath", "嬰兒澡盆"),
    ("patio or balcony", "陽台/露台"), ("balcony", "陽台"),
    ("elevator", "電梯"), ("single level home", "無樓梯平面住宅"),
    ("luggage dropoff allowed", "可寄放行李"), ("host greets you", "房東親自迎接"),
    ("long term stays allowed", "可長期入住"), ("laundromat nearby", "附近有自助洗衣"),
    ("cleaning available", "住宿期間可清潔"), ("cleaning products", "清潔用品"),
    ("paid parking", "付費停車"), ("free parking", "免費停車"), ("parking", "停車"),
    ("books and reading material", "書籍讀物"),
]


def zh_amenity(a):
    low = str(a).lower().strip()
    if low == "wifi":
        return "Wifi"
    for kw, zh in _AMEN_ZH:
        if kw in low:
            return zh
    return a


def amenities_zh(raw):
    """把 amenities 欄位（JSON 字串）解析並翻成繁中清單。"""
    import ast
    try:
        items = ast.literal_eval(raw) if isinstance(raw, str) else []
    except Exception:
        items = []
    return [zh_amenity(str(a)) for a in items if str(a).strip()]
