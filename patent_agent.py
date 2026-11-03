import os
import json
import time
import threading
import base64
import mimetypes
import ssl
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from litellm import completion

# =============================
# .env (本地环境变量)
# =============================
def _load_env_file(env_path: str) -> None:
    """
    轻量级读取本地 .env 文件（不依赖第三方库）。
    仅处理形如 KEY=VALUE 的行，忽略空行与注释行（# ...）。
    """
    try:
        if not env_path:
            return
        if not os.path.exists(env_path):
            return
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip()
                val = v.strip().strip("'").strip('"')
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        # 读取 .env 失败时不阻断程序，后续会按环境变量是否存在来报错提示
        return


_DEFAULT_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
_load_env_file(os.getenv("ENV_FILE", _DEFAULT_ENV_PATH))

# =============================
# API CONFIG
# =============================

api_key = os.getenv("LITELLM_PROXY_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("缺少环境变量：LITELLM_PROXY_API_KEY。请在运行前先设置该环境变量。")
os.environ["LITELLM_PROXY_API_KEY"] = api_key
os.environ["LITELLM_PROXY_API_BASE"] = "https://litellm-gke-test.suntekcorps.com"

MODEL_NAME = "litellm_proxy/zeta_gpt-5-2025-08-07"
# 本地下载图片时，默认严格校验证书。若公司内网证书链异常，可临时设为 0 跳过校验。
VERIFY_IMAGE_SSL = os.getenv("VERIFY_IMAGE_SSL", "1").strip() not in ("0", "false", "False")
# 若严格校验失败且命中证书错误，自动降级到不校验再重试一次（默认开启）。
AUTO_SSL_FALLBACK = os.getenv("AUTO_SSL_FALLBACK", "1").strip() not in ("0", "false", "False")


# =============================

# LLM ANALYSIS
# =============================s
def _clean_optional_text(value):
    """清洗可能缺失/占位的文本字段（例如 'Null'）。"""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.lower() in ("null", "none", "n/a", "na", "nan"):
        return None
    return s


def _is_sketch_patent(is_sketch):
    """数据标注 is_sketch==1 表示专利图为素描/线稿。"""
    if is_sketch is None:
        return False
    if is_sketch is True:
        return True
    try:
        return int(is_sketch) == 1
    except (TypeError, ValueError):
        return str(is_sketch).strip().lower() in ("1", "true", "yes")


def _download_image_bytes(image_url, timeout=20, retries=2):
    """下载远程图片并返回 (bytes, mime)。失败时按次数重试。"""
    request = urllib.request.Request(
        image_url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/*,*/*;q=0.8",
        },
    )

    def _fetch_with_context(ctx):
        last_err = None
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout, context=ctx) as resp:
                    raw = resp.read()
                    content_type = resp.headers.get_content_type()
                    mime = content_type if content_type and content_type != "application/octet-stream" else None
                    return raw, mime
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(0.8 * (attempt + 1))
        raise last_err

    primary_ctx = ssl.create_default_context() if VERIFY_IMAGE_SSL else ssl._create_unverified_context()
    try:
        return _fetch_with_context(primary_ctx)
    except Exception as e:
        # 仅在“严格校验 + 证书错误 + 允许自动降级”时，退到不校验重试一次
        is_ssl_cert_error = isinstance(e, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in str(e)
        if VERIFY_IMAGE_SSL and AUTO_SSL_FALLBACK and is_ssl_cert_error:
            fallback_ctx = ssl._create_unverified_context()
            return _fetch_with_context(fallback_ctx)
        raise


def _image_source_to_data_url(image_source, timeout=20):
    """将本地路径/HTTP(S)图片转为 data:image/...;base64,..."""
    if image_source is None:
        raise ValueError("image_source is None")

    source = str(image_source).strip()
    if not source:
        raise ValueError("image_source is empty")

    if source.startswith("data:"):
        return source

    if source.startswith(("http://", "https://")):
        raw, mime = _download_image_bytes(source, timeout=timeout, retries=2)
        if not mime:
            mime = mimetypes.guess_type(source)[0] or "image/jpeg"
    else:
        local_path = source
        if source.startswith("file://"):
            local_path = urllib.parse.urlparse(source).path
        with open(local_path, "rb") as f:
            raw = f.read()
        mime = mimetypes.guess_type(local_path)[0] or "image/jpeg"

    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def analyze_patent(
    product_image,
    patent_image,
    product_title,
    patent_prod,
    *,
    is_sketch=None,
    patent_title=None,
):
    sketch = _is_sketch_patent(is_sketch)
    product_title_clean = _clean_optional_text(product_title)
    patent_title_clean = _clean_optional_text(patent_title)
    patent_prod_clean = _clean_optional_text(patent_prod)

    block_text_aux = f"""
========================
辅助文本信息（可能缺失，仅作辅助，不得替代视觉判断）
========================
商品标题（如有）:
{product_title_clean or "未提供"}

专利名称/标题（如有）:
{patent_title_clean or "未提供"}

专利产品类别（如有）:
{patent_prod_clean or "未提供"}
"""

    block_title_rules = """
========================
【标题辅助规则（必须遵守）】
========================
1. 商品标题/专利名称仅用于辅助判断：品类是否一致、使用场景是否相同、是否存在明显类别不匹配。
2. 若标题显示类别明显不同（例如一个是“宠物玩具”，另一个是“测距仪”），且视觉也未呈现同类整体印象 → 应倾向不侵权并降低相似度。
3. 禁止仅凭标题关键词或营销词直接推断侵权/不侵权；最终仍以图像的整体视觉与关键特征为准。
4. 当标题缺失或疑似噪声（如为 Null/乱码/泛化词），必须完全忽略标题信息。
"""

    if sketch:
        block_patent_type = """
专利附图类型（数据标注）:
本条专利图已标注为【素描/线稿】（is_sketch=1）。商品图多为彩色实物摄影，表现形式不同是正常现象，不得据此否定形状上的近似。
"""
        block_sketch_rules = """
========================
【素描专利图 — 强制比对规则】
========================
以下规则优先于对「画面风格不同」的直觉：
1. 禁止因专利图为黑白线稿、商品图为彩色实拍而压低 similarity_score。
2. 禁止将素描线条与摄影的光影、饱和度、反光差异当作「外观设计的本质不同」。
3. 应抽象比对：三维外轮廓、长宽高比例、组件排布、开孔/凹槽/折线位置；若上述一致或高度近似，仅商品有颜色、背景、包装、贴纸不同 → 通常应判定中高相似度（除非该专利的保护要点明确包含特定色彩或表面图案）。
"""
    else:
        block_patent_type = """
专利附图类型（数据标注）: 未标注为素描图（is_sketch≠1）。若仍存在线稿与实拍差异，单纯颜色或呈现方式一般不足以单独否定近似，须以形状与关键特征为准。
"""
        block_sketch_rules = ""

    block_simple_material = """
========================
【极简外形与材质（强制内部分析）】
========================
1. 先判断：产品是否仅为常见几何体或行业通用极简造型（如简单圆柱、薄板、无纹饰方盒、朴素圆环等）？若是 → 设计空间较大，仅凭粗轮廓或大体结构相似，不足以认定高风险。
2. 极简外形下必须评估：材质、表面肌理、光泽/雾面、透明度、金属/塑料/织物等带来的整体视觉印象。若材质与表面处理使普通消费者整体观感明显不同 → 应显著降低 similarity_score，倾向 Low。
3. 严禁在「外形极简」时仅因轮廓略像即判 High；须整体观感也高度接近才可高分。
4. 若外形含强识别性的非通用造型或显著装饰，则材质差异权重相对降低（与上条平衡适用）。
"""

    prompt = f"""
你是一名专业的外观专利侵权分析专家，具备法律判断与视觉分析能力。

你的任务是：判断“商品图片”是否与“专利设计”构成外观相同或近似，从而存在侵权风险。

========================
基础信息
========================
{block_text_aux}
{block_patent_type}

{block_sketch_rules}
{block_title_rules}
========================
分析原则（必须严格遵守）
========================

1. 【整体观察 + 综合判断原则】
不得仅基于局部细节判断，必须从整体视觉效果出发。

2. 【一般消费者视角】
假设观察者为普通消费者，而非专业设计师。

3. 【关键特征优先】
重点关注最影响整体视觉效果的设计特征。

4. 【功能性特征排除】
由功能决定的设计（如接口、结构标准件）应降低权重。

5. 【设计空间原则】
- 若该类产品设计空间小 → 微小差异也重要
- 若设计空间大 → 需较大差异才不构成侵权

6. 【可见性原则】
以产品正常使用时可见部分为主。

7. 【现有设计影响】
常见设计元素应降低权重。

========================
分析维度（必须覆盖）
========================

请在内部分析时综合考虑以下维度（不要逐条输出，只用于推理）：

1. 整体形状结构
   - 外轮廓（方形/圆形/异形）
   - 比例关系（长宽高）
   - 结构布局

2. 关键设计特征（最重要）
   - 主要视觉区域
   - 标志性设计元素
   - 是否形成视觉记忆点

3. 视觉风格
   - 极简 / 工业 / 装饰性 / 科技感
   - 线条风格（圆润 vs 棱角）

4. 同类产品判断
   - 是否属于相同或类似产品类别
   - 使用场景是否一致

5. 差异分析
   - 是否存在显著差异（结构/比例/布局）
   - 差异是否足以改变整体视觉效果

6. 设计空间评估
   - 当前类别设计自由度高 or 低
   - 调整相似度判断标准

========================
特殊情况处理（必须考虑）
========================

- 如果仅存在细节修改，但整体视觉高度一致 → 仍视为高风险
- 如果仅比例缩放或轻微变形 → 通常仍视为相似
- 如果只是颜色不同 → 一般不影响判断
- 如果整体风格高度一致，即使细节不同 → 可能构成近似设计
- 如果差异集中在非显著区域 → 降低影响
- 如果产品属于高度标准化结构 → 提高判定严格度
- 如果产品整体设计简单且一致，但材质完全不同 → 不侵权（极简外形时材质/表面观感权重更高）
- 若已标注素描专利图：结构近似时勿因彩色商品图而漏判为不侵权

{block_simple_material}
========================
请务必避免以下所有错误认知，确保判断准确无误。
========================

【高风险误判来源】

1. 功能性结构误判（最严重）
2. 标准件/工业件误判
3. 仅因结构相似就判侵权（外形极简时尤须结合材质与整体观感）
4. 忽略类别差异

【高风险漏判来源】

5. 过度放大局部差异（含误把素描vs彩色实拍当成设计差异）
6. 忽略整体视觉一致性
7. 错误识别关键设计特征

【系统性认知错误】

8. 设计空间逻辑反转（必须修正）
9. 局部相似 ≠ 整体相似
10. 差异存在 ≠ 不侵权

========================
评分与风险映射规则（必须使用）
========================

请基于“整体视觉相似度”进行评分：

- 90-100 → 几乎相同（极高风险）
- 70-89  → 高度近似（高风险）
- 50-69  → 中等相似（中风险）
- 30-49  → 低相似（低风险）
- 0-29   → 明显不同（低风险）

风险等级映射：
- High   → similarity_score ≥ 70
- Medium → 50 ≤ similarity_score < 70
- Low    → similarity_score < 50

========================
输出要求（严格执行）
========================

你必须仅输出 JSON，不能包含任何额外说明、解释或多余文字。

输出格式如下：

{{
 "similarity_score": 整数（0-100）,
 "risk_level": "Low" 或 "Medium" 或 "High",
 "reason": "用1-2句话说明核心判断依据（强调整体视觉和关键特征）"
}}

注意：
- reason 必须简洁，但要体现“整体视觉 + 关键特征 + 差异影响”
- 不要输出分析过程
- 不要输出多余字段
"""
    product_image_data_url = _image_source_to_data_url(product_image)
    patent_image_data_url = _image_source_to_data_url(patent_image)

    response = completion(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=3000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},

                    {
                        "type": "image_url",
                        "image_url": {"url": product_image_data_url}
                    },

                    {
                        "type": "image_url",
                        "image_url": {"url": patent_image_data_url}
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content


# =============================
# LOAD DATASET
# =============================

def load_json_dataset(json_path):

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} 不存在")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Dataset loaded: {len(data)} samples")

    return data


# =============================
# PARSE LLM OUTPUT
# =============================

def parse_llm_output(text):

    try:
        result = json.loads(text)
        return result
    except:
        return {
            "similarity_score": None,
            "risk_level": "Unknown",
            "reason": text
        }


# =============================
# MAIN AGENT（多线程）
# =============================

def _process_one_sample(item, index, print_lock):
    """处理单条样本，返回 (index, result_dict)。供线程池调用。"""
    product_title = _clean_optional_text(item.get("product_title", ""))
    patent_prod = _clean_optional_text(item.get("patent_prod", ""))
    product_image = item.get("product_image")
    patent_image = item.get("patent_image")
    label = item.get("label")
    is_sketch = item.get("is_sketch")
    patent_title = _clean_optional_text(item.get("patent_title") or item.get("patent_name"))

    with print_lock:
        print("=" * 60)
        print("Patent:", patent_prod)
        print(f"Sample {index + 1}")
        title_for_print = product_title or ""
        print("Product:", (title_for_print[:60] + "...") if len(title_for_print) > 60 else title_for_print)
        print("Start LLM request...")

    start = time.time()
    try:
        llm_output = analyze_patent(
            product_image,
            patent_image,
            product_title,
            patent_prod,
            is_sketch=is_sketch,
            patent_title=patent_title,
        )
        elapsed = round(time.time() - start, 2)
        parsed = parse_llm_output(llm_output)
        with print_lock:
            print(f"Sample {index + 1} finished in {elapsed}s ->", parsed.get("risk_level"), parsed.get("similarity_score"))
    except Exception as e:
        parsed = {
            "similarity_score": None,
            "risk_level": "Error",
            "reason": str(e)
        }
        with print_lock:
            print(f"Sample {index + 1} LLM Error:", e)

    result = {
        "product_id": item.get("product_id_asin"),
        "patent_id": item.get("global_patent_id"),
        "similarity_score": parsed["similarity_score"],
        "risk_level": parsed["risk_level"],
        "reason": parsed["reason"],
        "label": label,
        "product_image": product_image,
        "patent_image": patent_image,
        "product_title": product_title,
        "patent_prod": patent_prod,
        "is_sketch": is_sketch,
        "patent_title": patent_title,
    }
    return (index, result)


def run_agent(json_path, sample_size=5, max_workers=5, progress_callback=None):
    """
    多线程跑完样本。max_workers 为并发线程数，建议 5~10，过大可能触发 API 限流。
    progress_callback(current, total) 可选，每完成一条会调用一次，用于 UI 进度展示。
    """
    dataset = load_json_dataset(json_path)
    sample_size = min(sample_size, len(dataset))
    print(f"\nRunning {sample_size} samples (threads={max_workers})\n")

    results_by_index = {}
    print_lock = threading.Lock()
    total_start = time.time()
    done_count = [0]  # 用 list 以便在闭包中修改

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one_sample, item, i, print_lock): i
            for i, item in enumerate(dataset[:sample_size])
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results_by_index[idx] = result
            done_count[0] += 1
            if progress_callback:
                try:
                    progress_callback(done_count[0], sample_size)
                except Exception:
                    pass

    results = [results_by_index[i] for i in range(sample_size)]
    print(f"\nAll {sample_size} samples done in {round(time.time() - total_start, 2)}s")
    return results


# =============================
# SAVE RESULTS
# =============================

def save_results(results, file_name="results.json"):

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {file_name}")


# =============================
# EVALUATE ACCURACY（AI 预测 vs 真实标签）
# =============================

def normalize_label(label):
    """将 JSON 中的 label 统一为 0（不侵权）或 1（侵权）。label=2 表示侵权。"""
    if label is None:
        return None
    if label in (1, 2, True, "1", "2", "true", "True", "yes", "Yes", "侵权", "infringement", "正"):
        return 1
    if label in (0, False, "0", "false", "False", "no", "No", "不侵权", "no_infringement", "负"):
        return 0
    s = str(label).strip().lower()
    if s in ("1", "2", "true", "yes", "侵权", "infringement"):
        return 1
    if s in ("0", "false", "no", "不侵权", "no_infringement"):
        return 0
    return None


def ai_prediction_to_binary(risk_level, similarity_score=None):
    """High/Medium -> 侵权(1)，Low -> 不侵权(0)。"""
    if risk_level is None:
        return None
    r = str(risk_level).strip()
    if r in ("High", "Medium"):
        return 1
    if r == "Low":
        return 0
    if similarity_score is not None and isinstance(similarity_score, (int, float)):
        return 1 if similarity_score >= 50 else 0
    return None


def evaluate_accuracy(results):
    """对比 AI 预测与真实 label，计算准确率/Precision/Recall/F1 并打印报告。"""
    correct = 0
    total = 0
    # 二分类混淆矩阵（以 1=侵权 为正类）
    tp = 0  # 真侵权，预测侵权
    fp = 0  # 真不侵权，预测侵权
    tn = 0  # 真不侵权，预测不侵权
    fn = 0  # 真侵权，预测不侵权
    invalid = 0
    invalid_label = 0   # 因标签无法解析而跳过
    invalid_pred = 0    # 因预测无法解析而跳过（多为 API 报错/超时）
    for r in results:
        true_binary = normalize_label(r.get("label"))
        pred_binary = ai_prediction_to_binary(
            r.get("risk_level"),
            r.get("similarity_score")
        )
        if true_binary is None or pred_binary is None:
            invalid += 1
            if true_binary is None:
                invalid_label += 1
            if pred_binary is None:
                invalid_pred += 1
            continue
        total += 1
        if pred_binary == 1 and true_binary == 1:
            tp += 1
        elif pred_binary == 1 and true_binary == 0:
            fp += 1
        elif pred_binary == 0 and true_binary == 1:
            fn += 1
        else:  # pred_binary == 0 and true_binary == 0
            tn += 1

        if pred_binary == true_binary:
            correct += 1
    accuracy = (correct / total * 100) if total else 0.0
    precision = (tp / (tp + fp) * 100) if (tp + fp) else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    print("\n" + "=" * 60)
    print("AI 评估报告（预测 vs 真实是否侵权）")
    print("=" * 60)
    print(f"有效对比条数: {total}")
    print(f"跳过条数: {invalid} (标签无法解析: {invalid_label}, 预测无法解析: {invalid_pred})")
    print(f"预测正确: {correct}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"精确率(Precision): {precision:.2f}%")
    print(f"召回率(Recall): {recall:.2f}%")
    print(f"F1分数(F1): {f1:.2f}%")
    print("=" * 60)
    return {
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "invalid_label": invalid_label,
        "invalid_pred": invalid_pred,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy_percent": round(accuracy, 2),
        "precision_percent": round(precision, 2),
        "recall_percent": round(recall, 2),
        "f1_percent": round(f1, 2),
    }


def collect_error_cases(results):
    """
    筛选预测与真实标签不一致的案例（与 evaluate_accuracy 判定规则一致）。
    返回列表，每项含原始字段及 true_binary / predicted_binary 说明。
    """
    errors = []
    for r in results:
        true_b = normalize_label(r.get("label"))
        pred_b = ai_prediction_to_binary(r.get("risk_level"), r.get("similarity_score"))
        if true_b is None or pred_b is None:
            continue
        if true_b == pred_b:
            continue
        errors.append({
            "true_binary": true_b,
            "predicted_binary": pred_b,
            "true_label_text": "侵权" if true_b == 1 else "不侵权",
            "predicted_label_text": "侵权" if pred_b == 1 else "不侵权",
            "raw_label": r.get("label"),
            "product_id": r.get("product_id"),
            "patent_id": r.get("patent_id"),
            "product_title": r.get("product_title"),
            "patent_prod": r.get("patent_prod"),
            "risk_level": r.get("risk_level"),
            "similarity_score": r.get("similarity_score"),
            "reason": r.get("reason"),
            "product_image": r.get("product_image"),
            "patent_image": r.get("patent_image"),
        })
    return errors


def save_error_cases(results, file_name="error_cases.json"):
    """将错误案例写入 JSON 文件。"""
    errors = collect_error_cases(results)
    payload = {
        "count": len(errors),
        "description": "预测与真实标签不一致的案例（true_binary: 0=不侵权, 1=侵权；与 evaluate_accuracy 规则一致）",
        "cases": errors,
    }
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n错误案例已保存至 {file_name}（共 {len(errors)} 条）")
    return payload


# =============================
# ENTRY
# =============================

if __name__ == "__main__":
    json_file = "DPAS_RADAR_2.0.0-test.json"
    results = run_agent(
        json_file,
        sample_size=10,
        max_workers=1,
    )
    save_results(results)
    metrics = evaluate_accuracy(results)
    with open("eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("评估指标已保存至 eval_metrics.json")
    save_error_cases(results)
