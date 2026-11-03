"""
专利侵权评估 - 可视化操作界面
支持设置线程数、测试数据量，运行后展示准确率等结果。
"""
import json
from pathlib import Path

import streamlit as st

from patent_agent import (
    run_agent,
    save_results,
    save_error_cases,
    evaluate_accuracy,
    load_json_dataset,
    normalize_label,
    ai_prediction_to_binary,
)


BASE_DIR = Path(__file__).resolve().parent
# 与 CLI 一致：优先用当前工作目录的数据文件（与直接运行 patent_agent.py 时行为一致）
_DATA_CANDIDATES = [
    Path.cwd() / "DPAS_RADAR_2.0.0-test.json",
    BASE_DIR / "DPAS_RADAR_2.0.0-test.json",
]
DATA_FILE = next((p for p in _DATA_CANDIDATES if p.exists()), _DATA_CANDIDATES[0])


def inject_css():
    st.markdown("""
    <style>
    /* 白色基调 */
    .stApp { background: #f8fafc; }
    h1, h2, h3, p, .stMarkdown { color: #1e293b !important; }
    .stSlider label { color: #475569 !important; }
    
    /* 侧边栏 */
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] .stMarkdown { color: #334155 !important; }
    
    /* 主标题区 */
    .hero { 
        text-align: center; 
        padding: 2rem 0 1.5rem; 
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #e2e8f0;
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .hero h1 { font-size: 2rem; font-weight: 700; color: #0f172a !important; }
    .hero p { color: #64748b !important; font-size: 1rem; }
    
    /* 指标卡片 */
    .metric-card {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #2563eb; }
    .metric-card .label { font-size: 0.85rem; color: #64748b; margin-top: 0.25rem; }
    .metric-card.accuracy .value { color: #059669; font-size: 2.5rem; }
    
    /* 运行按钮 */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }
    
    .stSuccess, .stInfo { border-radius: 8px; }
    .error-case-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    /* 错误案例图片区域：固定最大宽高，保持比例 */
    .error-case-img { max-width: 320px; max-height: 320px; object-fit: contain; margin: 0 auto; display: block; }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="专利侵权评估",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.markdown('<div class="hero"><h1>⚖️ 外观专利侵权评估</h1><p>设置线程数与测试数据量，一键运行并查看准确率</p></div>', unsafe_allow_html=True)

    # ---------- 侧边栏：参数 ----------
    with st.sidebar:
        st.header("📋 参数设置")
        st.markdown("---")
        sample_size = st.slider(
            "测试数据量（条）",
            min_value=10,
            max_value=1883,
            value=100,
            step=10,
            help="从测试集中取前 N 条进行评估",
        )
        max_workers = st.slider(
            "并发线程数",
            min_value=1,
            max_value=20,
            value=8,
            step=1,
            help="建议 5～10，过高可能触发 API 限流",
        )
        st.markdown("---")
        st.caption("数据文件：DPAS_RADAR_2.0.0-test.json")

    # ---------- 主区域：运行与结果 ----------
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        run_clicked = st.button("🚀 开始评估", type="primary", use_container_width=True)
    if "eval_results" in st.session_state:
        col_clear, _ = st.columns([1, 2])
        with col_clear:
            if st.button("清除结果", key="clear_results"):
                for k in ("eval_results", "eval_metrics"):
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

    if run_clicked:
        if not DATA_FILE.exists():
            st.error(f"测试集不存在：{DATA_FILE}")
            st.stop()
        progress_bar = st.progress(0, text="准备中…")
        status_text = st.empty()
        try:
            def on_progress(current, total):
                if total and total > 0:
                    progress_bar.progress(current / total, text=f"已完成 {current} / {total} 条")
                status_text.caption(f"已完成 {current} / {total} 条")
            results = run_agent(
                str(DATA_FILE),
                sample_size=sample_size,
                max_workers=max_workers,
                progress_callback=on_progress,
            )
            progress_bar.progress(1.0, text="计算准确率…")
            save_results(results)
            metrics = evaluate_accuracy(results)
            save_error_cases(results)
            st.session_state["eval_results"] = results
            st.session_state["eval_metrics"] = metrics
        except Exception as e:
            st.exception(e)
            st.stop()
        progress_bar.empty()
        status_text.empty()
        st.success("评估完成")
        st.caption("已生成 **error_cases.json**，包含本次所有预测与真实标签不一致的案例。")

    # 有缓存结果时展示（避免切换错误案例时因 rerun 丢失结果）
    if "eval_results" in st.session_state:
        results = st.session_state["eval_results"]
        metrics = st.session_state["eval_metrics"]
        acc = metrics.get("accuracy_percent", 0)
        precision = metrics.get("precision_percent", 0)
        recall = metrics.get("recall_percent", 0)
        f1 = metrics.get("f1_percent", 0)
        total = metrics.get("total", 0)
        correct = metrics.get("correct", 0)
        invalid = metrics.get("invalid", 0)
        invalid_label = metrics.get("invalid_label", 0)
        invalid_pred = metrics.get("invalid_pred", 0)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-card accuracy"><div class="value">{acc}%</div><div class="label">准确率</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card"><div class="value">{precision}%</div><div class="label">Precision(精确率)</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card"><div class="value">{recall}%</div><div class="label">Recall(召回率)</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="metric-card"><div class="value">{f1}%</div><div class="label">F1分数</div></div>',
                unsafe_allow_html=True,
            )

        # 第二行：样本规模与跳过情况
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(
                f'<div class="metric-card"><div class="value">{total}</div><div class="label">有效对比条数</div></div>',
                unsafe_allow_html=True,
            )
        with d2:
            st.markdown(
                f'<div class="metric-card"><div class="value">{correct}</div><div class="label">预测正确</div></div>',
                unsafe_allow_html=True,
            )
        with d3:
            st.markdown(
                f'<div class="metric-card"><div class="value">{invalid}</div><div class="label">跳过条数</div></div>',
                unsafe_allow_html=True,
            )
        if invalid > 0:
            st.caption(f"跳过原因：标签无法解析 {invalid_label} 条，预测无法解析（多为 API 报错/超时）{invalid_pred} 条")
            if invalid_pred > invalid_label:
                st.warning("「预测无法解析」较多时，多为并发请求导致 API 限流或超时。建议将**并发线程数**调低（如 5）后重新运行。")
        st.markdown("---")
        with st.expander("查看详细指标 JSON"):
            st.json(metrics)
        with st.expander("查看结果摘要（前 20 条）"):
            rows = [
                {
                    "product_id": r.get("product_id"),
                    "risk_level": r.get("risk_level"),
                    "similarity_score": r.get("similarity_score"),
                    "label": r.get("label"),
                }
                for r in results[:20]
            ]
            st.dataframe(rows, use_container_width=True)

        # 识别错误的案例
        error_cases = []
        for r in results:
            true_b = normalize_label(r.get("label"))
            pred_b = ai_prediction_to_binary(r.get("risk_level"), r.get("similarity_score"))
            if true_b is not None and pred_b is not None and true_b != pred_b:
                error_cases.append(r)
        if error_cases:
            st.subheader("识别错误的案例")
            st.caption(f"共 {len(error_cases)} 条预测与真实标签不一致，可选择查看产品图与专利图。")
            idx = st.selectbox(
                "选择案例序号",
                range(len(error_cases)),
                format_func=lambda i: f"案例 {i+1}: product_id={error_cases[i].get('product_id')} | 预测={error_cases[i].get('risk_level')} | 真实={'侵权' if normalize_label(error_cases[i].get('label'))==1 else '不侵权'}",
                key="error_case_select",
            )
            r = error_cases[idx]
            col_img_left, col_img_right = st.columns(2)
            # 固定显示宽度，保持比例，便于阅读
            img_width = 320
            with col_img_left:
                st.markdown("**商品图**")
                img_url = (r.get("product_image") or "").strip()
                if img_url and img_url != " ":
                    st.image(img_url, width=img_width)
                else:
                    st.caption("无图片 URL")
            with col_img_right:
                st.markdown("**专利图**")
                pat_url = (r.get("patent_image") or "").strip()
                if pat_url and pat_url != " ":
                    st.image(pat_url, width=img_width)
                else:
                    st.caption("无图片 URL")
            st.markdown("---")
            st.caption(f"商品标题: {r.get('product_title') or '-'}")
            st.caption(f"专利类别: {r.get('patent_prod') or '-'} | 相似度: {r.get('similarity_score')} | 原因: {r.get('reason') or '-'}")
        else:
            st.caption("本次无识别错误案例。")

    elif not run_clicked:
        st.info("👈 在左侧设置「测试数据量」和「并发线程数」，然后点击 **开始评估**。结果将显示准确率与统计信息。")


if __name__ == "__main__":
    main()
