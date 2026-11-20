"""把导出的 ONNX 模型内联进单文件 HTML，生成手机端一键基准页。

用法:
  python make_mobile_bench.py

依赖:
  export/saf.onnx, export/csc.onnx
产出:
  mobile_bench/AgriFilter_mobile_bench.html  (自包含，可直接在手机 Chrome 打开)
"""
import base64
import json
import shutil
from pathlib import Path

ROOT = Path(r"D:\agri_review_run")
EXP = ROOT / "export"
ORT = ROOT / "ort_dist"
OUT_DIR = ROOT / "mobile_bench"
TPL = ROOT / "scripts" / "mobile_bench_template.html"


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def main():
    saf = EXP / "saf.onnx"
    csc = EXP / "csc.onnx"
    for p in (saf, csc):
        if not p.exists():
            raise SystemExit(f"缺少模型文件: {p}")

    ort_js = (ORT / "ort.min.js").read_text(encoding="utf-8")
    # 防止 JS 里出现 </script> 提前闭合
    ort_js = ort_js.replace("</script", "<\\/script")

    tpl = TPL.read_text(encoding="utf-8")
    meta = {
        "saf_bytes": saf.stat().st_size,
        "csc_bytes": csc.stat().st_size,
        "env_label": "Chrome / Android · ONNX Runtime Web (WASM)",
    }
    html = (tpl
            .replace("__SAF_B64__", b64(saf))
            .replace("__CSC_B64__", b64(csc))
            .replace("__ORT_JS__", ort_js)
            .replace("__ORT_MJS_B64__", b64(ORT / "ort-wasm-simd-threaded.mjs"))
            .replace("__ORT_WASM_B64__", b64(ORT / "ort-wasm-simd-threaded.wasm"))
            .replace("__META_JSON__", json.dumps(meta, ensure_ascii=False)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "AgriFilter_mobile_bench.html"
    out.write_text(html, encoding="utf-8")

    # 附带模型原件，便于复核
    shutil.copy2(saf, OUT_DIR / "saf.onnx")
    shutil.copy2(csc, OUT_DIR / "csc.onnx")

    readme = f"""AgriFilter 端侧推理基准 —— 使用说明

【文件】
  AgriFilter_mobile_bench.html   自包含测试页面（两个模型已内联），{out.stat().st_size/1048576:.1f} MB
  saf.onnx / csc.onnx            导出的模型原件（仅供复核，测试本身不需要）

【准备】
  1. 完全离线可用：推理运行时与模型都已打包进这个 html，不需要联网。
  2. 必须用 Chrome 打开，不要用微信 / QQ 内置浏览器（它们不执行脚本）。

【把文件放到手机（任选一种）】
  A. USB：手机连电脑 -> 资源管理器把 html 复制到手机“Download”目录 -> 手机上“文件管理”里点开 -> 选 Chrome。
  B. 微信/QQ：发送给“文件传输助手” -> 手机端点开文件 -> 右上角“...” -> 用其他应用打开 -> Chrome。
  C. 阿里云盘：手机端云盘里直接下载，再用 Chrome 打开。

【测试】
  1. Chrome 打开页面 -> 点“开始测试”。
  2. 全程约 2 分钟（最后有 60 秒连续推理的电量参考测试），期间不要锁屏或切到后台。
  3. 结束后点“复制结果文本”，把文本框里的内容发回给我们就行（或截屏）。

【口径】
  - 运行时：ONNX Runtime Web（WASM / SIMD，单线程），与 PyTorch 逐元素最大偏差 max|Δ| <= 1.2e-7。
  - 每个模型：预热 10 次 + 计时 200 次，报告 mean / P50 / P95 / min。
"""
    (OUT_DIR / "使用说明.txt").write_text(readme, encoding="utf-8")

    print(f"[OK] {out}  ({out.stat().st_size/1048576:.2f} MB)")
    print(f"     SAF {meta['saf_bytes']/1048576:.2f} MB / CSC {meta['csc_bytes']/1048576:.2f} MB")


if __name__ == "__main__":
    main()
