"""Builds full-body.html from the traced layers JSON."""
import json, sys
layers = json.load(open(sys.argv[1]))
defs, out = [], []
for n, l in enumerate(layers):
    if not l["d"]: continue
    fill = l["fill"]
    g = l.get("grad")
    if g:
        defs.append(f'<linearGradient id="g{n}" gradientUnits="userSpaceOnUse" x1="{g["from"][0]}" y1="{g["from"][1]}" x2="{g["to"][0]}" y2="{g["to"][1]}"><stop offset="0" stop-color="{g["c0"]}"/><stop offset="1" stop-color="{g["c1"]}"/></linearGradient>')
        fill = f"url(#g{n})"
    out.append(f'<path fill="{fill}" d="{l["d"]}"/>')
paths = "<defs>\n" + "\n".join(defs) + "\n</defs>\n" + "\n".join(out)
html = f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>الشخصية الكاملة</title>
<style>
  :root {{ --bg:#efe9df; --panel:#fff; --ink:#3a2a1e; --muted:#7a6a5c; --accent:#c4613a; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1d1a17; --panel:#2a2622; --ink:#f1e6d8; --muted:#b8a998; --accent:#e07a3e; }} }}
  :root[data-theme="dark"] {{ --bg:#1d1a17; --panel:#2a2622; --ink:#f1e6d8; --muted:#b8a998; --accent:#e07a3e; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Segoe UI",Tahoma,sans-serif; }}
  main {{ max-width:720px; margin:0 auto; padding:24px 16px 40px; }}
  h1 {{ font-size:1.4rem; margin:0 0 6px; }}
  p {{ margin:0 0 16px; color:var(--muted); }}
  .controls {{ display:flex; flex-wrap:wrap; gap:12px 24px; align-items:center; background:var(--panel); padding:12px 16px; border-radius:10px; margin-bottom:16px; }}
  label {{ display:flex; gap:8px; align-items:center; }}
  button {{ font:inherit; padding:8px 16px; border:0; border-radius:8px; background:var(--accent); color:#fff; cursor:pointer; }}
  .sheet {{ background:#fff; border-radius:14px; box-shadow:0 6px 24px rgba(0,0,0,.12); overflow:hidden; }}
  .sheet svg {{ display:block; width:100%; height:auto; }}
</style>
</head>
<body>
<main>
  <h1>الشخصية الكاملة — فيكتور عالي الدقة</h1>
  <p>كل مسار مستخرج من حدود الألوان في الصورة الأصلية ومطابق لها. استخدم الأدوات للمقارنة.</p>
  <div class="controls">
    <label><input type="checkbox" id="showRef"> إظهار الصورة الأصلية</label>
    <label>شفافية الرسم <input type="range" id="vecOp" min="0" max="100" value="100"></label>
    <button id="dl">تنزيل ملف SVG</button>
  </div>
  <div class="sheet">
<svg id="art" xmlns="http://www.w3.org/2000/svg" viewBox="700 40 222 476" role="img" aria-label="ولد بطاقية زرقاء وثوب برتقالي واقف">
<image id="ref" href="reference.jpg" x="0" y="0" width="1024" height="559" style="display:none"/>
<g id="vector">
<rect id="paper" x="700" y="40" width="222" height="476" fill="#fefefd"/>
{paths}
</g>
</svg>
  </div>
</main>
<script>
  var ref = document.getElementById('ref'), vec = document.getElementById('vector'), paper = document.getElementById('paper');
  function sync() {{
    var on = document.getElementById('showRef').checked;
    ref.style.display = on ? '' : 'none';
    paper.style.display = on ? 'none' : '';
    vec.style.opacity = document.getElementById('vecOp').value / 100;
  }}
  document.getElementById('showRef').addEventListener('change', sync);
  document.getElementById('vecOp').addEventListener('input', sync);
  document.getElementById('dl').addEventListener('click', function () {{
    var svg = document.getElementById('art').cloneNode(true);
    svg.querySelector('#ref').remove();
    svg.querySelector('#paper').remove();
    svg.querySelector('#vector').style.opacity = '';
    var blob = new Blob([svg.outerHTML], {{ type: 'image/svg+xml' }});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'full-body.svg'; a.click();
    URL.revokeObjectURL(a.href);
  }});
</script>
</body>
</html>
"""
open(sys.argv[2], "w", encoding="utf-8").write(html)
