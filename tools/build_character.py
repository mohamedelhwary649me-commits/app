"""Generates character.html: a hand-traced SVG of reference.jpg.
Each head is drawn in local coords: origin = top-center of the cap."""
import math

def zig(x0, x1, step, amp, yfun):
    pts, up = [], True
    x = x0
    while x <= x1 + 1e-6:
        pts.append(f"{x:.1f},{yfun(x) + (-amp if up else amp):.1f}")
        up = not up
        x += step
    return "M" + " L".join(pts)

band_top = lambda x: 22 + 10 * (x / 53) ** 2
band_mid = lambda x: 30 + 9 * (x / 53) ** 2
ZIG = zig(-50, 50, 5, 2.2, band_mid)
BANDTOP = "M" + " L".join(f"{x},{band_top(x):.1f}" for x in range(-53, 54, 4))

defs = f"""
<linearGradient id="gCap" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="#94bcc6"/><stop offset="1" stop-color="#7aa9b7"/></linearGradient>
<linearGradient id="gRobe" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#c9663a"/><stop offset=".5" stop-color="#c4613a"/><stop offset="1" stop-color="#a94a2e"/></linearGradient>
<linearGradient id="gRobeV" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="#c8673b"/><stop offset="1" stop-color="#b65533"/></linearGradient>
<radialGradient id="gFace" cx="45%" cy="55%" r="60%">
  <stop offset="0" stop-color="#f6aa7c"/><stop offset=".75" stop-color="#f09e70"/><stop offset="1" stop-color="#e48e62"/></radialGradient>
<radialGradient id="gCheek"><stop offset="0" stop-color="#e5715a" stop-opacity=".55"/><stop offset="1" stop-color="#e5715a" stop-opacity="0"/></radialGradient>
<radialGradient id="gShadow"><stop offset="0" stop-color="#e6cfc2"/><stop offset="1" stop-color="#e6cfc2" stop-opacity="0"/></radialGradient>

<g id="ear">
  <path d="M-45,88 C-55,82 -64,90 -62,101 C-61,111 -54,118 -45,115Z" fill="#ee9c6e"/>
  <path d="M-48,95 C-54,93 -57,99 -56,104 C-55,109 -51,111 -48,109" fill="none" stroke="#d27c52" stroke-width="2.4" stroke-linecap="round"/>
</g>

<g id="head">
  <!-- back hair -->
  <path d="M-57,40 C-62,52 -66,64 -65,74 C-63,86 -56,96 -47,100 L47,100 C55,95 60,86 61,76 C62,64 60,52 56,40Z" fill="#232637"/>
  <path d="M-55,42 C-61,41 -67,44 -70,48 C-65,47 -61,48 -57,51Z" fill="#232637"/>
  <path d="M-63,62 C-66,68 -67,74 -66,80 C-64,74 -62,70 -60,68Z" fill="#232637"/>
  <path d="M57,52 C62,58 64,66 64,73 C61,68 58,65 55,63Z" fill="#232637"/>
  <use href="#ear"/><use href="#ear" transform="scale(-1,1)"/>
  <!-- face -->
  <path d="M-46,68 C-47,94 -44,114 -36,127 C-26,139 -12,144 0,144 C12,144 26,139 36,127 C44,114 47,94 46,68 C40,52 -40,52 -46,68Z" fill="url(#gFace)"/>
  <path d="M-45,90 C-44,112 -36,128 -22,138 C-34,124 -40,108 -40,90Z" fill="#e28a5f" opacity=".35"/>
  <ellipse cx="-30" cy="106" rx="13" ry="9" fill="url(#gCheek)"/>
  <ellipse cx="30" cy="106" rx="13" ry="9" fill="url(#gCheek)"/>
  <!-- shadow under fringe -->
  <path d="M-44,76 C-40,72 -30,70 -18,66 C-8,68 0,69 4,70 C10,62 20,56 27,52 C33,60 38,68 41,76 C38,72 34,66 28,60 C20,64 10,70 4,74 C-4,72 -14,70 -20,72 C-30,74 -38,76 -44,80Z" fill="#d9825a" opacity=".45"/>
  <!-- fringe -->
  <path d="M-58,42 C-61,60 -57,82 -47,99 C-46,92 -45,86 -44,81 C-43,75 -40,71 -36,70 C-31,68 -27,67 -24,66 C-21,64 -19,62 -17,60 C-10,62 -4,63 3,66 C7,59 12,55 17,53 C21,50 25,47 27,45 C30,49 32,52 34,55 C38,61 40,66 41,70 C43,79 45,88 46,97 L50,93 C56,86 59,77 59,68 C60,58 58,49 56,41 C40,33 -40,33 -58,42Z" fill="#232637"/>
  <path d="M-38,50 C-30,52 -24,58 -21,62 M-20,46 C-10,50 -4,56 0,62 M6,44 C12,46 16,50 18,54 M30,46 C38,52 44,60 48,70 M-50,56 C-52,64 -50,74 -46,82" fill="none" stroke="#3b3f57" stroke-width="2" stroke-linecap="round"/>
  <!-- cap -->
  <path d="M-57,45 C-58,31 -55,22 -50,15 C-38,3 -18,0 0,0 C18,0 38,3 50,15 C55,22 58,31 57,45 C40,41 -40,41 -57,45Z" fill="url(#gCap)"/>
  <path d="M-57,45 C-58,36 -57,28 -54,24 C-30,16 30,16 54,24 C57,28 58,36 57,45 C40,41 -40,41 -57,45Z" fill="#7eadb9"/>
  <path d="{BANDTOP}" fill="none" stroke="#6a9aa8" stroke-width="1.4"/>
  <path d="{ZIG}" fill="none" stroke="#a6cbd3" stroke-width="1.4" stroke-linejoin="round"/>
  <path d="M-57,45 C-40,41 40,41 57,45" fill="none" stroke="#6d9dab" stroke-width="1.6"/>
  <path d="M-34,8 C-20,3 20,3 34,8" fill="none" stroke="#a8ccd4" stroke-width="1.4" opacity=".8"/>
  <path d="M-46,20 C-40,12 -30,7 -20,5" fill="none" stroke="#b5d4db" stroke-width="3" stroke-linecap="round" opacity=".5"/>
</g>

<g id="nose">
  <path d="M-3,101 C-6,106 -9,109 -7,112 C-4,115 3,115 6,112" fill="none" stroke="#d47a50" stroke-width="2.2" stroke-linecap="round"/>
  <ellipse cx="0" cy="110" rx="5" ry="3.4" fill="#f6b48a" opacity=".7"/>
</g>

<!-- open eye looking forward, centered at 0,0 -->
<g id="eyeOpen">
  <ellipse rx="11.5" ry="11" fill="#fff"/>
  <circle cx="1" cy="1" r="8.6" fill="#1e2537"/>
  <circle cx="1" cy="2.5" r="6" fill="#2c3a55" opacity=".6"/>
  <circle cx="4" cy="-2.5" r="3" fill="#fff"/>
  <circle cx="-2.5" cy="4.5" r="1.3" fill="#fff" opacity=".8"/>
  <path d="M-12.5,-1 C-10,-10 8,-12 13,-3" fill="none" stroke="#1b1820" stroke-width="3" stroke-linecap="round"/>
</g>

<g id="bust">
  <path d="M-21,132 L-21,154 L21,154 L21,132Z" fill="#ec9a6c"/>
  <path d="M-21,136 C-10,146 10,146 21,136 L21,146 C10,152 -10,152 -21,146Z" fill="#d9825a" opacity=".7"/>
  <path d="M-57,180 C-55,168 -42,160 -25,155 L25,155 C42,160 55,168 57,180 C40,185 -40,185 -57,180Z" fill="url(#gRobe)"/>
  <path d="M-25,155 C-32,160 -36,170 -34,183 L-40,183 C-42,170 -36,160 -25,155Z" fill="#b5532f" opacity=".5"/>
  <!-- chest embroidery yoke -->
  <path d="M-26,156 C-28,164 -24,172 -14,176 L-4,178 L-4,160Z" fill="#c96d42"/>
  <path d="M26,156 C28,164 24,172 14,176 L4,178 L4,160Z" fill="#c96d42"/>
  <path d="M-23,160 C-22,167 -18,171 -9,173 M23,160 C22,167 18,171 9,173" fill="none" stroke="#e8a66c" stroke-width="1.6" stroke-dasharray="2.5 1.8"/>
  <!-- collar stand -->
  <path d="M-23,146 C-14,151 -6,153 -2,154 L-2,163 C-10,161 -18,157 -25,152Z" fill="#c4683f"/>
  <path d="M23,146 C14,151 6,153 2,154 L2,163 C10,161 18,157 25,152Z" fill="#c4683f"/>
  <path d="M-21,150 C-14,154 -8,156 -4,157 M21,150 C14,154 8,156 4,157" fill="none" stroke="#eab07a" stroke-width="1.5" stroke-dasharray="2 1.6"/>
  <!-- placket -->
  <path d="M-5,157 L5,157 L5,184 L-5,184Z" fill="#9c452c"/>
  <path d="M-8,160 L-8,184 M8,160 L8,184" stroke="#e4a067" stroke-width="2" stroke-dasharray="3 2"/>
  <path d="M0,157 L0,184" stroke="#7c331f" stroke-width="1.2"/>
</g>
"""

def brows(l, r, w=4.2):
    return f'<path d="{l} {r}" fill="none" stroke="#232637" stroke-width="{w}" stroke-linecap="round"/>'

LINE = 'fill="none" stroke="#1b1820" stroke-linecap="round"'
MOUTH = 'fill="none" stroke="#a24a33" stroke-linecap="round"'

heads = []

# 1 laughing
heads.append(("1", "translate(192.5,60)", "", f"""
  <path d="M-33,99 Q-22,86 -10,98" {LINE} stroke-width="3.6"/>
  <path d="M15,96 Q27,84 39,95" {LINE} stroke-width="3.6"/>
  {brows("M-41,78 Q-29,70 -17,74", "M10,72 Q22,65 35,72")}
  <use href="#nose"/>
  <path d="M-23,117 Q0,121 23,115 Q18,139 0,140 Q-18,139 -23,117Z" fill="#8a2a2c"/>
  <path d="M-21,118 Q0,122 21,116 L19,123 Q0,127 -19,124Z" fill="#fff"/>
  <path d="M-11,135 Q0,126 12,133 Q6,140 0,140 Q-7,140 -11,135Z" fill="#e5645a"/>
"""))

# 3 surprised
heads.append(("3", "translate(580,60)", "", f"""
  <g transform="translate(-22,96) scale(1.13)"><use href="#eyeOpen"/></g>
  <g transform="translate(25,95) scale(1.13)"><use href="#eyeOpen"/></g>
  {brows("M-40,76 Q-28,66 -15,71", "M14,69 Q26,63 37,71")}
  <use href="#nose"/>
  <ellipse cx="2" cy="129" rx="7.5" ry="8.5" fill="#8a2a2c"/>
  <path d="M-4,124 Q2,120 8,124 L7,127 Q2,125 -3,127Z" fill="#fff"/>
  <ellipse cx="2" cy="133" rx="4.5" ry="3" fill="#e5645a"/>
"""))

# 4 calm
heads.append(("4", "translate(195,304)", "", f"""
  <path d="M-36,96 Q-25,104 -13,96 M14,96 Q25,104 36,96" {LINE} stroke-width="3.2"/>
  <path d="M-36,96 l-3,-2 M36,96 l3,-2" {LINE} stroke-width="2.4"/>
  {brows("M-38,80 Q-28,70 -14,76", "M14,76 Q28,70 38,80")}
  <use href="#nose"/>
  <path d="M-16,117 Q0,128 17,117" {MOUTH} stroke-width="2.8"/>
"""))

# 5 angry
heads.append(("5", "translate(392.5,304)", "", f"""
  <g transform="translate(-24,93)"><use href="#eyeOpen"/></g>
  <g transform="translate(24,93)"><use href="#eyeOpen"/></g>
  <path d="M-38,78 L-9,88 L-9,82 L-40,74Z M38,78 L9,88 L9,82 L40,74Z" fill="#f09e70"/>
  <path d="M-41,77 Q-26,77 -10,87 M10,87 Q26,77 41,77" fill="none" stroke="#232637" stroke-width="5.5" stroke-linecap="round"/>
  <use href="#nose"/>
  <path d="M-11,124 Q0,117 11,124" {MOUTH} stroke-width="2.8"/>
"""))

# 6 wink
heads.append(("6", "translate(585.5,304)", "", f"""
  <path d="M-37,94 Q-26,84 -14,93" {LINE} stroke-width="3.4"/>
  <path d="M-37,94 l-3,2" {LINE} stroke-width="2.4"/>
  <g transform="translate(25.5,89)"><use href="#eyeOpen"/></g>
  {brows("M-37,80 Q-24,74 -10,82", "M11,76 Q26,64 40,76")}
  <use href="#nose"/>
  <path d="M-21,112 Q2,126 25,109" {MOUTH} stroke-width="2.8"/>
  <path d="M23,107 Q26,110 27,113" {MOUTH} stroke-width="2.2"/>
"""))

def head_svg(n, tr, extra, feats):
    return f'<g id="h{n}" transform="{tr}"><use href="#bust"/><use href="#head"/>{feats}</g>'

body = "".join(head_svg(*h) for h in heads)

# 2 thinking (custom pose)
think = f"""
<g id="h2" transform="translate(397.5,59)">
  <use href="#bust"/>
  <path d="M30,165 C44,168 54,176 58,190 L40,196Z" fill="#a94a2e"/>
  <g transform="translate(-17,-5) rotate(8 0 90)">
    <use href="#head"/>
    <g transform="translate(-10,-2)">
      <g transform="translate(-24,94)"><ellipse rx="11" ry="11.5" fill="#fff"/><circle cx="3" cy="-2" r="8" fill="#1e2537"/><circle cx="6" cy="-5" r="2.8" fill="#fff"/>
        <path d="M-12,-1 C-10,-10 8,-12 12,-3" {LINE} stroke-width="3"/></g>
      <g transform="translate(22,94)"><ellipse rx="11" ry="11.5" fill="#fff"/><circle cx="3" cy="-2" r="8" fill="#1e2537"/><circle cx="6" cy="-5" r="2.8" fill="#fff"/>
        <path d="M-12,-3 C-8,-12 10,-10 12,-1" {LINE} stroke-width="3"/></g>
      {brows("M-37,80 Q-28,74 -16,72", "M10,73 Q24,74 34,82")}
      <g transform="translate(-2,4)"><use href="#nose"/></g>
      <path d="M-12,124 Q-4,119 4,124" {MOUTH} stroke-width="2.6"/>
    </g>
  </g>
  <!-- hand under chin -->
  <path d="M-37,135 C-34,128 -22,127 -10,132 C-4,135 -4,141 -10,142 C-4,146 -6,152 -12,152 C-8,156 -12,162 -18,160 L-16,186 L-38,186 L-40,160 C-44,152 -42,140 -37,135Z" fill="#ee9c6e"/>
  <path d="M-30,140 C-22,138 -14,140 -10,142 M-28,150 C-22,149 -16,150 -12,152 M-36,135 C-30,132 -20,131 -12,133" fill="none" stroke="#d27c52" stroke-width="1.8" stroke-linecap="round"/>
  <!-- open book -->
  <path d="M-80,156 L-28,182 L-24,214 L-80,204Z" fill="#2c8086"/>
  <path d="M-77,152 L-28,177 L-28,184 L-77,159Z" fill="#f6ecd6"/>
  <path d="M-74,156 L-30,179" stroke="#d8cbb0" stroke-width="1"/>
  <path d="M-28,177 L40,163 L42,169 L-28,184Z" fill="#fbf3e2"/>
  <path d="M-28,182 L42,168 L48,208 L-24,216Z" fill="#3a9aa0"/>
  <path d="M-28,182 L42,168 L48,208 L-24,216Z" fill="none" stroke="#25686e" stroke-width="1.6"/>
  <path d="M-18,186 L36,175 L40,204 L-15,211Z" fill="none" stroke="#d6b062" stroke-width="1.3"/>
  <circle cx="24" cy="184" r="5" fill="#d9a14b" opacity=".8"/><circle cx="-6" cy="196" r="4" fill="#d9a14b" opacity=".6"/>
  <path d="M-28,182 L-24,216" stroke="#1f6166" stroke-width="2.4"/>
  <text transform="translate(12,203) rotate(-11)" font-size="15" font-weight="700" fill="#fff6dc" text-anchor="middle" font-family="Tahoma, sans-serif">قصص</text>
  <path d="M-86,196 C-88,186 -80,180 -72,184 L-70,200 C-76,204 -84,202 -86,196Z" fill="#ee9c6e"/>
</g>"""

full = f"""
<g id="full">
  <ellipse cx="810" cy="500" rx="110" ry="11" fill="url(#gShadow)"/>
  <!-- feet & sandals -->
  <path d="M728,496 C728,500 786,502 786,497 L784,492 L730,492Z" fill="#7a3e2c"/>
  <path d="M736,492 C730,482 738,470 752,469 L770,469 C780,474 785,484 783,493Z" fill="#f09c6e"/>
  <path d="M742,486 C752,482 766,482 778,486" fill="none" stroke="#7a3e2c" stroke-width="3" stroke-linecap="round"/>
  <path d="M888,496 C888,500 830,502 830,497 L832,492 L886,492Z" fill="#7a3e2c"/>
  <path d="M880,492 C886,482 878,470 864,469 L846,469 C836,474 831,484 833,493Z" fill="#f09c6e"/>
  <path d="M874,486 C864,482 850,482 838,486" fill="none" stroke="#7a3e2c" stroke-width="3" stroke-linecap="round"/>
  <!-- robe body -->
  <path d="M762,190 C756,204 752,224 750,250 L739,471 C770,478 846,478 877,471 L866,250 C864,224 860,204 854,190 C842,181 828,177 816,176 L800,176 C788,177 774,181 762,190Z" fill="url(#gRobe)"/>
  <path d="M752,240 C750,320 745,400 740,470 L752,472 C756,400 760,320 760,250Z" fill="#d07040" opacity=".5"/>
  <path d="M846,262 C852,330 858,400 862,472 L877,471 L866,250Z" fill="#9c432a" opacity=".45"/>
  <path d="M790,300 C786,360 782,420 778,474 M826,320 C830,380 834,430 836,474 M808,330 C812,380 814,430 812,476" fill="none" stroke="#a7472c" stroke-width="1.6" opacity=".55"/>
  <!-- sleeves -->
  <path d="M764,188 C750,194 742,212 737,236 L716,312 C724,318 736,320 745,318 L756,268 L760,226Z" fill="#c7643a"/>
  <path d="M852,188 C866,194 874,212 879,236 L900,312 C892,318 880,320 871,318 L860,268 L856,226Z" fill="#b2502f"/>
  <path d="M760,226 C758,240 754,256 752,270" fill="none" stroke="#a7472c" stroke-width="1.6"/>
  <path d="M856,226 C858,240 862,256 864,270" fill="none" stroke="#933d27" stroke-width="1.6"/>
  <path d="M719,301 C728,306 738,308 748,306 L745,320 C734,322 724,320 715,316Z" fill="#b6552f"/>
  <path d="M897,301 C888,306 878,308 868,306 L871,320 C882,322 892,320 901,316Z" fill="#a24a2c"/>
  <path d="M719,306 C728,311 738,312 747,310 M718,314 C728,317 738,318 746,316" fill="none" stroke="#e8a66c" stroke-width="1.5" stroke-dasharray="2.5 1.8"/>
  <path d="M897,306 C888,311 878,312 869,310 M898,314 C888,317 878,318 870,316" fill="none" stroke="#e8a66c" stroke-width="1.5" stroke-dasharray="2.5 1.8"/>
  <!-- hands -->
  <path d="M718,318 C712,330 714,346 720,356 C724,362 731,361 732,355 L734,346 C737,346 739,340 738,334 C741,330 743,324 744,320Z" fill="#f09c6e"/>
  <path d="M724,340 L726,354 M730,336 L731,350" stroke="#d27c52" stroke-width="1.4" stroke-linecap="round"/>
  <path d="M898,318 C904,330 902,346 896,356 C892,362 885,361 884,355 L882,346 C879,346 877,340 878,334 C875,330 873,324 872,320Z" fill="#ec966a"/>
  <path d="M892,340 L890,354 M886,336 L885,350" stroke="#d27c52" stroke-width="1.4" stroke-linecap="round"/>
  <!-- neck & collar -->
  <path d="M791,156 L791,180 L825,180 L825,156Z" fill="#ec9a6c"/>
  <path d="M791,160 C800,170 816,170 825,160 L825,172 C816,178 800,178 791,172Z" fill="#d9825a" opacity=".7"/>
  <path d="M788,180 C790,190 796,196 802,198 L802,184 C796,182 792,180 788,178Z M828,180 C826,190 820,196 814,198 L814,184 C820,182 824,180 828,178Z" fill="#c96d42"/>
  <path d="M789,172 C796,177 802,179 806,180 L806,188 C800,187 794,184 787,180Z M827,172 C820,177 814,179 810,180 L810,188 C816,187 822,184 829,180Z" fill="#c4683f"/>
  <path d="M790,176 C796,180 801,182 805,183 M826,176 C820,180 815,182 811,183" fill="none" stroke="#eab07a" stroke-width="1.3" stroke-dasharray="2 1.5"/>
  <!-- placket -->
  <path d="M799,184 L817,184 L817,252 L808,262 L799,252Z" fill="#c96d42"/>
  <path d="M801,186 L801,251 L808,259 L815,251 L815,186" fill="none" stroke="#e8a66c" stroke-width="1.6" stroke-dasharray="2.5 1.8"/>
  <path d="M804,186 L812,186 L812,246 L808,251 L804,246Z" fill="#9c452c"/>
  <path d="M808,186 L808,250" stroke="#7c331f" stroke-width="1.2"/>
  <!-- head -->
  <g transform="translate(806,52) scale(0.85)">
    <use href="#head"/>
    <g transform="translate(-24,94)"><use href="#eyeOpen"/></g>
    <g transform="translate(23,92)"><use href="#eyeOpen"/></g>
    {brows("M-38,80 Q-28,72 -14,76", "M12,74 Q26,70 38,78")}
    <use href="#nose"/>
    <path d="M-19,115 Q0,128 19,115" {MOUTH} stroke-width="3"/>
  </g>
</g>"""

html = f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ورقة الشخصية</title>
<style>
  :root {{ --bg:#efe9df; --panel:#fff; --ink:#3a2a1e; --muted:#7a6a5c; --accent:#c4613a; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1d1a17; --panel:#2a2622; --ink:#f1e6d8; --muted:#b8a998; --accent:#e07a3e; }} }}
  :root[data-theme="dark"] {{ --bg:#1d1a17; --panel:#2a2622; --ink:#f1e6d8; --muted:#b8a998; --accent:#e07a3e; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Segoe UI",Tahoma,sans-serif; }}
  main {{ max-width:1100px; margin:0 auto; padding:24px 16px 40px; }}
  h1 {{ font-size:1.4rem; margin:0 0 6px; }}
  p {{ margin:0 0 16px; color:var(--muted); }}
  .sheet {{ background:#fff; border-radius:14px; box-shadow:0 6px 24px rgba(0,0,0,.12); overflow:hidden; }}
  .sheet svg {{ display:block; width:100%; height:auto; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:12px 24px; align-items:center; background:var(--panel); padding:12px 16px; border-radius:10px; margin-bottom:16px; }}
  label {{ display:flex; gap:8px; align-items:center; }}
  button {{ font:inherit; padding:8px 16px; border:0; border-radius:8px; background:var(--accent); color:#fff; cursor:pointer; }}
</style>
</head>
<body>
<main>
  <h1>ورقة تعابير الشخصية — فيكتور SVG</h1>
  <p>الرسم مرسوم فوق الصورة الأصلية بطريقة التراكب. استخدم الأدوات لمقارنة الرسم بالأصل.</p>
  <div class="controls">
    <label><input type="checkbox" id="showRef"> إظهار الصورة الأصلية تحت الرسم</label>
    <label>شفافية الرسم <input type="range" id="vecOp" min="0" max="100" value="100"></label>
    <button id="dl">تنزيل ملف SVG</button>
  </div>
  <div class="sheet">
<svg id="art" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 559" role="img" aria-label="ولد بطاقية زرقاء وثوب برتقالي بست تعابير ووقفة كاملة">
<defs>{defs}</defs>
<image id="ref" href="reference.jpg" width="1024" height="559" style="display:none"/>
<g id="vector">
<rect id="paper" width="1024" height="559" fill="#fdfcf9"/>
{body}
{think}
{full}
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
    svg.querySelector('#paper').style.display = '';
    svg.querySelector('#vector').style.opacity = '';
    var blob = new Blob([svg.outerHTML], {{ type: 'image/svg+xml' }});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'character-sheet.svg'; a.click();
    URL.revokeObjectURL(a.href);
  }});
</script>
</body>
</html>
"""
open(__file__.replace("tools/build_character.py", "character.html"), "w", encoding="utf-8").write(html)
print("written")
