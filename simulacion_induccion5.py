"""
=============================================================================
SIMULACIÓN INTERACTIVA: CALENTAMIENTO POR INDUCCIÓN ELECTROMAGNÉTICA
Proyecto Final — Física para Ciencias Exactas 2 — Ingeniería Bioquímica
v2.0 — Visualización dinámica con Canvas JS + parámetros realistas
=============================================================================
Cadena causal:
  Ampère-Maxwell → Faraday-Lenz → Ley de Ohm → Efecto Joule → Skin Depth
=============================================================================
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calentamiento por Inducción EM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CSS — contraste asegurado en todos los elementos
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title {
    font-size: 1.7rem; font-weight: 700;
    color: #1a3a5c; border-bottom: 3px solid #2196F3;
    padding-bottom: 0.4rem; margin-bottom: 0.8rem;
  }
  /* step-card: fondo claro, texto oscuro */
  .step-card {
    background: linear-gradient(135deg, #dceefb, #eaf4ff);
    border-left: 6px solid #2196F3;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
    color: #0d2137;
  }
  .step-card h4 { color: #0d47a1; margin: 0 0 6px 0; font-size: 1.05rem; }
  .step-card .eq {
    font-family: 'Courier New', monospace;
    background: #e8f0fe; border: 1px solid #90CAF9;
    border-radius: 4px; padding: 6px 10px;
    color: #0D47A1; font-size: 0.95rem; margin: 6px 0;
  }
  .step-card p, .step-card .explain { color: #1a2a3a; font-size: 0.9rem; }
  .law-badge {
    display:inline-block; background:#1565C0; color:#fff;
    border-radius:12px; padding:2px 10px; font-size:0.8rem;
    margin-bottom:6px;
  }
  /* result-metric: fondo naranja claro, texto oscuro */
  .result-metric {
    background: #fff3e0; border-radius: 8px;
    border-left: 5px solid #FF9800;
    padding: 10px 14px; margin: 6px 0;
  }
  .result-metric .label { font-size:0.8rem; color:#5d4037; font-weight:600; }
  .result-metric .value { font-size:1.3rem; font-weight:700; color:#bf360c; }
  /* bio-card */
  .bio-card {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    border-left: 6px solid #4CAF50;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
    color: #1b3a1e;
  }
  .bio-card h4 { color: #1B5E20; margin: 0 0 6px 0; }
  .bio-card p { color: #1b3a1e; }
  /* warning-box */
  .warning-box {
    background:#fff8e1; border-left:5px solid #FFC107;
    border-radius:6px; padding:10px 14px; margin:6px 0;
    font-size:0.88rem; color:#3e2723;
  }
  /* sustentacion nav */
  .sustentacion-nav {
    background: #1a3a5c; color:#e3f2fd;
    border-radius:10px; padding:10px 14px; margin:6px 0;
  }
  .sustentacion-nav h3 { color:#90CAF9; margin:0 0 4px 0; }
  /* preset buttons area */
  .preset-label { color: #1a3a5c; font-weight:700; font-size:0.9rem; }
  /* Ensure stMetric text is always visible */
  [data-testid="stMetricValue"] { color: #0d47a1 !important; }
  [data-testid="stMetricLabel"] { color: #37474f !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS
# ──────────────────────────────────────────────────────────────
MU0 = 4 * np.pi * 1e-7   # [T·m/A]

# ──────────────────────────────────────────────────────────────
# PRESETS DE MATERIALES
# ──────────────────────────────────────────────────────────────
MATERIALES = {
    "🥇 Cobre":          {"sigma": 58.0,  "mu_r": 1.0,    "rho": 8960, "cp": 385,  "label": "Cu"},
    "⚙️ Aluminio":       {"sigma": 37.7,  "mu_r": 1.0,    "rho": 2700, "cp": 897,  "label": "Al"},
    "🔩 Acero (carbono)":{"sigma": 7.0,   "mu_r": 100.0,  "rho": 7800, "cp": 490,  "label": "Fe"},
    "🧬 Tejido biológico":{"sigma": 0.001,"mu_r": 1.0,    "rho": 1050, "cp": 3600, "label": "Bio"},
}

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parámetros del Sistema")
    st.markdown("---")

    # ── Presets de material ───────────────────────────────────
    st.markdown('<p class="preset-label">⚡ Preset de material</p>', unsafe_allow_html=True)
    preset_sel = st.selectbox("Seleccionar material", list(MATERIALES.keys()), index=2)
    preset = MATERIALES[preset_sel]
    sigma_default = preset["sigma"]
    mu_r_default  = preset["mu_r"]

    st.markdown("---")
    st.markdown("### 🔌 Bobina Inductora")
    I0   = st.slider("Corriente pico I₀ (A)",        10.0,  150.0, 80.0,  5.0)
    N    = st.slider("Número de espiras N",           20,    200,   100,   5)
    L    = st.slider("Longitud de bobina L (m)",      0.05,  0.30,  0.12,  0.01, format="%.2f")
    f    = st.slider("Frecuencia f (Hz)",             100.0, 50000.0, 5000.0, 100.0, format="%.0f")

    st.markdown("### 🔵 Material Conductor")
    R_cyl = st.slider("Radio del cilindro R (m)",    0.005, 0.05,  0.015, 0.001, format="%.3f")
    sigma = st.slider("Conductividad σ (MS/m)",      0.0001, 60.0, sigma_default, 0.001, format="%.4f")
    mu_r  = st.slider("Permeabilidad relativa μᵣ",   1.0,   500.0, mu_r_default, 1.0)

    st.markdown("### 🎬 Visualización animada")
    pos_metal = st.slider("Posición del metal (0=fuera izq → 1=fuera der)", 0.0, 1.0, 0.5, 0.01)
    anim_speed = st.slider("Velocidad de animación", 0.3, 3.0, 1.0, 0.1)

    st.markdown("---")
    modo = st.radio("🎓 Modo de visualización",
                    ["📊 Análisis Completo", "🎤 Modo Sustentación"])
    if modo == "🎤 Modo Sustentación":
        paso_sust = st.select_slider("Paso:", options=[1,2,3,4,5,6,7,8], value=1)

# ──────────────────────────────────────────────────────────────
# CÁLCULOS FÍSICOS
# ──────────────────────────────────────────────────────────────
omega    = 2 * np.pi * f
sigma_SI = sigma * 1e6
mu       = mu_r * MU0
B0       = MU0 * (N / L) * I0

# Factor de posición: cuánto del cilindro está dentro de la bobina
# pos_metal: 0=completamente fuera izq, 0.5=centrado, 1=completamente fuera der
# overlap: 0→1→0  (máximo al centro)
overlap = 1.0 - 2.0 * abs(pos_metal - 0.5)   # 0..1
overlap = max(0.0, overlap)
B0_eff   = B0 * overlap       # campo efectivo según posición

# Serie temporal (3 períodos)
t_max = 3.0 / f if f > 0 else 0.06
t     = np.linspace(0, t_max, 2000)
I_t   = I0 * np.cos(omega * t)
B_t   = B0_eff * np.cos(omega * t)
Phi_t = B_t * np.pi * R_cyl**2
eps_t = B0_eff * np.pi * R_cyl**2 * omega * np.sin(omega * t)

# Distribución radial
r_arr = np.linspace(0, R_cyl, 300)
J_r   = sigma_SI * B0_eff * omega * r_arr / 2
p_r   = J_r**2 / (2 * sigma_SI) if sigma_SI > 0 else np.zeros_like(r_arr)

# Potencia total
P_total = np.pi * sigma_SI * B0_eff**2 * omega**2 * L * R_cyl**4 / 16

# Skin depth
delta_0 = np.sqrt(2 / (mu * sigma_SI * omega)) if (mu * sigma_SI * omega) > 0 else 1.0
f_arr   = np.logspace(np.log10(100), np.log10(5e4), 400)
delta_f = np.sqrt(2 / (mu * sigma_SI * 2 * np.pi * f_arr + 1e-30))

# Temperatura
rho_mat  = preset["rho"]
cp_mat   = preset["cp"]
vol_cyl  = np.pi * R_cyl**2 * L
mass_cyl = rho_mat * vol_cyl
delta_T  = P_total / (mass_cyl * cp_mat) if mass_cyl > 0 else 0.0

# Intensidad normalizada para visualización (0–1)
B_norm = min(B0_eff / (MU0 * 200/0.12 * 150), 1.0)  # relativo al máximo posible
P_norm = min(P_total / 1e6, 1.0)
penetration_pct = min(delta_0 / R_cyl, 1.0) if R_cyl > 0 else 1.0

# ──────────────────────────────────────────────────────────────
# ENCABEZADO Y MÉTRICAS
# ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⚡ Calentamiento por Inducción Electromagnética</p>',
            unsafe_allow_html=True)
st.markdown(f"*Simulación educativa — Física para Ciencias Exactas 2 — Material: **{preset_sel}***")

c1,c2,c3,c4,c5 = st.columns(5)
with c1:  st.metric("Campo B₀ efectivo", f"{B0_eff*1e3:.3f} mT")
with c2:  st.metric("FEM máx |ε|",       f"{float(np.max(np.abs(eps_t)))*1e3:.3f} mV")
with c3:  st.metric("J máx (sup.)",      f"{J_r[-1]:.3e} A/m²")
with c4:  st.metric("Potencia P",        f"{P_total:.4f} W")
with c5:  st.metric("Skin depth δ",      f"{delta_0*1e3:.3f} mm")

# Indicador de posición
pos_label = "⬅️ Entrando" if pos_metal < 0.3 else ("✅ Centrado" if pos_metal < 0.7 else "➡️ Saliendo")
st.info(f"📍 Posición del metal: **{pos_label}** — Overlap con bobina: **{overlap*100:.0f}%** — Calentamiento activo: **{overlap*100:.0f}%**")

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# VISUALIZACIÓN ANIMADA EN CANVAS JS  (pestaña principal)
# ════════════════════════════════════════════════════════════════
def build_canvas(B0_eff, omega, f, N, R_cyl, L, sigma_SI, mu,
                 delta_0, P_total, pos_metal, overlap,
                 J_max, p_max, anim_speed):
    """Genera el HTML+JS del Canvas animado."""

    # Pasar valores físicos a JS
    js_params = {
        "B0_eff":    float(B0_eff),
        "omega":     float(omega),
        "freq":      float(f),
        "N_coils":   int(N),
        "R_cyl_m":   float(R_cyl),
        "L_m":       float(L),
        "sigma":     float(sigma_SI),
        "delta_mm":  float(delta_0 * 1e3),
        "P_total":   float(P_total),
        "pos_metal": float(pos_metal),
        "overlap":   float(overlap),
        "J_max":     float(J_max),
        "p_max":     float(p_max),
        "speed":     float(anim_speed),
        "penetration": float(min(delta_0 / R_cyl, 1.0)) if R_cyl > 0 else 1.0,
    }
    params_json = json.dumps(js_params)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0; padding: 0;
    background: #060d1a;
    font-family: 'Courier New', monospace;
    overflow: hidden;
  }}
  canvas {{ display: block; }}
  #info {{
    position: absolute; top: 8px; left: 10px;
    color: #a0c4e8; font-size: 11px; line-height: 1.6;
    text-shadow: 0 0 6px #000;
    pointer-events: none;
  }}
  #info span {{ color: #64ffda; font-weight: bold; }}
  #legend {{
    position: absolute; bottom: 8px; left: 10px;
    color: #a0c4e8; font-size: 10px; line-height: 1.7;
    text-shadow: 0 0 4px #000;
    pointer-events: none;
  }}
  #step_label {{
    position: absolute; top: 8px; right: 10px;
    color: #ffd740; font-size: 10px; text-align: right;
    line-height: 1.5; text-shadow: 0 0 6px #000;
    pointer-events: none;
  }}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="info">
  I(t) = I₀·cos(ωt) → B(t) = <span id="bval">–</span> mT<br>
  ε(t) = <span id="eval">–</span> mV &nbsp;|&nbsp; J_max = <span id="jval">–</span> A/m²<br>
  δ = <span id="dval">–</span> mm &nbsp;|&nbsp; P = <span id="pval">–</span> W<br>
  Overlap bobina–metal: <span id="ovval">–</span>%
</div>
<div id="step_label">
  ① Ampère-Maxwell<br>② Faraday-Lenz<br>③ J = σE (Ohm)<br>④ Efecto Joule<br>⑤ Skin depth
</div>
<div id="legend">
  <span style="color:#ffd740">●</span> Espiras bobina &nbsp;
  <span style="color:#4fc3f7">→</span> Campo B(t) &nbsp;
  <span style="color:#ff7043">○</span> Corrientes Foucault &nbsp;
  <span style="color:#ef5350">■</span> Calentamiento (Joule)
</div>

<script>
const P = {params_json};

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');

// Fixed size — avoids tearing/flicker in iframe
canvas.width  = 900;
canvas.height = 460;

// Throttle to ~30fps to prevent tearing
let lastTime = 0;
const FRAME_MS = 1000 / 30;

// ─────────────── tiempo ──────────────────────────────────────
// t is in seconds. We advance by a small fraction of one period per frame.
// This keeps animation smooth regardless of frequency.
// dt_per_frame = (cycles_per_frame / frequency)
// At speed=1: ~0.5 cycles/frame → smooth; speed slider only changes visual rate
let t = 0;
const CYCLES_PER_FRAME = 0.025;  // ~0.75 visual cycles/sec at 30fps — smooth and visible

// ─────────────── partículas de corriente ─────────────────────
const N_PARTICLES = 120;
const particles = [];
function initParticles() {{
  particles.length = 0;
  for (let i = 0; i < N_PARTICLES; i++) {{
    const angle  = Math.random() * 2 * Math.PI;
    const r_frac = Math.sqrt(Math.random()); // uniforme en área
    particles.push({{
      angle:  angle,
      r_frac: r_frac,      // 0..1  (fracción del radio)
      phase:  Math.random() * 2 * Math.PI,
      speed:  0.8 + Math.random() * 1.2,
    }});
  }}
}}
initParticles();

// ─────────────── dibujo principal ────────────────────────────
function draw() {{
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Fondo degradado
  const bg = ctx.createRadialGradient(W/2, H/2, 10, W/2, H/2, W*0.7);
  bg.addColorStop(0, '#0d1b2a');
  bg.addColorStop(1, '#060d1a');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // Coordenadas de la bobina (longitud visual fija)
  const coilCx  = W * 0.5;
  const coilCy  = H * 0.5;
  const coilW   = W * 0.55;    // ancho total del solenoide
  const coilH   = H * 0.28;    // radio visual
  const coilX0  = coilCx - coilW / 2;
  const coilX1  = coilCx + coilW / 2;

  // Posición del metal: pos_metal 0→1, donde 0.5 = centrado
  // Desplazamos el cilindro a lo largo del eje del solenoide
  const metalOffset = (P.pos_metal - 0.5) * (coilW * 1.4);
  const metalCx     = coilCx + metalOffset;
  const metalW      = coilW * 0.72;  // longitud del cilindro
  const metalH      = coilH * 0.72;  // radio del cilindro
  const metalX0     = metalCx - metalW / 2;
  const metalX1     = metalCx + metalW / 2;

  // Valor instantáneo de B (oscila)
  const B_inst = P.B0_eff * Math.cos(P.omega * t);
  const B_norm_inst = P.B0_eff > 0 ? B_inst / P.B0_eff : 0;  // -1..1
  const B_abs  = Math.abs(B_norm_inst);

  // ── CILINDRO CONDUCTOR ─────────────────────────────────────
  // Color según calentamiento (azul frío → rojo caliente)
  const heat = P.overlap * B_abs * Math.min(P.P_total / (P.P_total + 0.001 + 1e-9), 1.0);
  // Escalar heat de forma más visible: usar log
  const heatViz = Math.min(Math.sqrt(P.overlap) * B_abs, 1.0);

  function heatColor(frac, alpha) {{
    // frío (azul) → medio (naranja) → caliente (rojo-blanco)
    const r = Math.round(frac < 0.5
      ? 20 + frac * 2 * 200
      : 220 + (frac - 0.5) * 2 * 35);
    const g = Math.round(frac < 0.5
      ? 60 + frac * 2 * 100
      : 160 - (frac - 0.5) * 2 * 110);
    const b = Math.round(frac < 0.5
      ? 180 - frac * 2 * 150
      : 30);
    return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
  }}

  // Gradiente radial en el cilindro (superficie más caliente)
  const cylGrad = ctx.createRadialGradient(metalCx, coilCy, 0,
                                            metalCx, coilCy, metalH * 1.1);
  const skinFrac = Math.min(P.penetration, 1.0);
  // Interior: frío (si skin depth pequeño)
  cylGrad.addColorStop(0,   heatColor(heatViz * (1 - skinFrac * 0.6) * 0.5, 0.92));
  cylGrad.addColorStop(0.6, heatColor(heatViz * 0.7, 0.95));
  cylGrad.addColorStop(1.0, heatColor(heatViz, 1.0));

  // Sombra de brillo
  ctx.shadowColor = heatColor(heatViz, 0.6);
  ctx.shadowBlur  = 20 + heatViz * 40;

  ctx.beginPath();
  ctx.ellipse(metalCx, coilCy, metalW/2, metalH, 0, 0, 2*Math.PI);
  ctx.fillStyle = cylGrad;
  ctx.fill();

  // Contorno del cilindro
  ctx.strokeStyle = `rgba(255,255,255,0.55)`;
  ctx.lineWidth   = 1.8;
  ctx.stroke();
  ctx.shadowBlur  = 0;

  // ── CAMPO MAGNÉTICO (flechas animadas) ────────────────────
  const nFieldLines = 9;
  const fieldAlpha  = 0.35 + B_abs * 0.55;
  const fieldWidth  = 1.0 + B_abs * 2.5;
  const arrowColor  = B_norm_inst >= 0 ? '#4fc3f7' : '#f48fb1';

  for (let i = 0; i < nFieldLines; i++) {{
    const yLine = coilCy + (i/(nFieldLines-1) - 0.5) * coilH * 1.6;
    const yDist  = Math.abs(yLine - coilCy) / (coilH * 0.8);
    const a      = fieldAlpha * Math.max(0.1, 1 - yDist * yDist);

    // Offset animado: las líneas de campo se mueven a lo largo del eje
    const flowOffset = ((t * P.omega / (2*Math.PI)) % 1.0) * coilW;
    const dir        = B_norm_inst >= 0 ? 1 : -1;

    ctx.strokeStyle = arrowColor;
    ctx.lineWidth   = fieldWidth;
    ctx.globalAlpha = a;
    ctx.beginPath();
    ctx.moveTo(coilX0, yLine);
    ctx.lineTo(coilX1, yLine);
    ctx.stroke();

    // Flecha en el punto animado
    const arrowX = coilX0 + (dir > 0
      ? flowOffset % coilW
      : coilW - (flowOffset % coilW));
    ctx.globalAlpha = a * 1.5;
    drawArrow(ctx, arrowX, yLine, dir * 16, 0, arrowColor, 8);
  }}
  ctx.globalAlpha = 1.0;

  // ── ESPIRAS DE LA BOBINA ──────────────────────────────────
  const nVisible = Math.min(P.N_coils, 22);
  const coilSpacing = coilW / (nVisible - 1);
  for (let i = 0; i < nVisible; i++) {{
    const cx_i = coilX0 + i * coilSpacing;
    const coilAlpha = 0.75 + B_abs * 0.25;
    const glowIntensity = 8 + B_abs * 22;

    ctx.shadowColor = '#ffd740';
    ctx.shadowBlur  = glowIntensity;
    ctx.strokeStyle = `rgba(255,215,0,${{coilAlpha}})`;
    ctx.lineWidth   = 2.2;

    // Elipse superior
    ctx.beginPath();
    ctx.ellipse(cx_i, coilCy - coilH, coilSpacing * 0.38, coilH * 0.22, 0, 0, 2*Math.PI);
    ctx.stroke();
    // Elipse inferior
    ctx.beginPath();
    ctx.ellipse(cx_i, coilCy + coilH, coilSpacing * 0.38, coilH * 0.22, 0, 0, 2*Math.PI);
    ctx.stroke();
  }}
  ctx.shadowBlur = 0;

  // Líneas laterales de la bobina
  ctx.strokeStyle = 'rgba(255,215,0,0.3)';
  ctx.lineWidth   = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(coilX0, coilCy - coilH); ctx.lineTo(coilX1, coilCy - coilH); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(coilX0, coilCy + coilH); ctx.lineTo(coilX1, coilCy + coilH); ctx.stroke();
  ctx.setLineDash([]);

  // ── CORRIENTES DE FOUCAULT (partículas orbitales) ─────────
  if (P.overlap > 0.05 && B_abs > 0.05) {{
    const J_intensity = P.overlap * B_abs;
    const particleAlpha = J_intensity;
    const particleR = 3.5 + J_intensity * 3.0;

    for (let p of particles) {{
      // radio proporcional a r_frac (J ∝ r)
      const r_px = p.r_frac * metalH * 0.88;
      // velocidad angular: más rápido donde B es más intenso
      // Particle speed in rad/frame: proportional to J intensity, independent of physics dt
      const angVel = p.speed * 2.0 * J_intensity;  // rad/frame, visual only
      p.angle += angVel * 0.04 * P.speed;

      const px = metalCx + r_px * Math.cos(p.angle);
      const py = coilCy  + r_px * Math.sin(p.angle) * 0.38; // perspectiva

      // solo dibujar si dentro del cilindro (elipse)
      const inside = ((px - metalCx)**2 / (metalW/2)**2 + (py - coilCy)**2 / (metalH)**2) < 0.88;
      if (!inside) continue;

      // Color: naranja (interior) → rojo-blanco (borde, mayor J)
      const rFrac = p.r_frac;
      const pr    = Math.round(200 + rFrac * 55);
      const pg    = Math.round(100 - rFrac * 70);
      const pb    = 0;

      ctx.shadowColor = `rgba(${{pr}},${{pg}},0,0.8)`;
      ctx.shadowBlur  = 6 + rFrac * 6;
      ctx.beginPath();
      ctx.arc(px, py, particleR * (0.4 + rFrac * 0.8), 0, 2*Math.PI);
      ctx.fillStyle = `rgba(${{pr}},${{pg}},${{pb}},${{particleAlpha * (0.5 + rFrac*0.5)}})`;
      ctx.fill();
    }}
    ctx.shadowBlur = 0;
  }}

  // ── SKIN DEPTH: anillo visual ────────────────────────────
  if (P.overlap > 0.05) {{
    const skinRatio = Math.min(P.penetration, 1.0);
    const skinR_px  = metalH * skinRatio;
    const skinAlpha = 0.55 + (1 - skinRatio) * 0.3;

    ctx.strokeStyle = `rgba(200,130,255,${{skinAlpha}})`;
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    ctx.ellipse(metalCx, coilCy, skinR_px * (metalW/metalH) * 0.88, skinR_px * 0.4, 0, 0, 2*Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);

    // Etiqueta δ
    ctx.fillStyle = 'rgba(200,130,255,0.85)';
    ctx.font = '11px Courier New';
    ctx.fillText('δ = ' + P.delta_mm.toFixed(2) + ' mm', metalCx + skinR_px * (metalW/metalH) * 0.88 + 4, coilCy);
  }}

  // ── OSCILÓGRAFO MINI (I y B) en esquina ──────────────────
  drawOscilloscope(ctx, 10, H - 90, 240, 80, t, P);

  // ── ETIQUETAS FÍSICAS ────────────────────────────────────
  // Bobina
  ctx.fillStyle = 'rgba(255,215,0,0.9)';
  ctx.font      = 'bold 12px Courier New';
  ctx.fillText('Bobina: N=' + P.N_coils + ', L=' + (P.L_m*100).toFixed(0) + ' cm', coilX0 - 10, coilCy - coilH - 22);

  // Campo B en tiempo real
  const Bmag = (P.B0_eff * Math.abs(Math.cos(P.omega * t)) * 1000).toFixed(2);
  ctx.fillStyle = B_norm_inst >= 0 ? '#4fc3f7' : '#f48fb1';
  ctx.font      = 'bold 11px Courier New';
  ctx.fillText('B(t) = ' + Bmag + ' mT', coilCx - 35, coilCy - coilH - 40);

  // Metal
  const metalLabel = P.pos_metal < 0.2 ? '← entrando' :
                     P.pos_metal > 0.8 ? 'saliendo →' : 'en el centro';
  ctx.fillStyle = 'rgba(255,160,60,0.9)';
  ctx.font      = '11px Courier New';
  ctx.fillText('Metal (' + metalLabel + ')', metalCx - 40, coilCy + metalH + 24);

  // Actualizar HUD
  document.getElementById('bval').textContent = (P.B0_eff * 1000 * Math.abs(Math.cos(P.omega * t))).toFixed(3);
  document.getElementById('eval').textContent = (P.B0_eff * Math.PI * P.R_cyl_m**2 * P.omega * Math.abs(Math.sin(P.omega * t)) * 1000).toFixed(3);
  document.getElementById('jval').textContent = P.J_max.toExponential(2);
  document.getElementById('dval').textContent = P.delta_mm.toFixed(2);
  document.getElementById('pval').textContent = P.P_total.toFixed(4);
  document.getElementById('ovval').textContent = (P.overlap * 100).toFixed(0);

  // dt = fraction of period, so animation is smooth at ANY frequency
  const dt = (CYCLES_PER_FRAME * P.speed) / P.freq;
  t += dt;
  setTimeout(function(){{ requestAnimationFrame(draw); }}, FRAME_MS);
}}

// ─── Osciloscopio mini ───────────────────────────────────────
function drawOscilloscope(ctx, x, y, w, h, t, P) {{
  ctx.fillStyle = 'rgba(0,10,20,0.75)';
  ctx.strokeStyle = 'rgba(100,180,255,0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = 'rgba(100,180,255,0.5)';
  ctx.font = '9px Courier New';
  ctx.fillText('I(t)  B(t)  ε(t)', x+5, y+12);

  const nPts = 200;
  const colors = ['#42a5f5', '#ef5350', '#ab47bc'];
  const signals = [
    (tt) => Math.cos(P.omega * tt),
    (tt) => Math.cos(P.omega * tt) * P.overlap,
    (tt) => Math.sin(P.omega * tt) * P.overlap,
  ];

  for (let s = 0; s < 3; s++) {{
    ctx.beginPath();
    ctx.strokeStyle = colors[s];
    ctx.lineWidth   = 1.4;
    for (let i = 0; i < nPts; i++) {{
      const tt   = t - (nPts - i) / nPts * (3.0 / P.freq);
      const val  = signals[s](tt);
      const px_i = x + 5 + (i / nPts) * (w - 10);
      const py_i = y + h/2 - val * (h/2 - 10);
      i === 0 ? ctx.moveTo(px_i, py_i) : ctx.lineTo(px_i, py_i);
    }}
    ctx.stroke();
  }}
}}

// ─── Flecha ──────────────────────────────────────────────────
function drawArrow(ctx, x, y, dx, dy, color, size) {{
  const angle = Math.atan2(dy, dx);
  ctx.fillStyle   = color;
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - size * Math.cos(angle - 0.4),
             y - size * Math.sin(angle - 0.4));
  ctx.lineTo(x - size * Math.cos(angle + 0.4),
             y - size * Math.sin(angle + 0.4));
  ctx.closePath();
  ctx.fill();
}}

draw();
</script>
</body>
</html>
"""
    return html


# ══════════════════════════════════════════════════════════════
# MODO ANÁLISIS COMPLETO
# ══════════════════════════════════════════════════════════════
if modo == "📊 Análisis Completo":

    # Cadena causal
    st.markdown("### 🔗 Cadena Causal del Proceso")
    flow_steps = [
        ("⚡ Corriente AC", "I(t)=I₀cos(ωt)", "#1565C0"),
        ("🧲 Campo B(t)",   "Ampère-Maxwell\nB=μ₀(N/L)I", "#1976D2"),
        ("🌊 Flujo Φ(t)",   "Φ=B·πr²",          "#1E88E5"),
        ("⚡ FEM ε(t)",     "Faraday-Lenz\nε=−dΦ/dt", "#2196F3"),
        ("🌀 Corrientes J(r)","Ohm: J=σE",      "#42A5F5"),
        ("🔥 Efecto Joule", "⟨p⟩=J²/2σ",        "#EF5350"),
        ("📏 Skin Depth δ", "δ=√(2/μσω)",       "#7B1FA2"),
        ("🧬 Hipertermia",  "NPs 41–46°C",      "#2E7D32"),
    ]
    cols_f = st.columns(8)
    for i, (lbl, eq, col) in enumerate(flow_steps):
        with cols_f[i]:
            st.markdown(
                f"<div style='background:{col};color:#fff;border-radius:8px;"
                f"padding:8px 4px;text-align:center;font-size:0.75rem;min-height:78px;'>"
                f"<b>{lbl}</b><br>"
                f"<span style='font-size:0.68rem;font-family:monospace;white-space:pre;'>{eq}</span>"
                f"</div>", unsafe_allow_html=True)

    st.markdown("---")

    tab_anim, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎬 Simulación Animada",
        "📡 I(t) y B(t)",
        "🌊 Flujo y FEM",
        "🌀 Corrientes J(r)",
        "🔥 Potencia p(r)",
        "📏 Skin Depth δ(f)",
    ])

    # ── TAB ANIMACIÓN ─────────────────────────────────────────
    with tab_anim:
        st.markdown("""
        <div class='step-card'>
          <span class='law-badge'>VISUALIZACIÓN DINÁMICA COMPLETA</span>
          <h4>Sistema completo animado — mueve el slider "Posición del metal" en la barra lateral</h4>
          <p>Observa cómo cambian las corrientes de Foucault, el calentamiento (Joule) y el campo B
          al desplazar el cilindro dentro/fuera de la bobina. Las partículas naranjas son las
          corrientes de Foucault. El anillo morado punteado es la profundidad de penetración δ.</p>
        </div>
        """, unsafe_allow_html=True)

        html_canvas = build_canvas(
            B0_eff=B0_eff, omega=omega, f=f, N=N, R_cyl=R_cyl, L=L,
            sigma_SI=sigma_SI, mu=mu, delta_0=delta_0,
            P_total=P_total, pos_metal=pos_metal, overlap=overlap,
            J_max=float(J_r[-1]), p_max=float(p_r[-1]) if p_r[-1]>0 else 1.0,
            anim_speed=anim_speed
        )
        components.html(html_canvas, height=500, scrolling=False)

        # Indicadores bajo el canvas
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown(f"""<div class='result-metric'>
              <div class='label'>Campo B₀ efectivo</div>
              <div class='value'>{B0_eff*1e3:.3f} mT</div></div>""",
              unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""<div class='result-metric'>
              <div class='label'>Overlap bobina–metal</div>
              <div class='value'>{overlap*100:.0f} %</div></div>""",
              unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""<div class='result-metric'>
              <div class='label'>Potencia disipada P</div>
              <div class='value'>{P_total:.5f} W</div></div>""",
              unsafe_allow_html=True)
        with col_d:
            pen_str = "✅ Total" if delta_0 >= R_cyl else f"⚠️ {delta_0/R_cyl*100:.0f}%"
            st.markdown(f"""<div class='result-metric'>
              <div class='label'>Penetración δ/R</div>
              <div class='value'>{pen_str}</div></div>""",
              unsafe_allow_html=True)

    # ── TAB 1: I(t) y B(t) ────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class='step-card'>
          <span class='law-badge'>LEY DE AMPÈRE-MAXWELL</span>
          <h4>Paso 1 → 2: Corriente alterna → Campo magnético</h4>
          <div class='eq'>B(t) = μ₀·(N/L)·I₀·cos(ωt) = B₀·cos(ωt)</div>
          <p>El campo generado es proporcional a N/L (densidad de espiras) y a I₀.
          Ambas señales están en fase. El campo efectivo se reduce cuando el metal
          está fuera del solenoide (overlap &lt; 100%).</p>
        </div>
        """, unsafe_allow_html=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, facecolor="#f0f4f8")
        fig.suptitle("Corriente alterna y campo magnético generado", fontsize=13,
                     fontweight="bold", color="#1a3a5c")
        t_ms = t * 1e3
        ax1.plot(t_ms, I_t, color="#1565C0", lw=2, label="I(t) = I0·cos(wt)")
        ax1.axhline(0, color='#888', lw=0.7, ls='--')
        ax1.fill_between(t_ms, I_t, alpha=0.18, color="#1565C0")
        ax1.set_ylabel("Corriente I (A)", fontsize=11, color="#1a2a3a")
        ax1.legend(fontsize=10, loc='upper right')
        ax1.set_facecolor("#ffffff")
        ax1.tick_params(colors='#333')
        ax1.grid(True, alpha=0.3)
        for sp in ax1.spines.values(): sp.set_color('#bbb')

        ax2.plot(t_ms, B_t * 1e3, color="#c62828", lw=2, label="B_ef(t) = B0·overlap·cos(wt)")
        ax2.axhline(0, color='#888', lw=0.7, ls='--')
        ax2.fill_between(t_ms, B_t * 1e3, alpha=0.18, color="#c62828")
        ax2.set_ylabel("Campo B (mT)", fontsize=11, color="#1a2a3a")
        ax2.set_xlabel("Tiempo (ms)", fontsize=11, color="#1a2a3a")
        ax2.legend(fontsize=10, loc='upper right')
        ax2.set_facecolor("#ffffff")
        ax2.tick_params(colors='#333')
        ax2.grid(True, alpha=0.3)
        for sp in ax2.spines.values(): sp.set_color('#bbb')
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""<div class='result-metric'>
              <div class='label'>Amplitud B₀ efectiva</div>
              <div class='value'>{B0_eff*1e3:.4f} mT</div></div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""<div class='result-metric'>
              <div class='label'>Frecuencia angular ω</div>
              <div class='value'>{omega:.1f} rad/s</div></div>""", unsafe_allow_html=True)

    # ── TAB 2: Flujo y FEM ────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class='step-card'>
          <span class='law-badge'>LEY DE FARADAY-LENZ</span>
          <h4>Paso 3 → 4: Del flujo a la FEM inducida</h4>
          <div class='eq'>ε(t) = −dΦ_B/dt = B₀·π·R²·ω·sin(ωt)</div>
          <p>El flujo magnético variable induce una FEM. Nótese el desfase de 90° entre Φ y ε.
          El signo negativo (Lenz) indica que la corriente inducida se opone al cambio del flujo.</p>
        </div>
        """, unsafe_allow_html=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, facecolor="#f0f4f8")
        fig.suptitle("Flujo magnético y FEM inducida", fontsize=13,
                     fontweight="bold", color="#1a3a5c")
        t_ms = t * 1e3
        ax1.plot(t_ms, Phi_t*1e6, color="#6a1b9a", lw=2, label="Phi_B(t) = B·pi·R²")
        ax1.axhline(0, color='#888', lw=0.7, ls='--')
        ax1.fill_between(t_ms, Phi_t*1e6, alpha=0.2, color="#6a1b9a")
        ax1.set_ylabel("Flujo Φ_B (μWb)", fontsize=11, color="#1a2a3a")
        ax1.legend(fontsize=10); ax1.set_facecolor("#ffffff")
        ax1.tick_params(colors='#333'); ax1.grid(True, alpha=0.3)
        for sp in ax1.spines.values(): sp.set_color('#bbb')

        ax2.plot(t_ms, eps_t*1e3, color="#bf360c", lw=2,
                 label="|eps(t)| = B0·pi·R²·w·sin(wt)")
        ax2.axhline(0, color='#888', lw=0.7, ls='--')
        ax2.fill_between(t_ms, eps_t*1e3, alpha=0.2, color="#bf360c")
        ax2.set_ylabel("|FEM ε| (mV)", fontsize=11, color="#1a2a3a")
        ax2.set_xlabel("Tiempo (ms)", fontsize=11, color="#1a2a3a")
        ax2.legend(fontsize=10); ax2.set_facecolor("#ffffff")
        ax2.tick_params(colors='#333'); ax2.grid(True, alpha=0.3)
        for sp in ax2.spines.values(): sp.set_color('#bbb')

        eps_max_val = float(np.max(np.abs(eps_t)))
        T_period = 1/f if f > 0 else 1
        t_peak = T_period/4 * 1e3
        ax2.annotate("Desfase 90° con Φ",
                     xy=(t_peak, eps_max_val*1e3*0.95),
                     xytext=(t_peak + T_period*1e3*0.12, eps_max_val*1e3*0.6),
                     fontsize=9, color="#b71c1c",
                     arrowprops=dict(arrowstyle='->', color='#b71c1c', lw=1.5))
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        st.markdown(f"""<div class='result-metric'>
          <div class='label'>FEM máxima |ε_max|</div>
          <div class='value'>{eps_max_val*1e3:.4f} mV</div></div>""", unsafe_allow_html=True)

    # ── TAB 3: Corrientes J(r) ────────────────────────────────
    with tab3:
        st.markdown("""
        <div class='step-card'>
          <span class='law-badge'>LEY DE OHM LOCAL  ·  J = σE</span>
          <h4>Paso 5: Distribución radial de corrientes de Foucault</h4>
          <div class='eq'>J(r) = (σ · B₀ · ω · r) / 2</div>
          <p>La densidad de corriente crece linealmente con r: las corrientes son más
          intensas en la periferia que en el centro. Este comportamiento es la causa
          del calentamiento diferencial radial.</p>
        </div>
        """, unsafe_allow_html=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#f0f4f8")
        fig.suptitle("Distribución radial de corrientes de Foucault J(r)",
                     fontsize=13, fontweight="bold", color="#1a3a5c")

        ax1.plot(r_arr*1e3, J_r, color="#c62828", lw=2.5)
        ax1.fill_between(r_arr*1e3, J_r, alpha=0.25, color="#c62828")
        ax1.set_xlabel("Radio r (mm)", fontsize=11, color="#1a2a3a")
        ax1.set_ylabel("J(r) (A/m²)", fontsize=11, color="#1a2a3a")
        ax1.set_title("Perfil radial de J", fontsize=11, color="#1a2a3a")
        ax1.set_facecolor("#ffffff"); ax1.tick_params(colors='#333')
        ax1.axvline(R_cyl*1e3, color='navy', ls='--', lw=1.5, label=f"R = {R_cyl*1e3:.1f} mm")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        for sp in ax1.spines.values(): sp.set_color('#bbb')

        N_grid = 180
        x_g = np.linspace(-R_cyl, R_cyl, N_grid)
        y_g = np.linspace(-R_cyl, R_cyl, N_grid)
        X, Y = np.meshgrid(x_g, y_g)
        R_grid = np.sqrt(X**2 + Y**2)
        J_map = np.where(R_grid <= R_cyl, sigma_SI * B0_eff * omega * R_grid / 2, np.nan)
        im = ax2.imshow(J_map, extent=[-R_cyl*1e3, R_cyl*1e3, -R_cyl*1e3, R_cyl*1e3],
                        origin='lower', cmap='hot', aspect='equal')
        circle = plt.Circle((0,0), R_cyl*1e3, fill=False, edgecolor='white', lw=2, ls='--')
        ax2.add_patch(circle)
        plt.colorbar(im, ax=ax2, label="J (A/m²)")
        ax2.set_xlabel("x (mm)", fontsize=11); ax2.set_ylabel("y (mm)", fontsize=11)
        ax2.set_title("Corte transversal — J (vista superior)", fontsize=11, color="#1a2a3a")
        ax2.set_facecolor("#111")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""<div class='result-metric'>
              <div class='label'>J en superficie (r=R)</div>
              <div class='value'>{J_r[-1]:.3e} A/m²</div></div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""<div class='result-metric'>
              <div class='label'>J en centro (r=0)</div>
              <div class='value'>0.000 A/m²</div></div>""", unsafe_allow_html=True)

    # ── TAB 4: Potencia ───────────────────────────────────────
    with tab4:
        st.markdown("""
        <div class='step-card'>
          <span class='law-badge'>EFECTO JOULE</span>
          <h4>Paso 6: Densidad de potencia media disipada ⟨p(r)⟩</h4>
          <div class='eq'>⟨p(r)⟩ = J²(r) / (2σ) = (σ/2)·(B₀·ω·r/2)²</div>
          <div class='eq'>P_total = π·σ·B₀²·ω²·L·R⁴ / 16</div>
          <p>La potencia crece con r²: calentamiento máximo en superficie, nulo en el eje.
          Duplicar f → ×4 en P. Duplicar R → ×16 en P.</p>
        </div>
        """, unsafe_allow_html=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#f0f4f8")
        fig.suptitle("Densidad de potencia disipada ⟨p(r)⟩ — Efecto Joule",
                     fontsize=13, fontweight="bold", color="#1a3a5c")

        ax1.plot(r_arr*1e3, p_r, color="#e65100", lw=2.5)
        ax1.fill_between(r_arr*1e3, p_r, alpha=0.3, color="#e65100")
        ax1.set_xlabel("Radio r (mm)", fontsize=11, color="#1a2a3a")
        ax1.set_ylabel("⟨p(r)⟩ (W/m³)", fontsize=11, color="#1a2a3a")
        ax1.set_title("Perfil radial de potencia", fontsize=11, color="#1a2a3a")
        ax1.set_facecolor("#ffffff"); ax1.tick_params(colors='#333')
        ax1.axvline(R_cyl*1e3, color='navy', ls='--', lw=1.5, label=f"R={R_cyl*1e3:.1f} mm")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        for sp in ax1.spines.values(): sp.set_color('#bbb')

        P_map = np.where(R_grid <= R_cyl,
                         (sigma_SI/2) * (B0_eff * omega * R_grid / 2)**2, np.nan)
        im2 = ax2.imshow(P_map, extent=[-R_cyl*1e3, R_cyl*1e3, -R_cyl*1e3, R_cyl*1e3],
                         origin='lower', cmap='inferno', aspect='equal')
        circle2 = plt.Circle((0,0), R_cyl*1e3, fill=False, edgecolor='white', lw=2, ls='--')
        ax2.add_patch(circle2)
        plt.colorbar(im2, ax=ax2, label="⟨p⟩ (W/m³)")
        ax2.set_xlabel("x (mm)", fontsize=11); ax2.set_ylabel("y (mm)", fontsize=11)
        ax2.set_title("Mapa térmico — gradiente radial", fontsize=11, color="#1a2a3a")
        ax2.set_facecolor("#111")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""<div class='result-metric'>
              <div class='label'>Potencia total P</div>
              <div class='value'>{P_total:.5f} W</div></div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""<div class='result-metric'>
              <div class='label'>Tasa calentamiento ΔT/s</div>
              <div class='value'>{delta_T:.4f} K/s</div></div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""<div class='result-metric'>
              <div class='label'>Potencia en superficie</div>
              <div class='value'>{float(p_r[-1]):.4e} W/m³</div></div>""", unsafe_allow_html=True)

        st.markdown("""<div class='warning-box'>
        📌 <b>Relaciones clave:</b> P ∝ σ · B₀² · ω² · R⁴  —
        Duplicar f ⟹ ×4P  |  Duplicar R ⟹ ×16P  |  Duplicar σ ⟹ ×2P
        </div>""", unsafe_allow_html=True)

    # ── TAB 5: Skin Depth ─────────────────────────────────────
    with tab5:
        st.markdown("""
        <div class='step-card'>
          <span class='law-badge'>SKIN DEPTH — PROFUNDIDAD DE PENETRACIÓN</span>
          <h4>Paso 7: Concentración superficial de corrientes a alta frecuencia</h4>
          <div class='eq'>δ = √(2 / (μ · σ · ω)) = √(2ρ / (μ · ω))</div>
          <div class='eq'>J(z) = J₀ · e^(−z/δ)</div>
          <p>A mayor frecuencia, las corrientes se confinan en una capa más delgada.
          Cuando δ ≪ R el calentamiento es superficial; cuando δ ≥ R es volumétrico.
          Crítico para el diseño de sistemas de hipertermia oncológica.</p>
        </div>
        """, unsafe_allow_html=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#f0f4f8")
        fig.suptitle("Profundidad de penetración (Skin Depth) δ(f)",
                     fontsize=13, fontweight="bold", color="#1a3a5c")

        ax1.loglog(f_arr, delta_f*1e3, color="#6a1b9a", lw=2.5)
        ax1.axvline(f, color="#e53935", ls='--', lw=2, label=f"f = {f:.0f} Hz → δ = {delta_0*1e3:.2f} mm")
        ax1.axhline(delta_0*1e3, color="#e53935", ls=':', lw=1.5)
        ax1.scatter([f], [delta_0*1e3], color="#e53935", s=90, zorder=5)
        ax1.axhline(R_cyl*1e3, color='#2e7d32', ls='--', lw=1.5, label=f"R = {R_cyl*1e3:.1f} mm")
        ax1.set_xlabel("Frecuencia f (Hz)", fontsize=11, color="#1a2a3a")
        ax1.set_ylabel("δ (mm)", fontsize=11, color="#1a2a3a")
        ax1.set_title(f"δ({f:.0f} Hz) = {delta_0*1e3:.3f} mm", fontsize=11, color="#1a2a3a")
        ax1.set_facecolor("#ffffff"); ax1.tick_params(colors='#333')
        ax1.grid(True, alpha=0.3, which='both'); ax1.legend()
        for sp in ax1.spines.values(): sp.set_color('#bbb')

        z_max = max(5*delta_0, R_cyl)
        z_arr = np.linspace(0, z_max, 300)
        J_z   = np.exp(-z_arr/delta_0) if delta_0 > 0 else np.zeros_like(z_arr)
        ax2.plot(J_z, z_arr*1e3, color="#6a1b9a", lw=2.5, label="J(z) = J0·exp(-z/delta)")
        ax2.axhline(delta_0*1e3, color="#e53935", ls='--', lw=1.5, label=f"z=δ={delta_0*1e3:.2f} mm")
        ax2.axvline(1/np.e, color='#555', ls=':', lw=1.2, label="J/J₀ = 1/e ≈ 0.368")
        ax2.fill_betweenx(z_arr*1e3, J_z, 0, where=(z_arr<=delta_0),
                          alpha=0.2, color="#6a1b9a", label="Capa activa δ")
        ax2.set_xlabel("J(z)/J₀", fontsize=11, color="#1a2a3a")
        ax2.set_ylabel("Profundidad z (mm)", fontsize=11, color="#1a2a3a")
        ax2.set_title("Decaimiento exponencial con profundidad", fontsize=11, color="#1a2a3a")
        ax2.invert_yaxis()
        ax2.set_facecolor("#ffffff"); ax2.tick_params(colors='#333')
        ax2.grid(True, alpha=0.3); ax2.legend(fontsize=9)
        for sp in ax2.spines.values(): sp.set_color('#bbb')
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        col_a, col_b = st.columns(2)
        cmp = "✅ Penetración total (volumétrico)" if delta_0 >= R_cyl else f"⚠️ Solo superficial ({delta_0/R_cyl*100:.0f}% del radio)"
        with col_a:
            st.markdown(f"""<div class='result-metric'>
              <div class='label'>Tipo de calentamiento</div>
              <div class='value' style='font-size:1rem;'>{cmp}</div></div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""<div class='result-metric'>
              <div class='label'>δ/R (penetración relativa)</div>
              <div class='value'>{min(delta_0/R_cyl,1.0)*100:.1f} %</div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MODO SUSTENTACIÓN
# ══════════════════════════════════════════════════════════════
else:
    pasos_info = {
        1: {
            "titulo": "Corriente Alterna en la Bobina",
            "ley": "Principio de partida",
            "eq": "I(t) = I₀ · cos(ωt)",
            "que_ocurre": "Una corriente alterna de frecuencia f circula por la bobina inductora de N vueltas. Esta corriente oscila sinusoidalmente en el tiempo.",
            "que_cambia": f"I₀ = {I0:.0f} A  |  f = {f:.0f} Hz  |  ω = {omega:.1f} rad/s",
            "efecto": "Genera un campo magnético variable en el tiempo dentro de la bobina.",
            "color": "#1565C0", "emoji": "⚡",
        },
        2: {
            "titulo": "Campo Magnético — Ley de Ampère-Maxwell",
            "ley": "Ley de Ampère-Maxwell",
            "eq": f"B(t) = μ₀·(N/L)·I(t) = B₀·cos(ωt)  →  B₀ = {B0*1e3:.3f} mT",
            "que_ocurre": "La corriente alterna genera un campo magnético B(t) variable. Dentro del solenoide largo el campo es uniforme. El campo efectivo depende de qué fracción del metal está dentro.",
            "que_cambia": f"B₀ = {B0*1e3:.3f} mT  |  B₀ efectivo (overlap {overlap*100:.0f}%) = {B0_eff*1e3:.3f} mT",
            "efecto": "El campo magnético variable penetra el material conductor e inicia la inducción.",
            "color": "#1976D2", "emoji": "🧲",
        },
        3: {
            "titulo": "Flujo Magnético Variable",
            "ley": "Definición de flujo magnético",
            "eq": f"Φ_B(t) = B(t)·π·R²  →  Φ_max = {B0_eff*np.pi*R_cyl**2*1e9:.3f} nWb",
            "que_ocurre": "El campo B(t) atraviesa el área circular del conductor. Al variar B, el flujo también varía sinusoidalmente.",
            "que_cambia": f"πR² = {np.pi*R_cyl**2*1e6:.3f} mm²  |  Φ ∝ B₀_eff · R²",
            "efecto": "La variación del flujo es la causa directa de la FEM inducida (Faraday).",
            "color": "#9C27B0", "emoji": "🌊",
        },
        4: {
            "titulo": "FEM Inducida — Ley de Faraday-Lenz",
            "ley": "Ley de Faraday-Lenz",
            "eq": "ε = −dΦ_B/dt = B₀·π·R²·ω·sin(ωt)",
            "que_ocurre": "La variación temporal del flujo induce una FEM en toda trayectoria cerrada. El signo negativo (Lenz) indica oposición al cambio. Hay un desfase de 90° con respecto a Φ.",
            "que_cambia": f"ε_max = {B0_eff*np.pi*R_cyl**2*omega*1e3:.4f} mV  |  Desfase 90°",
            "efecto": "La FEM impulsa cargas eléctricas dentro del material conductor.",
            "color": "#FF5722", "emoji": "⚡",
        },
        5: {
            "titulo": "Corrientes de Foucault — Ley de Ohm Local",
            "ley": "J = σE (Ley de Ohm diferencial)",
            "eq": "J(r,t) = (σ · B₀ · ω · r / 2) · sin(ωt)",
            "que_ocurre": "La FEM impulsa corrientes circulares perpendiculares al campo B: las corrientes de Foucault. Su densidad crece linealmente con r.",
            "que_cambia": f"J_max(R) = {J_r[-1]:.3e} A/m²  |  J(0) = 0 A/m²",
            "efecto": "Las corrientes circulan por el material resistivo y disipan energía como calor.",
            "color": "#F44336", "emoji": "🌀",
        },
        6: {
            "titulo": "Calentamiento — Efecto Joule",
            "ley": "Efecto Joule: ⟨p⟩ = J²/σ",
            "eq": "⟨p(r)⟩ = J²(r)/(2σ)  |  P = π·σ·B₀²·ω²·L·R⁴/16",
            "que_ocurre": "Las corrientes de Foucault disipan energía al circular por el material. La potencia crece con r²: calentamiento máximo en superficie, nulo en el eje.",
            "que_cambia": f"P_total = {P_total:.5f} W  |  ΔT/s ≈ {delta_T:.4f} K/s",
            "efecto": "El conductor se calienta. P ∝ ω²·R⁴: duplicar f → ×4P; duplicar R → ×16P.",
            "color": "#FF6F00", "emoji": "🔥",
        },
        7: {
            "titulo": "Skin Depth — Profundidad de Penetración",
            "ley": "Ecuación de difusión EM",
            "eq": f"δ = √(2/(μ·σ·ω)) = {delta_0*1e3:.3f} mm  →  J(z) = J₀·e^(−z/δ)",
            "que_ocurre": "A alta frecuencia, las corrientes se confinan en una capa delgada δ. Debajo de δ, J cae a 1/e de su valor superficial.",
            "que_cambia": f"δ = {delta_0*1e3:.3f} mm  |  R/δ = {R_cyl/delta_0:.2f}  →  {'Volumétrico' if delta_0 >= R_cyl else 'Superficial'}",
            "efecto": "Limita la profundidad de calentamiento. Crítico en hipertermia para tumores profundos.",
            "color": "#7B1FA2", "emoji": "📏",
        },
        8: {
            "titulo": "Aplicación — Hipertermia Oncológica por Nanopartículas",
            "ley": "Aplicación biomédica integrada",
            "eq": "T_objetivo ∈ [41, 46] °C  |  f ∈ [50 kHz, 1 MHz]  |  d_NP ∈ [1–100 nm]",
            "que_ocurre": "Nanopartículas magnéticas inyectadas en el tumor se excitan con campo magnético alterno. Disipan calor por corrientes de Foucault y pérdidas por histéresis.",
            "que_cambia": "Tamaño NP → mecanismo (Néel vs Browniano). Frecuencia → profundidad de penetración (ec. 7).",
            "efecto": "El tumor alcanza 41–46 °C → apoptosis. Tejido sano apenas se calienta (sin NPs).",
            "color": "#2E7D32", "emoji": "🧬",
        },
    }

    info = pasos_info[paso_sust]
    st.markdown(f"""
    <div class='sustentacion-nav'>
      <h3>{info['emoji']} Paso {paso_sust}/8 — {info['titulo']}</h3>
      <span style='color:#90CAF9;font-size:0.85rem;'>Material activo: {preset_sel}</span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(paso_sust / 8)

    col_info, col_graf = st.columns([1, 1.4])

    with col_info:
        st.markdown(f"""
        <div class='step-card' style='border-left-color:{info["color"]}'>
          <span class='law-badge' style='background:{info["color"]};'>{info['ley']}</span>
          <h4>Ecuación central:</h4>
          <div class='eq'>{info['eq']}</div>
          <hr style='border-color:#c5d8f0; margin:10px 0;'>
          <p><b>🔍 ¿Qué ocurre?</b><br>{info['que_ocurre']}</p>
          <p><b>🔢 Parámetro clave:</b><br><code style='background:#e8f0fe;color:#1a237e;padding:2px 5px;border-radius:3px;'>{info['que_cambia']}</code></p>
          <p><b>➡️ Consecuencia:</b><br>{info['efecto']}</p>
        </div>
        """, unsafe_allow_html=True)
        if paso_sust > 1:
            st.caption(f"◀ Anterior: Paso {paso_sust-1} — {pasos_info[paso_sust-1]['titulo']}")
        if paso_sust < 8:
            st.caption(f"▶ Siguiente: Paso {paso_sust+1} — {pasos_info[paso_sust+1]['titulo']}")

    with col_graf:
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        t_ms = t * 1e3

        def style_ax_dark(ax):
            ax.tick_params(colors='#aaa')
            for sp in ax.spines.values(): sp.set_color('#444')
            ax.grid(True, alpha=0.2, color='#444')
            ax.title.set_color('white')
            for lb in ax.get_xticklabels()+ax.get_yticklabels(): lb.set_color('#ccc')

        if paso_sust == 1:
            ax.plot(t_ms, I_t, color="#64B5F6", lw=2.5, label="I(t) = I0·cos(wt)")
            ax.fill_between(t_ms, I_t, alpha=0.2, color="#64B5F6")
            ax.set_ylabel("I (A)", color="#ccc"); ax.set_xlabel("t (ms)", color="#ccc")
            ax.set_title("Corriente alterna en la bobina", color="white")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
        elif paso_sust == 2:
            ax.plot(t_ms, I_t/I0, color="#64B5F6", lw=1.5, ls='--', label="I(t)/I₀", alpha=0.7)
            ax.plot(t_ms, B_t/(B0 if B0>0 else 1), color="#F06292", lw=2.5, label="B_eff(t)/B₀")
            ax.set_ylabel("Valor normalizado", color="#ccc"); ax.set_xlabel("t (ms)", color="#ccc")
            ax.set_title("B_ef(t) = B₀·overlap·cos(ωt)", color="white")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
            ax.text(0.5, 0.05, f"overlap = {overlap*100:.0f}%", transform=ax.transAxes,
                    ha='center', color='#FFD740', fontsize=11, fontweight='bold')
        elif paso_sust == 3:
            ax.plot(t_ms, Phi_t*1e9, color="#CE93D8", lw=2.5, label="Phi_B(t) (nWb)")
            ax.fill_between(t_ms, Phi_t*1e9, alpha=0.2, color="#CE93D8")
            ax.set_ylabel("Φ_B (nWb)", color="#ccc"); ax.set_xlabel("t (ms)", color="#ccc")
            ax.set_title("Flujo magnético variable", color="white")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
        elif paso_sust == 4:
            ax.plot(t_ms, Phi_t/(np.max(np.abs(Phi_t))+1e-30), color="#CE93D8",
                    lw=1.5, ls='--', label="Φ_B (norm)", alpha=0.7)
            ax.plot(t_ms, eps_t/(np.max(np.abs(eps_t))+1e-30), color="#FF7043",
                    lw=2.5, label="ε (norm)")
            ax.set_ylabel("Normalizado", color="#ccc"); ax.set_xlabel("t (ms)", color="#ccc")
            ax.set_title("Faraday-Lenz: Φ y ε desfasados 90°", color="white")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
        elif paso_sust == 5:
            ax.plot(r_arr*1e3, J_r, color="#EF5350", lw=2.5, label="J(r) = sigma·B0·w·r/2")
            ax.fill_between(r_arr*1e3, J_r, alpha=0.25, color="#EF5350")
            ax.set_xlabel("r (mm)", color="#ccc"); ax.set_ylabel("J(r) [A/m²]", color="#ccc")
            ax.set_title("Corrientes de Foucault — J(r)", color="white")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
        elif paso_sust == 6:
            ax.plot(r_arr*1e3, p_r, color="#FFA726", lw=2.5, label="<p> = J²/(2·sigma)")
            ax.fill_between(r_arr*1e3, p_r, alpha=0.3, color="#FFA726")
            ax.set_xlabel("r (mm)", color="#ccc"); ax.set_ylabel("⟨p⟩ [W/m³]", color="#ccc")
            ax.set_title("Efecto Joule — potencia disipada", color="white")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
        elif paso_sust == 7:
            ax.semilogx(f_arr, delta_f*1e3, color="#CE93D8", lw=2.5)
            ax.axvline(f, color="#FF7043", ls='--', lw=2, label=f"f={f:.0f}Hz → δ={delta_0*1e3:.2f}mm")
            ax.axhline(delta_0*1e3, color="#FF7043", ls=':', lw=1.5)
            ax.axhline(R_cyl*1e3, color="#66BB6A", ls='--', lw=1.5, label=f"R={R_cyl*1e3:.1f}mm")
            ax.set_xlabel("f (Hz)", color="#ccc"); ax.set_ylabel("δ (mm)", color="#ccc")
            ax.set_title("Skin Depth δ(f)", color="white")
            ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        elif paso_sust == 8:
            theta = np.linspace(0, 2*np.pi, 100)
            ax.fill(0.3*np.cos(theta), 0.3*np.sin(theta), color="#EF5350", alpha=0.55, label="Tumor")
            ax.plot(0.3*np.cos(theta), 0.3*np.sin(theta), color="#EF5350", lw=2)
            rng = np.random.default_rng(42)
            rp = rng.uniform(0, 0.25, 30); tp = rng.uniform(0, 2*np.pi, 30)
            ax.scatter(rp*np.cos(tp), rp*np.sin(tp), c='orange', s=18, zorder=5, label="NPs magnéticas")
            ax.add_patch(plt.Circle((0,0), 0.6, fill=False, edgecolor="#66BB6A", lw=2.5, ls='--'))
            ax.text(0, 0.67, "Tejido sano", ha='center', color="#66BB6A", fontsize=9)
            for xl in np.linspace(-0.65, 0.65, 5):
                ax.annotate("", xy=(xl,0.65), xytext=(xl,-0.65),
                            arrowprops=dict(arrowstyle="-|>", color="#64B5F6", lw=1.2, mutation_scale=7), alpha=0.5)
            ax.text(0, 0, "41–46 °C\nApoptosis", ha='center', va='center',
                    color="white", fontsize=10, fontweight='bold', zorder=6)
            ax.set_xlim(-0.85,0.85); ax.set_ylim(-0.85,0.85); ax.set_aspect('equal')
            ax.set_title("Hipertermia magnética — esquema", color="white")
            ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9, loc='lower right')

        style_ax_dark(ax)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    st.markdown("---")
    st.markdown("### 🔗 Posición en la cadena causal")
    cols_chain = st.columns(8)
    emojis_c = ["⚡","🧲","🌊","⚡","🌀","🔥","📏","🧬"]
    labels_c = ["I(t)","B(t)","Φ(t)","ε(t)","J(r)","p(r)","δ(f)","Bio"]
    for i,(em,lb) in enumerate(zip(emojis_c, labels_c)):
        with cols_chain[i]:
            active = (i+1 == paso_sust)
            bg = "#1565C0" if active else "#1e2a3a"
            border = "3px solid #64B5F6" if active else "1px solid #2a3a4a"
            st.markdown(
                f"<div style='background:{bg};border:{border};border-radius:8px;"
                f"padding:8px 4px;text-align:center;color:white;font-size:0.83rem;'>"
                f"{em}<br><b>{lb}</b></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='background:#1a3a5c;border-radius:8px;padding:12px 16px;color:#cfd8e3;font-size:0.82rem;'>
<b style='color:#90CAF9;'>📖 Formalismo:</b>
B = μ₀(N/L)I₀cos(ωt) → ε = B₀πR²ω·sin(ωt) → J(r) = σB₀ωr/2 →
⟨p(r)⟩ = σ(B₀ωr)²/8 → P = πσB₀²ω²LR⁴/16 → δ = √(2/μσω)<br>
<b style='color:#90CAF9;'>📚 Referencias:</b>
Rudnev et al. (2017) | Giuliani (2008) EPL 81(6) | Fernández Villarroel et al. (2025) Appl. Sci. 15(4) | Bermúdez et al. (2009)<br>
<b style='color:#90CAF9;'>🤖 IAG (Nivel 4):</b> Desarrollado con Claude Sonnet 4.6 (Anthropic, 2026). Material activo: {preset_sel}
</div>
""", unsafe_allow_html=True)
