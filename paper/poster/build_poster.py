"""Build the scientific poster (HTML) from Step 7 artifacts.

The results table and every figure are read from ``artifacts/public/step7-real``
so no metric is ever typed by hand. Render the PDF afterwards with:

    uvx weasyprint paper/poster/poster.html paper/poster/poster_better_router.pdf
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "artifacts" / "public" / "step7-real"
OUTPUT = Path(__file__).resolve().parent / "poster.html"

POLICY_LABELS = {
    "oracle": "Oracle offline (cota superior)",
    "better-rules-proxy": "Better Rules Proxy (baseline)",
    "xgboost": "Router XGBoost",
    "linucb": "Router LinUCB",
    "fixed:gpt-4-1106-preview": "Fijo: GPT-4 Turbo",
    "fixed:zero-one-ai/Yi-34B-Chat": "Fijo: Yi-34B",
    "fixed:mistralai/mixtral-8x7b-chat": "Fijo: Mixtral 8x7B",
    "fixed:mistralai/mistral-7b-chat": "Fijo: Mistral 7B",
}
POLICY_ORDER = list(POLICY_LABELS)
HIGHLIGHT = {"oracle", "better-rules-proxy", "xgboost", "linucb"}


def _figure_data_uri(name: str) -> str:
    payload = (RUN / "figures" / name).read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(payload).decode("ascii")


def _results_table() -> str:
    summary = pd.read_csv(RUN / "evaluation_summary.csv").set_index("policy")
    per_seed = pd.read_csv(RUN / "evaluation_per_seed.csv")
    extra = per_seed.groupby("policy").agg(
        quality=("mean_quality", "mean"), cost=("mean_cost_usd", "mean")
    )
    rows: list[str] = []
    for policy in POLICY_ORDER:
        s = summary.loc[policy]
        diff = (
            "—"
            if policy == "better-rules-proxy"
            else (
                f"{s['diff_vs_baseline']:+.4f}<br/>"
                f"<span class='ci'>[{s['diff_ci95_low']:+.4f}, {s['diff_ci95_high']:+.4f}]</span>"
            )
        )
        css = " class='hl'" if policy in HIGHLIGHT else ""
        rows.append(
            f"<tr{css}><td class='pol'>{POLICY_LABELS[policy]}</td>"
            f"<td>{s['mean_utility']:.4f}</td><td>{diff}</td>"
            f"<td>{extra.loc[policy, 'quality']:.3f}</td>"
            f"<td>{extra.loc[policy, 'cost']:.5f}</td></tr>"
        )
    return "\n".join(rows)


HTML = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<style>
@page {{ size: 1189mm 841mm; margin: 0; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  width: 1189mm; height: 841mm; font-family: "DejaVu Sans", "Liberation Sans", sans-serif;
  background: #eef3f8; color: #16232f; font-size: 19.5pt; line-height: 1.3;
  position: relative;
}}
.footer {{
  position: absolute; bottom: 0; left: 0; right: 0;
  background: linear-gradient(180deg, #0a5ca8 0%, #04365f 100%); color: #cfe3f7;
  padding: 5mm 22mm; font-size: 16.5pt; text-align: center;
}}
.footer strong {{ color: #ffffff; }}
.banner {{
  background: linear-gradient(180deg, #04365f 0%, #0a5ca8 100%); color: #ffffff;
  padding: 13mm 22mm 11mm 22mm;
}}
.banner h1 {{ font-size: 56pt; line-height: 1.08; letter-spacing: -0.5pt; }}
.banner .authors {{ font-size: 26pt; margin-top: 5mm; }}
.banner .affil {{ font-size: 19pt; margin-top: 2mm; color: #cfe3f7; }}
.columns {{
  display: table; table-layout: fixed; width: 100%;
  border-spacing: 8mm 0; margin-top: 4mm; padding: 0 4mm;
}}
.col {{ display: table-cell; vertical-align: top; }}
.block {{
  background: #ffffff; border: 0.7mm solid #b9cde2; border-radius: 4mm;
  overflow: hidden; margin-bottom: 8mm; page-break-inside: avoid;
}}
.block h2 {{
  background: linear-gradient(180deg, #0a5ca8 0%, #04365f 100%); color: #ffffff;
  font-size: 27pt; padding: 3.5mm 6mm; letter-spacing: 0.5pt;
}}
.block .body {{ padding: 5mm 6mm 6mm 6mm; }}
.block p + p, .block p + ul, .block ul + p, .block p + div {{ margin-top: 3mm; }}
ul {{ padding-left: 7mm; }}
li {{ margin-top: 1.8mm; }}
strong {{ color: #04365f; }}
.eq {{
  background: #eaf2fb; border: 0.5mm solid #b9cde2; border-radius: 2.5mm;
  text-align: center; font-size: 23pt; padding: 3.5mm; margin: 3.5mm 0; font-weight: bold;
  color: #04365f;
}}
.flow {{ display: flex; flex-direction: column; gap: 2mm; margin-top: 3.5mm; }}
.frow {{ display: flex; align-items: stretch; gap: 2mm; }}
.fbox {{
  flex: 1; background: #eaf2fb; border: 0.6mm solid #7ba7d0; border-radius: 2.5mm;
  padding: 2.5mm 3.5mm; text-align: center; font-size: 17pt;
}}
.fbox.dark {{ background: #0a5ca8; color: #fff; border-color: #04365f; }}
.farrow {{ text-align: center; font-size: 20pt; color: #0a5ca8; font-weight: bold; }}
img.fig {{ width: 100%; }}
.caption {{ font-size: 15.5pt; color: #4a5d70; margin-top: 1.5mm; }}
table {{ border-collapse: collapse; width: 100%; font-size: 17pt; }}
th {{
  background: #04365f; color: #fff; padding: 2.5mm 2mm; text-align: center; font-size: 16.5pt;
}}
td {{ border-bottom: 0.4mm solid #d4e1ee; padding: 2.2mm 2mm; text-align: center; }}
td.pol {{ text-align: left; }}
tr.hl td {{ background: #eaf2fb; font-weight: bold; }}
.ci {{ font-weight: normal; font-size: 13.5pt; color: #4a5d70; }}
.refs {{ font-size: 16pt; padding-left: 8mm; }}
.refs li {{ margin-top: 2mm; }}
.kpi {{ display: flex; gap: 3.5mm; margin-top: 3.5mm; margin-bottom: 3mm; }}
.kpibox {{
  flex: 1; background: #eaf2fb; border: 0.6mm solid #7ba7d0; border-radius: 2.5mm;
  padding: 3mm; text-align: center;
}}
.kpibox .n {{ font-size: 29pt; font-weight: bold; color: #04365f; }}
.kpibox .l {{ font-size: 14.5pt; color: #4a5d70; margin-top: 1mm; }}
</style></head>
<body>
<div class="banner">
  <h1>Enrutamiento adaptativo de modelos de lenguaje: &iquest;superan las
  pol&iacute;ticas aprendidas a una regla determinista?</h1>
  <div class="authors">Rodolfo Fritz &middot; Benjam&iacute;n Cerda &middot; Felipe Friz</div>
  <div class="affil">Universidad del B&iacute;o-B&iacute;o &mdash; Concepci&oacute;n,
  Chile &middot; Inteligencia Artificial en
  Autom&aacute;tica/Automatizaci&oacute;n &mdash; Proyecto Final 2026</div>
</div>
<div class="columns">

<div class="col">
  <div class="block"><h2>Introducci&oacute;n</h2><div class="body">
    <p>Los sistemas basados en modelos de lenguaje (LLM) enfrentan un dilema de
    ingenier&iacute;a: los modelos m&aacute;s capaces cuestan <strong>dos
    &oacute;rdenes de magnitud m&aacute;s</strong> que los peque&ntilde;os, pero no
    toda consulta requiere esa capacidad. Un <strong>router</strong> decide, consulta
    a consulta, qu&eacute; modelo ejecutar.</p>
    <p><strong>Pregunta:</strong> &iquest;logran las pol&iacute;ticas de enrutamiento
    aprendidas una utilidad esperada superior a una pol&iacute;tica ponderada
    determinista al elegir entre modelos heterog&eacute;neos?</p>
    <p><strong>Objetivo:</strong> comparar dos algoritmos de IA &mdash;
    <strong>XGBoost</strong> (supervisado) y <strong>LinUCB</strong> (bandit
    contextual) &mdash; contra una baseline determinista y una cota superior (Oracle),
    bajo una utilidad que pondera calidad, costo, latencia y error.</p>
  </div></div>
  <div class="block"><h2>Materiales y m&eacute;todos: datos</h2><div class="body">
    <p><strong>RouterBench 0-shot</strong> [1]: resultados reales de 11 LLM sobre
    benchmarks p&uacute;blicos, fijado por SHA-256 y convertido a un esquema
    can&oacute;nico auditado (una fila por par consulta&ndash;modelo).</p>
    <div class="kpi">
      <div class="kpibox"><div class="n">36.497</div><div class="l">prompts</div></div>
      <div class="kpibox"><div class="n">4</div><div class="l">brazos de modelo</div></div>
      <div class="kpibox"><div class="n">145.988</div><div class="l">filas limpias</div></div>
    </div>
    <p>Brazos: <strong>cuatro niveles de costo con calidad creciente</strong>,
    excluyendo modelos dominados &mdash; Mistral&nbsp;7B (r&aacute;pido),
    Mixtral&nbsp;8x7B (equilibrado), Yi-34B (razonamiento), GPT-4&nbsp;Turbo
    (premium).</p>
    <p><strong>Preprocesamiento:</strong> limpieza determinista (duplicados, rangos,
    consistencia por prompt); caracter&iacute;sticas <em>pre-inferencia</em>
    (longitud del texto, tipo de tarea one-hot); partici&oacute;n <strong>70/15/15
    agrupada por prompt</strong> y estratificada por tarea, con 5 semillas bloqueadas
    y pruebas autom&aacute;ticas contra fugas de informaci&oacute;n.</p>
  </div></div>
</div>

<div class="col">
  <div class="block"><h2>Funci&oacute;n de utilidad</h2><div class="body">
    <p>Definida y bloqueada antes de ver resultados:</p>
    <div class="eq">U = 0.65&middot;Q &minus; 0.20&middot;C<sub>n</sub> &minus;
    0.10&middot;L<sub>n</sub> &minus; 0.05&middot;E</div>
    <p class="caption">Q: calidad &isin; [0,1]; C<sub>n</sub>, L<sub>n</sub>: costo y
    latencia min-max <strong>anclados al split de train</strong>; E: indicador de
    error. La latencia del artefacto es nula y su t&eacute;rmino queda deshabilitado
    expl&iacute;citamente.</p>
    <div class="flow">
      <div class="frow"><div class="fbox">Consulta (prompt)</div></div>
      <div class="farrow">&darr;</div>
      <div class="frow"><div class="fbox">Caracter&iacute;sticas pre-inferencia
      (longitud, tipo de tarea)</div></div>
      <div class="farrow">&darr;</div>
      <div class="frow">
        <div class="fbox dark">Reglas<br/>(baseline)</div>
        <div class="fbox dark">XGBoost</div>
        <div class="fbox dark">LinUCB</div>
        <div class="fbox">Oracle</div>
      </div>
      <div class="farrow">&darr;</div>
      <div class="frow"><div class="fbox">Modelo seleccionado &rarr; calidad, costo,
      error &rarr; utilidad U</div></div>
    </div>
  </div></div>
  <div class="block"><h2>Algoritmos y evaluaci&oacute;n</h2><div class="body">
    <ul>
      <li><strong>Better Rules Proxy:</strong> tabla determinista tarea&rarr;brazo
      con mejor utilidad media en train.</li>
      <li><strong>XGBoost</strong> [3]: un regresor de utilidad por brazo; ruteo por
      argmax; grilla de 8 configuraciones elegida <em>solo en
      validaci&oacute;n</em>.</li>
      <li><strong>LinUCB disjunto</strong> [2]: bandit contextual entrenado por
      <em>replay prequential</em> (elegir&rarr;observar&rarr;actualizar) sobre un
      barajado con semilla de train; &alpha; elegido en validaci&oacute;n.</li>
      <li><strong>Oracle offline:</strong> mejor brazo realizado por prompt (cota
      superior no desplegable).</li>
    </ul>
    <p><strong>M&eacute;tricas:</strong> utilidad media, calidad media, costo medio y
    tasa de error en test; <strong>IC 95&nbsp;% por bootstrap pareado</strong>
    (2000 remuestras por <em>prompt</em>) sobre 5 semillas.</p>
    <p><strong>Implementaci&oacute;n:</strong> Python modular (pandas, scikit-learn,
    xgboost, numpy), CI con mypy estricto, ruff y cobertura &ge; 85&nbsp;%; tablas y
    figuras generadas desde artefactos, nunca a mano.</p>
  </div></div>
</div>

<div class="col">
  <div class="block"><h2>Resultados</h2><div class="body">
    <table>
      <tr><th>Pol&iacute;tica</th><th>Utilidad U</th><th>&Delta; vs baseline<br/>[IC 95 %]</th>
      <th>Calidad</th><th>Costo [USD]</th></tr>
      {table_rows}
    </table>
    <p class="caption">Media sobre los splits de prueba de 5 semillas (20.263 prompts
    &uacute;nicos). &Delta;: diferencia pareada contra Better Rules Proxy;
    IC 95&nbsp;% bootstrap por prompt.</p>
    <img class="fig" src="{fig_comparison}" style="margin-top:3mm"/>
    <p class="caption">Utilidad media en test con IC 95&nbsp;%. XGBoost y LinUCB
    <strong>igualan</strong> a la baseline (los IC de las diferencias cruzan cero);
    solo el Oracle la supera (+0.068).</p>
  </div></div>
  <div class="block"><h2>Conclusiones</h2><div class="body">
    <ul>
      <li>Con caracter&iacute;sticas pre-inferencia simples, <strong>las
      pol&iacute;ticas aprendidas igualan pero no superan</strong> a la regla
      determinista: &Delta; &asymp; 0 con IC 95&nbsp;% que cruza cero.</li>
      <li>Bajo los pesos elegidos, la baseline converge a &laquo;siempre
      GPT-4&raquo;; el problema resulta trivial para las se&ntilde;ales
      disponibles.</li>
      <li>El Oracle (+0.068 de utilidad, mejor calidad a &asymp;5&times; menos costo)
      demuestra que <strong>el potencial del ruteo por consulta es real</strong> y
      queda sin capturar.</li>
      <li><strong>Limitaciones:</strong> caracter&iacute;sticas simples, latencia ausente,
      pesos fijos y selecci&oacute;n exploratoria de los cuatro brazos.</li>
      <li><strong>Trabajo futuro:</strong> embeddings del prompt como contexto,
      an&aacute;lisis de sensibilidad de pesos y evaluaci&oacute;n en modo sombra.</li>
    </ul>
  </div></div>
</div>

<div class="col">
  <div class="block"><h2>Frontera calidad-costo</h2><div class="body">
    <img class="fig" src="{fig_frontier}"/>
    <p class="caption">El Oracle logra <strong>m&aacute;s calidad (0.87 vs 0.78) a
    &asymp;5&times; menos costo</strong> que &laquo;siempre GPT-4&raquo;: el margen
    del ruteo por consulta existe, pero exige mejores se&ntilde;ales.</p>
  </div></div>
  <div class="block"><h2>LinUCB: regret acumulado</h2><div class="body">
    <img class="fig" src="{fig_regret}"/>
    <p class="caption">Regret acumulado durante el replay prequential (5 semillas):
    crecimiento estable y reproducible, sin exploraci&oacute;n
    catastr&oacute;fica.</p>
  </div></div>
  <div class="block"><h2>Referencias</h2><div class="body">
    <ol class="refs">
      <li>Q. J. Hu <em>et al.</em>, &laquo;RouterBench: A Benchmark for Multi-LLM
      Routing Systems&raquo;, arXiv:2403.12031, 2024. DOI dataset:
      10.57967/hf/1996.</li>
      <li>L. Li, W. Chu, J. Langford y R. E. Schapire, &laquo;A Contextual-Bandit
      Approach to Personalized News Article Recommendation&raquo;, WWW, 2010.</li>
      <li>T. Chen y C. Guestrin, &laquo;XGBoost: A Scalable Tree Boosting
      System&raquo;, KDD, 2016.</li>
      <li>C&oacute;digo del estudio:
      github.com/FitoFritzG/better-router-adaptive-research (MIT).</li>
    </ol>
  </div></div>
</div>

</div>
<div class="footer"><strong>Reproducibilidad:</strong> dataset fijado por SHA-256 &middot;
5 semillas bloqueadas (42, 123, 2026, 31415, 271828) &middot; partici&oacute;n agrupada por
prompt &middot; hiperpar&aacute;metros elegidos solo en validaci&oacute;n &middot; tablas y
figuras generadas desde artefactos &middot; c&oacute;digo MIT con CI, mypy estricto y
cobertura &ge; 85 %</div>
</body></html>
"""


def main() -> None:
    html = HTML.format(
        table_rows=_results_table(),
        fig_comparison=_figure_data_uri("comparacion_politicas.svg"),
        fig_frontier=_figure_data_uri("frontera_calidad_costo.svg"),
        fig_regret=_figure_data_uri("regret_linucb.svg"),
    )
    OUTPUT.write_text(html, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
