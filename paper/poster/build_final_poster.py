from __future__ import annotations

from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "poster_evocascade_ideal.html"

HTML = r"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<style>
@page { size: A1 landscape; margin: 0; }
* { box-sizing: border-box; }
body { margin:0; width:841mm; height:594mm; background:#edf3f8; color:#172635;
  font-family:"DejaVu Sans",Arial,sans-serif; font-size:14.7pt; line-height:1.28; position:relative; }
header { background:linear-gradient(180deg,#04365f,#0a5ca8); color:white; padding:9mm 15mm 7mm; }
h1 { margin:0; font-size:38pt; line-height:1.05; }
.authors { margin-top:3mm; font-size:19pt; }
.affil { margin-top:1mm; font-size:12.5pt; color:#d6e8f8; }
.warning { margin:2.5mm 8mm 0; padding:2.5mm 4mm; background:#fff7df; border:0.6mm solid #b7791f;
  border-radius:2mm; color:#5c3900; font-weight:bold; font-size:12.5pt; }
.grid { display:grid; grid-template-columns:1fr 1fr 1.08fr; gap:5mm; padding:3mm 6mm 11mm; }
.card { background:white; border:0.45mm solid #b9cde2; border-radius:2.5mm; overflow:hidden; margin-bottom:4mm; break-inside:avoid; }
.card h2 { margin:0; padding:2.3mm 4mm; color:white; font-size:18pt; background:linear-gradient(180deg,#0a5ca8,#04365f); }
.card .body { padding:3.3mm 4mm 3.8mm; }
p { margin:0 0 2mm; }
ul,ol { margin:0.5mm 0 0; padding-left:6mm; }
li { margin:1mm 0; }
strong { color:#04365f; }
.eq { text-align:center; background:#eaf2fb; border:0.4mm solid #9ebfdd; border-radius:2mm;
  padding:2.7mm; margin:2.5mm 0; color:#04365f; font-size:18pt; font-weight:bold; }
.kpis { display:flex; gap:2.5mm; }
.kpi { flex:1; text-align:center; background:#eaf2fb; border:0.4mm solid #9ebfdd; border-radius:2mm; padding:2.4mm; }
.kpi b { display:block; color:#04365f; font-size:21pt; }
.kpi span { font-size:10.5pt; color:#4a5d70; }
.flow { display:flex; flex-direction:column; gap:1.4mm; }
.box { text-align:center; background:#eaf2fb; border:0.4mm solid #7ba7d0; border-radius:1.5mm; padding:2mm; }
.box.dark { background:#0a5ca8; color:white; }
.arrow { text-align:center; color:#0a5ca8; font-weight:bold; font-size:16pt; line-height:1; }
table { width:100%; border-collapse:collapse; font-size:11.5pt; }
th { color:white; background:#04365f; padding:1.5mm 0.8mm; }
td { padding:1.35mm 0.8mm; text-align:center; border-bottom:0.3mm solid #d4e1ee; }
td:first-child { text-align:left; }
.hl td { background:#eaf2fb; font-weight:bold; }
.small { font-size:10.5pt; color:#4a5d70; }
.bar-row { display:grid; grid-template-columns:38mm 1fr 18mm; gap:2mm; align-items:center; margin:1.4mm 0; font-size:11pt; }
.track { height:6mm; background:#e4edf5; border-radius:1mm; overflow:hidden; }
.bar { height:100%; background:#0a5ca8; }
.bar.evo { background:#14865f; }
.bar.oracle { background:#6b4cb3; }
.refs { font-size:9.8pt; }
.bottom { display:grid; grid-template-columns:1fr 1fr 1fr; gap:5mm; margin:0 6mm 11mm; }
.bottom .card { margin-bottom:0; }
footer { position:absolute; left:0; right:0; bottom:0; padding:3mm 12mm; text-align:center;
  color:#d6e8f8; background:linear-gradient(180deg,#0a5ca8,#04365f); font-size:10.5pt; }
</style></head><body>
<header>
<h1>Enrutamiento adaptativo de modelos de lenguaje: ¿superan las políticas aprendidas a una regla determinista?</h1>
<div class="authors">Rodolfo Fritz · Benjamín Cerda · Felipe Friz</div>
<div class="affil">Universidad del Bío-Bío · Ingeniería Civil en Automatización · Inteligencia Artificial en Automática/Automatización · Proyecto Final 2026</div>
</header>
<div class="warning">EvoCascade-Ideal utiliza un verificador perfecto simulado con la etiqueta del benchmark. Su mejora es una cota superior experimental, no desempeño directamente desplegable.</div>
<div class="grid">
<section>
<div class="card"><h2>Problema de ingeniería</h2><div class="body">
<p>Los modelos de lenguaje difieren en capacidad, costo y confiabilidad. Ejecutar siempre el modelo más potente desperdicia recursos; elegir siempre el más barato reduce la calidad.</p>
<p><strong>Pregunta:</strong> ¿una política aprendida puede seleccionar por consulta el modelo que maximiza una utilidad conjunta?</p>
<p><strong>Objetivo:</strong> comparar dos algoritmos de IA —XGBoost y LinUCB— contra una regla determinista, brazos fijos y un Oracle offline.</p>
</div></div>
<div class="card"><h2>Dataset y preparación</h2><div class="body">
<p><strong>RouterBench 0-shot:</strong> resultados de modelos heterogéneos sobre benchmarks públicos. El artefacto se fija por revisión, tamaño y SHA-256.</p>
<div class="kpis"><div class="kpi"><b>36.497</b><span>prompts</span></div><div class="kpi"><b>4</b><span>modelos</span></div><div class="kpi"><b>145.988</b><span>filas</span></div></div>
<ul>
<li>limpieza de duplicados, rangos y consistencia por prompt;</li>
<li>transformación a esquema canónico <code>(prompt_id, model_id)</code>;</li>
<li>características pre-inferencia: longitud, estructura y grupo de tarea;</li>
<li>split 70/15/15 agrupado por prompt y estratificado;</li>
<li>normalización ajustada únicamente con train;</li>
<li>cinco semillas bloqueadas para estimar estabilidad.</li>
</ul>
<p class="small">El dataset procesado no se redistribuye: se entregan scripts, checksums y resultados agregados.</p>
</div></div>
<div class="card"><h2>Función de utilidad</h2><div class="body">
<div class="eq">U = 0,65·Q − 0,20·Cₙ − 0,10·Lₙ − 0,05·E</div>
<p><strong>Q:</strong> calidad; <strong>Cₙ:</strong> costo normalizado; <strong>Lₙ:</strong> latencia normalizada; <strong>E:</strong> error.</p>
<p class="small">RouterBench no aporta latencia en el artefacto usado. El término queda deshabilitado sin renormalizar los pesos restantes.</p>
</div></div>
<div class="card"><h2>Decisiones metodológicas</h2><div class="body"><ul>
<li>ningún prompt aparece en más de un split;</li>
<li>hiperparámetros elegidos exclusivamente en validación;</li>
<li>test usado para la evaluación final congelada;</li>
<li>bootstrap pareado por prompt, 2.000 remuestras;</li>
<li>resultados negativos conservados y discutidos.</li>
</ul></div></div>
</section>
<section>
<div class="card"><h2>Algoritmos de IA</h2><div class="body">
<ul>
<li><strong>XGBoost:</strong> un regresor de utilidad por modelo; selección por argmax.</li>
<li><strong>LinUCB:</strong> bandit contextual disjunto con replay prequential; α ajustado en validación.</li>
<li><strong>EvoCascade-Ideal (bonus):</strong> política lineal optimizada con sep-CMA-ES sobre acciones simples y cascadas.</li>
</ul>
<p><strong>Comparadores:</strong> Better Rules Proxy, cuatro brazos fijos y Oracle offline.</p>
</div></div>
<div class="card"><h2>Diseño experimental</h2><div class="body">
<ul>
<li>semillas: 42, 123, 2026, 31415 y 271828;</li>
<li>métricas: utilidad, calidad, costo, error, regret e IC 95 %;</li>
<li>evaluación multi-semilla y diferencias pareadas;</li>
<li>CI en Python 3.12 y 3.13;</li>
<li>cobertura ≥85 %, Ruff, formato, mypy estricto y build.</li>
</ul>
</div></div>
<div class="card"><h2>Mecanismo de cascada</h2><div class="body">
<div class="flow"><div class="box">Consulta</div><div class="arrow">↓</div><div class="box dark">Modelo intermedio económico</div><div class="arrow">↓</div><div class="box">Verificador: ¿respuesta aceptable?</div><div class="arrow">↓</div><div style="display:flex;gap:2mm"><div class="box" style="flex:1">Sí (~75 %)<br/>aceptar</div><div class="box dark" style="flex:1">No (~25 %)<br/>escalar a GPT-4</div></div></div>
<p class="small">En el estudio, el verificador observa si la calidad verdadera es cero o si la llamada falla. Esa señal no está disponible automáticamente en producción.</p>
</div></div>
<div class="card"><h2>Ablación del verificador</h2><div class="body">
<table><tr><th>Escenario</th><th>U</th><th>Δ baseline</th></tr>
<tr class="hl"><td>Verificador ideal</td><td>0,52846</td><td>+0,03230</td></tr>
<tr><td>Escalar siempre</td><td>0,49568</td><td>-0,00048</td></tr>
<tr><td>Nunca escalar</td><td>0,43113</td><td>-0,06503</td></tr></table>
<div class="kpis" style="margin-top:2.5mm"><div class="kpi"><b>83,3 %</b><span>acciones cascada</span></div><div class="kpi"><b>24,8 %</b><span>escaladas ideales</span></div></div>
<p class="small">La mejora depende de detectar correctamente cuándo escalar; sin esa detección, la política no supera la baseline.</p>
</div></div>
<div class="card"><h2>Reproducibilidad</h2><div class="body"><p>Paquete Python modular, adquisición verificable, configuración versionada, resultados agregados, pruebas unitarias e integración, documentación en español y ZIP académico generado automáticamente.</p></div></div>
</section>
<section>
<div class="card"><h2>Resultados principales</h2><div class="body">
<table><tr><th>Política</th><th>U</th><th>Δ</th><th>IC 95 % Δ</th></tr>
<tr class="hl"><td>Oracle offline</td><td>0,5641</td><td>+0,0680</td><td>[0,0656; 0,0705]</td></tr>
<tr class="hl"><td>EvoCascade-Ideal</td><td>0,5282</td><td>+0,0321</td><td>[0,0302; 0,0341]</td></tr>
<tr class="hl"><td>Better Rules Proxy</td><td>0,4961</td><td>0</td><td>[0; 0]</td></tr>
<tr><td>XGBoost</td><td>0,4961</td><td>≈0</td><td>[-0,0005; 0,0006]</td></tr>
<tr><td>LinUCB</td><td>0,4959</td><td>-0,0001</td><td>[-0,0005; 0,0002]</td></tr>
<tr><td>Yi-34B fijo</td><td>0,4217</td><td>-0,0744</td><td>[-0,0783; -0,0704]</td></tr>
<tr><td>Mixtral fijo</td><td>0,3559</td><td>-0,1401</td><td>[-0,1445; -0,1358]</td></tr>
<tr><td>Mistral fijo</td><td>0,1979</td><td>-0,2982</td><td>[-0,3031; -0,2932]</td></tr></table>
<div style="margin-top:3mm">
<div class="bar-row"><span>Oracle</span><div class="track"><div class="bar oracle" style="width:100%"></div></div><b>0,564</b></div>
<div class="bar-row"><span>EvoCascade</span><div class="track"><div class="bar evo" style="width:93.6%"></div></div><b>0,528</b></div>
<div class="bar-row"><span>Baseline</span><div class="track"><div class="bar" style="width:87.9%"></div></div><b>0,496</b></div>
<div class="bar-row"><span>XGBoost</span><div class="track"><div class="bar" style="width:87.9%"></div></div><b>0,496</b></div>
<div class="bar-row"><span>LinUCB</span><div class="track"><div class="bar" style="width:87.9%"></div></div><b>0,496</b></div>
</div>
<p class="small">XGBoost y LinUCB no superan significativamente la baseline. EvoCascade-Ideal captura cerca del 47 % de la brecha hacia el Oracle bajo verificación perfecta simulada.</p>
</div></div>
<div class="card"><h2>Interpretación y limitaciones</h2><div class="body"><ul>
<li>El Oracle confirma que existe margen para routing por consulta.</li>
<li>Las características simples no permiten a XGBoost ni LinUCB capturar ese margen.</li>
<li>EvoCascade-Ideal cuantifica el potencial de una verificación post-inferencia selectiva.</li>
<li>El verificador usa ground truth; no es desplegable.</li>
<li>El benchmark es offline, sin latencia real ni dinámica de proveedores.</li>
<li>Los cuatro modelos se seleccionaron exploratoriamente.</li>
</ul></div></div>
<div class="card"><h2>Conclusión defendible</h2><div class="body">
<p>La verificación post-inferencia puede recuperar parte de la brecha entre una política fija y el Oracle, pero su valor depende críticamente de la precisión, costo y latencia de un verificador real.</p>
<p><strong>Trabajo futuro:</strong> entrenar un verificador y medir sensibilidad, especificidad, falsos positivos, falsos negativos y costo total.</p>
<p class="small">No se demuestra superioridad frente a Sakana Fugu, sistemas comerciales ni routing productivo.</p>
</div></div>
<div class="card"><h2>Referencias</h2><div class="body"><ol class="refs">
<li>Hu et al., RouterBench, arXiv:2403.12031, 2024.</li>
<li>Chen y Guestrin, XGBoost, KDD, 2016.</li>
<li>Li et al., Contextual Bandit, WWW, 2010.</li>
<li>Ros y Hansen, sep-CMA-ES, PPSN, 2008.</li>
<li>Sakana AI, TRINITY y Conductor, 2025.</li>
<li>github.com/FitoFritzG/better-router-adaptive-research.</li>
</ol></div></div>
</section>
</div>
<div class="bottom">
<div class="card"><h2>Cumplimiento de la pauta</h2><div class="body"><p><strong>Datos:</strong> limpieza, transformación, normalización y características.</p><p><strong>IA:</strong> XGBoost y LinUCB implementados y comparados.</p><p><strong>Evaluación:</strong> seis métricas cuantitativas e IC 95 %.</p></div></div>
<div class="card"><h2>Bonus implementado</h2><div class="body"><p>Análisis de hiperparámetros, evaluación multi-semilla, bootstrap pareado y extensión EvoCascade-Ideal con sep-CMA-ES.</p><p class="small">La mejora adicional está implementada, probada y documentada con sus limitaciones.</p></div></div>
<div class="card"><h2>Archivos entregados</h2><div class="body"><p>Póster PDF, código Python modular, configuración, dependencias, pruebas, resultados agregados, documentación y verificador automático del ZIP.</p><p class="small">No se incluyen secretos, cachés ni datos sin licencia de redistribución.</p></div></div>
</div>
<footer><strong>Entrega reproducible:</strong> procesamiento de datos · XGBoost y LinUCB · métricas cuantitativas · análisis crítico · bonus de hiperparámetros y EvoCascade · código MIT con CI</footer>
</body></html>"""


def main() -> None:
    OUTPUT.write_text(HTML, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
