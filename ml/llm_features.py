"""LLM-scalar transcript features — the n-safe text lever.

The ablation showed sample size + feature dimensionality are the bottleneck (in-domain CV
~0, only low-dim signals like sentiment transfer). So instead of high-dim text embeddings,
have an LLM read the WHOLE assessed-role transcript and emit a handful of session-level
construct ratings (1-5). Document-level => no functional aggregation, low-dim => learnable.

Local ollama (local inference). German transcripts -> gemma3.
Output: ml/llm/<dataset>/<session>.json with the construct scores.
"""
import os
import re
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import ollama
import discover_utils.data.handler.nova_db_handler as mh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsconfig as ds_cfg

MODEL = 'gemma3:12b'
OUT = Path(__file__).resolve().parent / 'llm'
# v2: dialogue-context variant — merged two-speaker transcript with dataset-neutral role
# labels (B = bewertete Person, G = Gegenüber), so 'eingehen' is judged with the partner's
# turns visible and no profession label (Arzt/Lehrer) can drag domain-specific standards in.
MODEL2 = 'gemma3:12b'   # dialog-variant model; see MODEL_CFG + bake-off notes below
OUT2 = Path(__file__).resolve().parent / 'llm_dialog'

# Per-model generation configs, ported from an in-house sampling-profile table (model-card
# values, structured-JSON task): temperature 0.4 — NOT greedy, every current card
# warns against temp 0 (Qwen: endless repetitions; Gemma: degeneration; plausibly
# part of gemma4's all-5s collapse under temp~0/top_k=1). presence_penalty 0 so JSON
# structural tokens aren't penalised. think=False strips <think> blocks on reasoning
# families (qwen3.x, gemma4, glm) — non-thinking models ignore it (verified in-house).
# seed pinned for reproducible feature extraction despite temp>0; parse-failure
# retries bump the seed (resample instead of climbing a temperature ladder).
MODEL_CFG = {
    'gemma3:12b':  {'temperature': 0.4, 'top_p': 0.95, 'top_k': 64, 'repeat_penalty': 1.0},
    'gemma4:12b':  {'temperature': 0.4, 'top_p': 0.95, 'top_k': 64, 'repeat_penalty': 1.0},
    'qwen3:14b':   {'temperature': 0.4, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 0.0, 'repeat_penalty': 1.0},
    'qwen3.5:27b': {'temperature': 0.4, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 0.0, 'repeat_penalty': 1.0},
    # phase-10c bake-off round 3 (remote 4090, ollama 0.32.1, label-free screen):
    # qwen3.6 = HF-card nothink values (validated in-house); phi4/granite ship no
    # vendor recs -> neutral profile. Screen verdicts: qwen3.6 clean pass (std .80),
    # phi4 marginal pass (std .37, band-compressed), granite4.1:8b ceiling collapse.
    'qwen3.6:27b': {'temperature': 0.7, 'top_p': 0.80, 'top_k': 20, 'min_p': 0.0,
                    'presence_penalty': 1.5, 'repeat_penalty': 1.0},
    'phi4:14b': {'temperature': 0.4, 'top_p': 0.95, 'repeat_penalty': 1.0},
    'granite4.1:8b': {'temperature': 0.4, 'top_p': 0.95, 'repeat_penalty': 1.0},
    'granite4.1:30b': {'temperature': 0.4, 'top_p': 0.95, 'repeat_penalty': 1.0},
}
# think=True variants — model-card thinking-mode values (qwen3.x: temp 0.6/top_p 0.95/
# top_k 20; gemma4 analog). Interactive deployments typically disable thinking because latency is crucial
# there; here it is not, so thinking may buy rating quality. gemma3 has no thinking mode.
MODEL_CFG_THINK = {
    'gemma4:12b':  {'temperature': 0.6, 'top_p': 0.95, 'top_k': 64, 'repeat_penalty': 1.0},
    'qwen3:14b':   {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'presence_penalty': 0.0, 'repeat_penalty': 1.0},
    'qwen3.5:27b': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'presence_penalty': 0.0, 'repeat_penalty': 1.0},
    # qwen3.6 think = HF-card thinking values (in-house 'think' profile)
    'qwen3.6:27b': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0,
                    'presence_penalty': 0.0, 'repeat_penalty': 1.0},
}
MODEL3 = 'qwen3:14b'    # think bake-off 2026-07-02: qwen3:14b fast + valid JSON + real spread;
                        # gemma4 think = ceiling-compressed again (+1 parse fail);
                        # qwen3.5:27b think = mostly unparseable at 4096 tokens
OUT3 = Path(__file__).resolve().parent / 'llm_dialog_think'


def _generate(model, prompt, seed=0, num_predict=None, think=False):
    cfg = MODEL_CFG_THINK[model] if think else MODEL_CFG[model]
    # think=True needs headroom for reasoning tokens before the JSON
    options = dict(cfg, seed=seed, num_predict=num_predict or (4096 if think else 256))
    # explicit context: the longest prompt in the corpus is 5.1k tokens (theory battery
    # on the 15.2k-char teacher session; German runs ~3.2 chars/token), so the 16k-char
    # slice in query_dialog never binds and nothing is truncated. Local runs
    # used llama-server -c 32768 (verified, 0 truncations in logs) — pin >= that need
    # so fresh installs with a smaller default behave identically
    options.setdefault('num_ctx', 8192)
    try:
        r = ollama.generate(model=model, prompt=prompt, think=think, options=options)
    except TypeError:                      # older ollama lib without the think kwarg
        r = ollama.generate(model=model, prompt=prompt, options=options)
    return r['response']
CONSTRUCTS = ['empathie', 'struktur', 'klarheit', 'eingehen', 'engagement']  # de: empathy/structure/clarity/responsiveness/engagement

PROMPT = """Du bewertest die Gesprächsführung einer Person (Rolle: {role}) anhand eines Transkripts ihrer Äußerungen.
Bewerte NUR diese Person auf einer Skala von 1 (sehr schlecht) bis 5 (sehr gut) in diesen Dimensionen:
- empathie: einfühlsam, geht auf Gefühle ein
- struktur: klar gegliedert, roter Faden
- klarheit: verständlich, präzise formuliert
- eingehen: reagiert auf das Gegenüber, stellt Fragen
- engagement: aktiv, zugewandt, engagiert

Transkript:
{text}

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt, keine Erklärung:
{{"empathie": <1-5>, "struktur": <1-5>, "klarheit": <1-5>, "eingehen": <1-5>, "engagement": <1-5>}}"""


PROMPT2 = """Du bewertest die Gesprächsführung der Person "B" anhand des folgenden Gesprächstranskripts.
"B" ist die zu bewertende Person, "G" ist ihr Gesprächspartner bzw. ihre Gesprächspartnerin.
Bewerte NUR Person B auf einer Skala von 1 (sehr schlecht) bis 5 (sehr gut) in diesen Dimensionen:
- empathie: einfühlsam, geht auf Gefühle von G ein
- struktur: klar gegliedert, roter Faden
- klarheit: verständlich, präzise formuliert
- eingehen: reagiert auf das von G Gesagte, greift es auf, stellt Fragen
- engagement: aktiv, zugewandt, engagiert

Sei streng und differenziert: 3 bedeutet durchschnittlich, 4 gut, 5 nur bei außergewöhnlich
guter Gesprächsführung ohne erkennbare Schwächen. Nutze die gesamte Skala; achte auch auf
Schwächen (übergangene Einwände, fehlende Struktur, Monologisieren, vage Aussagen).

Transkript:
{text}

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt, keine Erklärung:
{{"empathie": <1-5>, "struktur": <1-5>, "klarheit": <1-5>, "eingehen": <1-5>, "engagement": <1-5>}}"""


# v4: theory-driven constructs, anchored on the actual expert instruments
# (expert instrument documents, not released): BGR items (Scheffer; Empathie / Struktur / verbaler
# Ausdruck / Gesamteindruck) + the OSCE communication checklist (Spiegeln, NURSE,
# Aktives Zuhören, Kohärenz — each with its 1-5 anchor wording). Nonverbal BGR item
# omitted (transcript judge). All scales oriented 5 = best (gesamtnote = inverted
# Schulnote), matching the target after dsconfig inversion.
CONSTRUCTS4 = ['spiegeln', 'nurse', 'aktives_zuhoeren', 'kohaerenz',
               'bgr_empathie', 'bgr_verbal', 'gesamtnote', 'partner_zufrieden']
OUT4 = Path(__file__).resolve().parent / 'llm_theory'
MODEL4 = 'gemma3:12b'

PROMPT4 = """Du bewertest die Gesprächsführung der Person "B" anhand des folgenden Gesprächstranskripts.
"B" ist die zu bewertende Person, "G" ist ihr Gesprächspartner bzw. ihre Gesprächspartnerin.
Bewerte NUR Person B nach diesen Kriterien aus Expertenbögen für Gesprächsführung (alle Skalen: 5 = am besten):

- spiegeln: Paraphrasieren (Gesagtes in eigenen Worten wiedergeben) / Verbalisieren (vermutete oder geäußerte Emotionen benennen). 1 = spiegelt nie; 2 = an wenig treffender Stelle oder sprachlich hölzern; 3 = an treffender Stelle, aber wenig authentisch/elaboriert; 4 = an treffender Stelle, elaboriert und authentisch; 5 = zusätzlich als fragendes Angebot, sodass G sich dazu äußern kann.
- nurse: benennt Emotionen von G, zeigt Verständnis für die Lage/Sicht von G, respektiert Gefühle/Sicht von G, bietet Unterstützung an, exploriert bei Unklarheit. 1 = nichts davon; 2 = ein Aspekt an wenig treffender Stelle oder hölzern; 3 = ein Aspekt, wenig authentisch/elaboriert; 4 = ein Aspekt an treffender Stelle, elaboriert und authentisch; 5 = mehrere Aspekte an treffenden Stellen, elaboriert und authentisch.
- aktives_zuhoeren: offene Fragen, ausreden lassen, Hörsignale, zur Weiterrede ermutigen, Nachfragen, Gehörtes zusammenfassen. 1 = kein aktives Zuhören; 2 = selten oder an unpassender Stelle; 3 = mehrfach, aber wenig Variation; 4 = häufig und vielfältig, aber noch ausbaufähig; 5 = durchgängig angemessener, vielfältiger Einsatz.
- kohaerenz: Stimmigkeit und roter Faden der Gesprächsführung. 1 = keine erkennbare Struktur, G muss die Führung übernehmen; 3 = schablonenhaft, minimal flexibel oder inkonsistente Führung; 5 = sehr gute Struktur, Kontrolle über das Gespräch, transparente Überleitungen, flexibel.
- bgr_empathie: 5 = geht durchgehend verständnisvoll auf Hinweise und Bedürfnisse von G ein bzw. reagiert angemessen; 1 = geht nicht auf offensichtliche Hinweise/Bedürfnisse ein oder reagiert unangemessen.
- bgr_verbal: 5 = kommuniziert so, dass G sie/ihn leicht versteht (Wortwahl, angemessene Ausdrucksweise); 1 = erschwert das Verstehen.
- gesamtnote: Gesamteindruck des Gesprächs als invertierte Schulnote, 5 = sehr gut, 1 = mangelhaft.
- partner_zufrieden: Wie zufrieden verlässt G vermutlich das Gespräch? 5 = sehr zufrieden, 1 = sehr unzufrieden.

Sei streng und differenziert: 3 bedeutet durchschnittlich, 5 nur ohne erkennbare Schwächen. Nutze die gesamte Skala.

Transkript:
{text}

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt, keine Erklärung:
{{"spiegeln": <1-5>, "nurse": <1-5>, "aktives_zuhoeren": <1-5>, "kohaerenz": <1-5>, "bgr_empathie": <1-5>, "bgr_verbal": <1-5>, "gesamtnote": <1-5>, "partner_zufrieden": <1-5>}}"""


# v5: conversational phases + techniques, anchored on a published phase/technique annotation guideline
# (Benien-2003 phase model + Gerich-2015
# techniques; guideline cited in the paper). Session-level quality scalars, not per-sentence codes.
# Overlaps llm4 deliberately (spiegeln ~ paraphrasieren+verbalisieren, kohaerenz ~
# strukturieren) — different instrument/granularity; PCA handles collinearity.
CONSTRUCTS5 = ['anfang_rahmen', 'phasen_vollstaendig', 'beschluss_konkret',
               'abschluss_qualitaet', 'verbalisieren', 'paraphrasieren', 'strukturieren']
OUT5 = Path(__file__).resolve().parent / 'llm_phases'
MODEL5 = 'gemma3:12b'

PROMPT5 = """Du bewertest die Gesprächsführung der Person "B" anhand des folgenden Gesprächstranskripts eines Beratungsgesprächs.
"B" ist die zu bewertende (beratende) Person, "G" ist ihr Gesprächspartner bzw. ihre Gesprächspartnerin.
Bewerte NUR Person B. Grundlage: Gesprächsphasenmodell (Benien) und Gesprächstechniken (Verbalisieren, Paraphrasieren, Strukturieren). Phasen dürfen in der Reihenfolge variieren und mehrfach vorkommen; bewerte Vorhandensein und Qualität, nicht starre Abfolge. Alle Skalen: 1 = fehlt/sehr schlecht, 5 = vorbildlich.

- anfang_rahmen: Anfangsphase — Begrüßung, ggf. Smalltalk, zeitlicher Rahmen (Zeitangabe) und inhaltlicher Rahmen (Anliegen/Agenda) werden geklärt. 5 = Kontakt hergestellt UND Zeit- und Inhaltsrahmen explizit geklärt.
- phasen_vollstaendig: Sind die der Situation angemessenen Phasen erkennbar (Anfang, Information, Argumentation/Lösungssuche, Beschluss, Abschluss)? 5 = alle klar erkennbar und sinnvoll gefüllt; 1 = Gespräch bleibt in einer Phase stecken oder Phasen fehlen.
- beschluss_konkret: Beschlussphase — werden Vorschläge in konkrete Handlungsschritte übersetzt, ggf. (SMARTe) Ziele oder Vereinbarungen? 5 = konkrete, gemeinsam getragene nächste Schritte; 1 = nichts vereinbart.
- abschluss_qualitaet: Abschlussphase — wertschätzende Reflexion, ggf. Terminvereinbarung, Verabschiedung. 5 = würdigender, runder Abschluss; 1 = abruptes Ende.
- verbalisieren: B fokussiert wahrgenommene Gefühle von G — kurze Aufmerksamkeitsreaktionen, direktes Benennen eines Gefühls, klärende Fragen bei vagen Andeutungen, weiterführende emotionale offene Fragen. 5 = mehrfach, treffend, vertiefend.
- paraphrasieren: B gibt die sachliche Kernaussage von G in eigenen Worten wieder (ggf. verknappt). 5 = mehrfach und treffend an wichtigen Stellen.
- strukturieren: B wechselt auf die metakommunikative Ebene, um Ablauf zu steuern und transparent zu machen — Agenda, Überleitungen, Zusammenfassen des Gesagten. 5 = durchgängig klare, transparente Steuerung.

Sei streng und differenziert: 3 bedeutet durchschnittlich, 5 nur ohne erkennbare Schwächen. Nutze die gesamte Skala.

Transkript:
{text}

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt, keine Erklärung:
{{"anfang_rahmen": <1-5>, "phasen_vollstaendig": <1-5>, "beschluss_konkret": <1-5>, "abschluss_qualitaet": <1-5>, "verbalisieren": <1-5>, "paraphrasieren": <1-5>, "strukturieren": <1-5>}}"""


def _transcript_text(ah, db, ses, rol):
    try:
        d = ah.load(dataset=db, session=ses, scheme='transcript', role=rol, annotator='annotator_main').data
    except Exception:
        return ''
    if d is None or len(d) == 0:
        return ''
    return ' '.join(str(x) for x in pd.DataFrame(d)['name'])


def _dialog_text(ah, db, ses, assessed, foil):
    """Merged two-speaker transcript, time-ordered, dataset-neutral labels B/G."""
    parts = []
    for rol, tag in [(assessed, 'B'), (foil, 'G')]:
        try:
            d = ah.load(dataset=db, session=ses, scheme='transcript', role=rol, annotator='annotator_main').data
        except Exception:
            continue
        if d is None or len(d) == 0:
            continue
        df = pd.DataFrame(d)
        parts.append(pd.DataFrame({'from': df['from'], 'text': tag + ': ' + df['name'].astype(str).str.strip()}))
    if not parts:
        return ''
    return '\n'.join(pd.concat(parts).sort_values('from')['text'])


def _parse(resp, constructs=CONSTRUCTS):
    m = re.search(r'\{[^{}]*\}', resp, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return {c: float(obj[c]) for c in constructs if c in obj}
    except Exception:
        return None


def query(text, role):
    prompt = PROMPT.format(role=role, text=text[:12000])
    r = ollama.generate(model=MODEL, prompt=prompt,
                        options={'temperature': 1e-6, 'num_predict': 80, 'top_k': 1})
    return _parse(r['response'])


def query_dialog(text, seed=0, model=None, think=False, prompt=PROMPT2, constructs=CONSTRUCTS):
    p = prompt.format(text=text[:16000])   # two speakers -> longer budget
    return _parse(_generate(model or (MODEL3 if think else MODEL2), p, seed=seed, think=think),
                  constructs=constructs)


def _dialog_text_roleless(ah, db, ses, assessed, foil):
    """Merged two-speaker transcript, time-ordered, WITHOUT speaker labels — the
    attribution-noise ablation (phase 10d): the judge must identify the counselor."""
    parts = []
    for rol in (assessed, foil):
        try:
            d = ah.load(dataset=db, session=ses, scheme='transcript', role=rol, annotator='annotator_main').data
        except Exception:
            continue
        if d is None or len(d) == 0:
            continue
        df = pd.DataFrame(d)
        parts.append(pd.DataFrame({'from': df['from'], 'text': df['name'].astype(str).str.strip()}))
    if not parts:
        return ''
    return '\n'.join(pd.concat(parts).sort_values('from')['text'])


def run_dialog(datasets=None, think=False, out_root=None, model=None,
               prompt=PROMPT2, constructs=CONSTRUCTS, tag='dialog',
               seed_base=0, text_builder=None):
    load_dotenv('.env')
    dbconf = dict(db_host=os.environ['dbHost'], db_port=int(os.environ['dbPort']),
                  db_user=os.environ['dbUser'], db_password=os.environ['dbPassword'])
    ah = mh.AnnotationHandler(**dbconf)
    out_root = out_root or ((OUT3 if think else OUT2))
    model = model or (MODEL3 if think else MODEL2)
    for dataset in (datasets or ds_cfg.READY):
        cfg = ds_cfg.DATASETS[dataset]
        (out_root / dataset).mkdir(parents=True, exist_ok=True)
        sess = ds_cfg.sessions(dataset)
        done = fail = 0
        for ses in sess:
            outf = out_root / dataset / f'{ses}.json'
            if outf.exists():
                done += 1
                continue
            text = (text_builder or _dialog_text)(ah, cfg['db'], ses, cfg['assessed_role'], cfg['foil_role'])
            if len(text) < 20:
                continue
            scores = None
            for attempt in range(3):                       # resample with a new seed, not identical retries
                scores = query_dialog(text, seed=seed_base + attempt, model=model, think=think,
                                      prompt=prompt, constructs=constructs)
                if scores and len(scores) == len(constructs):
                    break
            if scores and len(scores) == len(constructs):
                outf.write_text(json.dumps(scores))
                done += 1
            else:
                fail += 1
            if (done + fail) % 10 == 0:
                print(f'  {dataset} ({tag}{"+think" if think else ""}): {done} ok, {fail} fail / {len(sess)}')
        print(f'{dataset} ({tag}{"+think" if think else ""}/{model}): {done} ok, {fail} fail -> {out_root.name}/{dataset}/')


def run(datasets=None):
    load_dotenv('.env')
    dbconf = dict(db_host=os.environ['dbHost'], db_port=int(os.environ['dbPort']),
                  db_user=os.environ['dbUser'], db_password=os.environ['dbPassword'])
    ah = mh.AnnotationHandler(**dbconf)
    for dataset in (datasets or ds_cfg.READY):
        cfg = ds_cfg.DATASETS[dataset]
        rol = cfg['assessed_role']
        (OUT / dataset).mkdir(parents=True, exist_ok=True)
        sess = ds_cfg.sessions(dataset)
        done = fail = 0
        for ses in sess:
            outf = OUT / dataset / f'{ses}.json'
            if outf.exists():
                done += 1
                continue
            text = _transcript_text(ah, cfg['db'], ses, rol)
            if len(text) < 20:
                continue
            scores = None
            for _ in range(3):
                scores = query(text, rol)
                if scores and len(scores) == len(CONSTRUCTS):
                    break
            if scores and len(scores) == len(CONSTRUCTS):
                outf.write_text(json.dumps(scores))
                done += 1
            else:
                fail += 1
            if (done + fail) % 10 == 0:
                print(f'  {dataset}: {done} ok, {fail} fail / {len(sess)}')
        print(f'{dataset}: {done} ok, {fail} fail -> ml/llm/{dataset}/')


def run_theory(datasets=None):
    """v4: theory-driven constructs (BGR + OSCE anchors), gemma3 nothink -> ml/llm_theory/."""
    run_dialog(datasets=datasets, out_root=OUT4, model=MODEL4,
               prompt=PROMPT4, constructs=CONSTRUCTS4, tag='theory')


def run_phases(datasets=None):
    """v5: phase/technique constructs (phase/technique guideline anchors) -> ml/llm_phases/."""
    run_dialog(datasets=datasets, out_root=OUT5, model=MODEL5,
               prompt=PROMPT5, constructs=CONSTRUCTS5, tag='phases')


# v6: PROMPT4 + anti-band-compression instruction. Screening (30 sessions) and full
# run (phase 9) beat v4 (stack 0.446 vs 0.439) — but the variant was SELECTED via
# target correlations on a 30-session dev subset, so the paper keeps v4 as headline.
PROMPT6 = PROMPT4.replace(
    'Sei streng und differenziert: 3 bedeutet durchschnittlich, 5 nur ohne erkennbare Schwächen. Nutze die gesamte Skala.',
    'Sei streng und differenziert: 3 bedeutet durchschnittlich, 5 nur ohne erkennbare Schwächen. '
    'WICHTIG: Nutze die GESAMTE Skala von 1 bis 5. Vermeide es, aus Vorsicht alles mit 3 oder 4 zu '
    'bewerten; vergib auch 1, 2 und 5, wenn das Gespräch es rechtfertigt, und differenziere zwischen den Dimensionen.')
OUT6 = Path(__file__).resolve().parent / 'llm_theory_v2'


def run_theory_v2(datasets=None):
    """v6: anti-band theory judge -> ml/llm_theory_v2/ (prefix llm6_ in experiment)."""
    run_dialog(datasets=datasets, out_root=OUT6, model=MODEL4,
               prompt=PROMPT6, constructs=CONSTRUCTS4, tag='theory-v2')


# v7/v8: theory-v2 prompt on the round-3 bake-off passers (screened label-free on
# 30 sessions; run remotely via OLLAMA_HOST tunnel).
OUT7 = Path(__file__).resolve().parent / 'llm_theory_qwen36'
OUT8 = Path(__file__).resolve().parent / 'llm_theory_phi4'


def run_theory_qwen36(datasets=None):
    """v7: anti-band theory judge on qwen3.6-27B -> ml/llm_theory_qwen36/ (llm7_)."""
    run_dialog(datasets=datasets, out_root=OUT7, model='qwen3.6:27b',
               prompt=PROMPT6, constructs=CONSTRUCTS4, tag='theory-qwen36')


def run_theory_phi4(datasets=None):
    """v8: anti-band theory judge on phi4-14B -> ml/llm_theory_phi4/ (llm8_)."""
    run_dialog(datasets=datasets, out_root=OUT8, model='phi4:14b',
               prompt=PROMPT6, constructs=CONSTRUCTS4, tag='theory-phi4')


# Roleless ablation (phase 10d): the transcript is merged WITHOUT B:/G: labels; the
# prompt keeps the identical construct anchors but tells the judge the labels are
# absent and B/G refer to the (self-identified) counselor and their interlocutor.
# Controlled test of speaker-attribution noise: if labels matter, removing them
# should hurt the mic-separated corpora more than the diarised one (vaccines).
PROMPT6_ROLELESS = PROMPT6.replace(
    'Du bewertest die Gesprächsführung der Person "B" anhand des folgenden Gesprächstranskripts.\n'
    '"B" ist die zu bewertende Person, "G" ist ihr Gesprächspartner bzw. ihre Gesprächspartnerin.\n'
    'Bewerte NUR Person B',
    'Du bewertest die Gesprächsführung der beratenden Person anhand des folgenden '
    'Gesprächstranskripts. Das Transkript enthält BEIDE Gesprächspartner in zeitlicher '
    'Reihenfolge, aber OHNE Sprecherkennzeichnung. Identifiziere selbst, welche Äußerungen '
    'von der beratenden Person stammen (sie führt das Gespräch). Im Folgenden bezeichnet '
    '"B" die beratende Person und "G" ihr Gegenüber. Bewerte NUR Person B')

OUT9 = Path(__file__).resolve().parent / 'llm_theory_granite30'


def run_theory_granite30(datasets=None):
    """v9: anti-band theory judge on granite4.1-30B -> ml/llm_theory_granite30/ (llm9_)."""
    run_dialog(datasets=datasets, out_root=OUT9, model='granite4.1:30b',
               prompt=PROMPT6, constructs=CONSTRUCTS4, tag='theory-granite30')


def run_theory_granite30_roleless(datasets=None):
    """Attribution-noise ablation: granite-30B judge on label-free merged transcripts."""
    run_dialog(datasets=datasets,
               out_root=Path(__file__).resolve().parent / 'llm_theory_granite30_roleless',
               model='granite4.1:30b', prompt=PROMPT6_ROLELESS, constructs=CONSTRUCTS4,
               tag='theory-granite30-roleless', text_builder=_dialog_text_roleless)


def run_theory_qwen36_seed(seed_base, datasets=None):
    """Seed-shifted qwen3.6 runs for seed-averaging (llm7 de-noising; seeds 10/20)."""
    run_dialog(datasets=datasets,
               out_root=Path(__file__).resolve().parent / f'llm_theory_qwen36_s{seed_base}',
               model='qwen3.6:27b', prompt=PROMPT6, constructs=CONSTRUCTS4,
               tag=f'theory-qwen36-s{seed_base}', seed_base=seed_base)


if __name__ == '__main__':
    run()
    run_dialog()
    run_dialog(think=True)
    run_theory()
    run_phases()
    run_theory_v2()
