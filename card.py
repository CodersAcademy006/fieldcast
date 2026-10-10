"""Turn forecast.json into a one-page printable field card. Code picks the day and owns every number;
local Gemma (Ollama) only writes the prose around them, so it cannot invent a forecast.
Run: .venv/bin/python card.py  -> card.html"""
import html
import json
import re
import sys

import requests
import sentry_sdk

import tracing

MODEL = "gemma3:4b"
TRIES = 5
PROMPT = ("You write a pocket field card for a nature walk near Pune, India. Use ONLY the facts below. "
          "Do not add numbers, species or places that are not listed. No emojis, no markdown, no em dashes. "
          "Write 3 short sentences: why this day, what to look for and in what order, then one reminder to put the phone away.\n")


def likely_for(d):
    return ", ".join(f"{k} ({round(v * 100)}%)" for k, v in d["likely"].items())


def facts_for(d):
    return (f"Day: {d['weekday']} {d['date']}. Max temperature {d['tmax']} C. Rain {d['rain_mm']} mm.\n"
            f"Model expects about {d['butterfly_species']:.0f} butterfly species and {d['bird_species']:.0f} bird species logged today by local observers.\n"
            f"Most likely to be logged: {likely_for(d)}.")


def best(days):
    ok = [d for d in days if d["rain_mm"] < 5] or days
    return max(ok, key=lambda d: d["butterfly_species"] + d["bird_species"])


def gemma(prompt):
    with sentry_sdk.start_span(op="gen_ai.chat", name=f"chat {MODEL}") as span:
        for k, v in {"gen_ai.operation.name": "chat", "gen_ai.system": "ollama", "gen_ai.request.model": MODEL,
                     "gen_ai.request.temperature": 0.4, "gen_ai.request.messages": json.dumps([{"role": "user", "content": prompt}])}.items():
            span.set_data(k, v)
        r = requests.post("http://localhost:11434/api/generate", timeout=300,
                          json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.4, "num_predict": 220}})
        r.raise_for_status()
        j = r.json()
        text = j["response"].strip()
        span.set_data("gen_ai.usage.input_tokens", j["prompt_eval_count"])
        span.set_data("gen_ai.usage.output_tokens", j["eval_count"])
        span.set_data("gen_ai.usage.total_tokens", j["prompt_eval_count"] + j["eval_count"])
        span.set_data("gen_ai.response.text", json.dumps([text]))
        span.set_data("fieldcast.tokens_per_second", round(j["eval_count"] / (j["eval_duration"] / 1e9), 1))
        return text


def problems(text, facts):
    """Rules the card must satisfy: no number Gemma was not given, exactly three sentences."""
    with sentry_sdk.start_span(op="gen_ai.execute_tool", name="execute_tool check_card") as span:
        out = []
        bad = sorted(set(re.findall(r"\d+(?:\.\d+)?", text)) - set(re.findall(r"\d+(?:\.\d+)?", facts)))
        if bad:
            out.append(f"invented numbers {bad}")
        n = len(re.findall(r"[.!?](?:\s|$)", text))
        if n != 3:
            out.append(f"{n} sentences")
        span.set_data("gen_ai.tool.name", "check_card")
        span.set_data("gen_ai.tool.output", json.dumps(out))
        return out


def write(d, facts, likely):
    """Ask Gemma, reject a card that breaks the rules, retry, then fall back to a plain template."""
    prompt = PROMPT + facts
    for attempt in range(1, TRIES + 1):
        text = gemma(prompt)
        bad = problems(text, facts)
        if not bad:
            return text
        sentry_sdk.capture_message(f"Gemma card rejected: {bad} (attempt {attempt}/{TRIES})", level="warning")
    return (f"{d['weekday']} looks best this week: {d['tmax']} C and {d['rain_mm']} mm of rain. "
            f"Look for {likely.split(',')[0].split(' (')[0]} first. Put the phone away and go.")


def bench(n):
    """Run Gemma n times on this week's facts with no retry and count how often each rule is broken."""
    d = best(json.load(open("forecast.json"))["days"])
    facts = facts_for(d)
    broke = {"invented numbers": 0, "sentences": 0}
    for i in range(n):
        out = problems(gemma(PROMPT + facts), facts)
        for k in broke:
            broke[k] += any(p.startswith(k) or (k == "sentences" and "sentences" in p) for p in out)
        print(i + 1, out or "clean", flush=True)
    print(f"of {n} runs: {broke}")


def main():
    f = json.load(open("forecast.json"))
    d = best(f["days"])
    likely = likely_for(d)
    with sentry_sdk.start_transaction(name="invoke_agent fieldcast-card", op="gen_ai.invoke_agent"):
        text = write(d, facts_for(d), likely)
    rows = "".join(f"<tr><td>{x['weekday'][:3]} {x['date'][5:]}</td><td>{x['tmax']}</td><td>{x['rain_mm']}</td>"
                   f"<td>{x['butterfly_species']:.0f}</td><td>{x['bird_species']:.0f}</td></tr>" for x in f["days"])
    page = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Fieldcast</title>
<style>
:root{{--bg:#f5f2ea;--ink:#1d2a1f;--mute:#5d6b5f;--line:#d9d4c4}}
@media(prefers-color-scheme:dark){{:root{{--bg:#141a15;--ink:#e8eadf;--mute:#9aa89c;--line:#2b352d}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:100%/1.5 system-ui,sans-serif}}
main{{max-width:34rem;margin:0 auto;padding:2rem 1rem}}
h1{{font-size:clamp(2rem,8vw,3rem);line-height:1.05;letter-spacing:-.02em;margin:0 0 .25rem}}
.sub{{color:var(--mute);margin:0 0 1.5rem}}
p.card{{font-size:1.125rem;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:1rem 0}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.9rem}}
th,td{{text-align:left;padding:.35rem 0;border-bottom:1px solid var(--line)}}th{{color:var(--mute);font-weight:500}}
small{{color:var(--mute);display:block;margin-top:1.5rem}}
@media print{{:root{{--bg:#fff;--ink:#000;--mute:#444;--line:#bbb}}}}
</style><main>
<h1>{html.escape(d['weekday'])}, {html.escape(d['date'][5:])}</h1>
<p class=sub>Best day this week near Pune. Max {d['tmax']} C, rain {d['rain_mm']} mm.</p>
<p class=card>{html.escape(text)}</p>
<p><b>Likely to be seen:</b> {html.escape(likely)}</p>
<table><tr><th>Day<th>Max C<th>Rain mm<th>Butterfly spp<th>Bird spp</tr>{rows}</table>
<small>Forecast {f['generated']}. TabPFN trained on iNaturalist research-grade sightings and Open-Meteo weather; prose by Gemma running locally. Probabilities are chances a species is logged that day, not guarantees.</small>
</main>"""
    open("card.html", "w").write(page)
    print(text)
    return "card.html"


if __name__ == "__main__":
    tracing.init()
    bench(int(sys.argv[2])) if sys.argv[1:2] == ["--bench"] else main()
    sentry_sdk.flush()
