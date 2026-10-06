"""Turn forecast.json into a one-page printable field card. Code picks the day and owns every number;
local Gemma (Ollama) only writes the prose around them, so it cannot invent a forecast.
Run: .venv/bin/python card.py  -> card.html"""
import html
import json

import requests

MODEL = "gemma3:4b"


def best(days):
    ok = [d for d in days if d["rain_mm"] < 5] or days
    return max(ok, key=lambda d: d["butterfly_species"] + d["bird_species"])


def gemma(prompt):
    r = requests.post("http://localhost:11434/api/generate", timeout=300,
                      json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.4, "num_predict": 220}})
    r.raise_for_status()
    return r.json()["response"].strip()


def main():
    f = json.load(open("forecast.json"))
    d = best(f["days"])
    likely = ", ".join(f"{k} ({round(v * 100)}%)" for k, v in d["likely"].items())
    prompt = (
        "You write a pocket field card for a nature walk near Pune, India. Use ONLY the facts below. "
        "Do not add numbers, species or places that are not listed. No emojis, no markdown, no em dashes. "
        "Write 3 short sentences: why this day, what to look for and in what order, then one reminder to put the phone away.\n"
        f"Day: {d['weekday']} {d['date']}. Max temperature {d['tmax']} C. Rain {d['rain_mm']} mm.\n"
        f"Model expects about {d['butterfly_species']:.0f} butterfly species and {d['bird_species']:.0f} bird species logged today by local observers.\n"
        f"Most likely to be logged: {likely}.")
    text = gemma(prompt)
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


if __name__ == "__main__":
    main()
