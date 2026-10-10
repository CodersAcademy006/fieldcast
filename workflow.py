"""Durable Fieldcast pipeline on Temporal: fetch -> (backtest) -> forecast -> card.
Each step is an activity with retries; the iNaturalist fetch heartbeats and resumes from its page checkpoint.
Run:  temporal server start-dev                 (terminal 1, UI at http://localhost:8233)
      .venv/bin/python workflow.py worker       (terminal 2)
      .venv/bin/python workflow.py run          (terminal 3; add --backtest to rerun the held-out year)"""
import asyncio
import sys
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    import pandas as pd
    import tracing

QUEUE = "fieldcast"
GROUPS = {"butterfly": 47224, "bird": 3}  # iNaturalist taxon ids: Papilionoidea, Aves


async def beat(fn, *args):
    """Run blocking code in a thread and heartbeat until it ends, so Temporal can tell a slow step from a dead one."""
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    while not task.done():
        activity.heartbeat()
        await asyncio.wait({task}, timeout=10)
    return task.result()


@activity.defn
async def fetch_obs(group: str) -> int:
    import fetch
    rows = await beat(fetch.observations, GROUPS[group], group)
    pd.DataFrame(rows, columns=["id", "date", "sci", "common", "group"]).to_parquet(fetch.OUT / f"obs_{group}.parquet", index=False)
    return len(rows)


@activity.defn
async def fetch_weather() -> int:
    import fetch
    w = await beat(fetch.weather)
    w.to_parquet(fetch.OUT / "weather.parquet", index=False)
    return len(w)


@activity.defn
async def merge_obs() -> int:
    import fetch
    obs = pd.concat([pd.read_parquet(fetch.OUT / f"obs_{g}.parquet") for g in GROUPS])
    obs["date"] = pd.to_datetime(obs["date"])
    obs.to_parquet(fetch.OUT / "obs.parquet", index=False)
    return len(obs)


@activity.defn
async def run_backtest() -> str:
    import model
    await beat(model.backtest)
    return "metrics.json"


@activity.defn
async def run_forecast() -> str:
    import model
    await beat(model.forecast)
    return "forecast.json"


@activity.defn
async def make_card() -> str:
    import card
    return await beat(card.main)


@workflow.defn
class Fieldcast:
    @workflow.run
    async def run(self, backtest: bool = False) -> str:
        step = dict(start_to_close_timeout=timedelta(hours=1), heartbeat_timeout=timedelta(minutes=2),
                    retry_policy=RetryPolicy(initial_interval=timedelta(seconds=5), backoff_coefficient=2, maximum_attempts=8))
        await asyncio.gather(*[workflow.execute_activity(fetch_obs, g, **step) for g in GROUPS],
                             workflow.execute_activity(fetch_weather, **step))
        await workflow.execute_activity(merge_obs, **step)
        if backtest:
            await workflow.execute_activity(run_backtest, **step)
        await workflow.execute_activity(run_forecast, **step)
        return await workflow.execute_activity(make_card, **step)


async def main():
    client = await Client.connect("localhost:7233")
    if sys.argv[1] == "worker":
        tracing.init()
        await Worker(client, task_queue=QUEUE, workflows=[Fieldcast],
                     activities=[fetch_obs, fetch_weather, merge_obs, run_backtest, run_forecast, make_card]).run()
    else:
        print(await client.execute_workflow(Fieldcast.run, "--backtest" in sys.argv, id="fieldcast", task_queue=QUEUE))


if __name__ == "__main__":
    asyncio.run(main())
