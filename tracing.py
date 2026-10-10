"""Sentry agent tracing. Off unless SENTRY_DSN is set, so the project still runs with no account."""
import os

import sentry_sdk


def init():
    if os.environ.get("SENTRY_DSN"):
        sentry_sdk.init(dsn=os.environ["SENTRY_DSN"], traces_sample_rate=1.0, send_default_pii=False, release="fieldcast@1")
