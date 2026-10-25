"""AWS Lambda entry point.

Mangum adapts the same FastAPI application that `uvicorn` serves locally, so
there is one app and one code path rather than a separate Lambda variant that
drifts from the thing that was tested.

Lambda invokes this through its runtime interface client, not through the
image's CMD — which is exactly why index resolution lives in rag/config.py
rather than in a shell entrypoint (docs/decisions/006).
"""

from mangum import Mangum

from api.main import app
from rag import telemetry, tracing

_asgi = Mangum(app, lifespan="off")


def handler(event, context):
    """Route a warming ping away from the ASGI app, and serve everything else.

    A cold /ask takes ~20s — mostly importing torch and loading the embedding
    model (it was ~47s before docs/decisions/017) — against a 30s ceiling on HTTP
    API integrations that AWS does not allow to be raised. That leaves little
    margin, so a schedule keeps a warm container with the model already loaded
    (infra/main.tf, aws_cloudwatch_event_rule.warm).

    The ping must touch the retriever, not just the process: /health returns in
    17.8s cold because it never loads the model, so warming on /health would warm
    the wrong thing. It deliberately stops short of Bedrock — generation is not
    what is slow, and pinging it every five minutes would spend tokens to measure
    nothing.
    """
    if isinstance(event, dict) and event.get("warm"):
        from rag.generate import retrieve_chunks

        retrieve_chunks("warm", k=1)
        # Build the tracing client here too: fetching its keys and importing
        # langfuse (~0.8s) then happen on the schedule's clock, not a visitor's.
        tracing.client()
        return {"warmed": True}

    try:
        return _asgi(event, context)
    finally:
        # After the response is built and before Lambda freezes the process, which
        # is the only window in which a queued trace can still leave. Bounded, and
        # logged with whether it finished, because its cost lands on the visitor.
        sent = tracing.flush()
        if sent is not None:
            telemetry.logger.info(
                "trace_flush",
                extra={"fields": {"request_id": getattr(context, "aws_request_id", None), **sent}},
            )
