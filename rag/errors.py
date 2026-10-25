"""Failure types for the generation path, split by whether retrying helps.

Two categories, because that is the only distinction a caller can act on. An API
turns one into 503 with Retry-After and the other into 502; the agent's bounded
retry should attempt the first and give up immediately on the second. Retrying an
AccessDeniedException spends the user's remaining seconds to fail identically — and
on this deployment those seconds are scarce, since an HTTP API integration is cut
off at 30s (docs/decisions/009).

The AWS vocabulary stops here. Nothing above rag/ should import botocore to find
out whether a request is worth repeating.
"""


class GenerationError(RuntimeError):
    """Generation could not produce an answer."""


class TransientGenerationError(GenerationError):
    """Throttling, timeout, or a service-side fault. Retrying may succeed."""


class PermanentGenerationError(GenerationError):
    """Bad request, missing permission, model not enabled. Retrying will not help."""


# Bedrock error codes that mean "try again". Everything else is treated as
# permanent: guessing optimistically about an unknown code costs the caller a
# timeout budget it may not have.
RETRYABLE_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelTimeoutException",
        "ModelNotReadyException",
    }
)
