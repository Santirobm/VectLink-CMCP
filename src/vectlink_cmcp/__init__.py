"""VectLink-CMCP: Experimental Two-Level Multi-Agent Memory Integration."""

from .cmcp_adapter import (
    CMCPAdapter,
    CMCPAdapterError,
    EnvelopeMetadata,
    IncompatibleSchemaError,
    PayloadCorruptError,
    StateOp,
)
from .integration_runtime import (
    ConsumerSession,
    ConventionalSocketTransport,
    ProducerSession,
)
from .vectlink_adapter import (
    VectLinkArenaTransport,
    VectLinkRingTransport,
)

__all__ = [
    "CMCPAdapter",
    "CMCPAdapterError",
    "ConsumerSession",
    "ConventionalSocketTransport",
    "EnvelopeMetadata",
    "IncompatibleSchemaError",
    "PayloadCorruptError",
    "ProducerSession",
    "StateOp",
    "VectLinkArenaTransport",
    "VectLinkRingTransport",
]
