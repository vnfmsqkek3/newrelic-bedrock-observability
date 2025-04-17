import os
import atexit
import logging
from typing import Any, Callable, Dict, Optional

from newrelic_telemetry_sdk import (
    Event,
    EventBatch,
    EventClient,
    Harvester,
    Span,
    SpanBatch,
    SpanClient,
)

logger = logging.getLogger("newrelic_bedrock_observability")

# Bedrock 특화 이벤트 이름 정의
BEDROCK_EVENT_NAME = "BedrockEvent"
BEDROCK_SUMMARY_EVENT_NAME = "BedrockSummary"
BEDROCK_EMBEDDING_EVENT_NAME = "BedrockEmbedding"


class BedrockMonitoring:
    """AWS Bedrock 모니터링을 위한 클래스"""
    
    def __init__(
        self,
        use_logger: Optional[bool] = None,
    ):
        self.use_logger = use_logger if use_logger else False
        self.headers_by_id: dict = {}
        self.initialized = False

    def _set_license_key(
        self,
        license_key: Optional[str] = None,
    ):
        self.license_key = (
            license_key
            or os.getenv("NEW_RELIC_LICENSE_KEY")
            or os.getenv("NEW_RELIC_INSERT_KEY")
        )  # type: ignore

        if (
            not isinstance(self.license_key, str) and self.license_key is not None
        ) or self.license_key is None:
            raise TypeError("license_key instance type must be str and not None")

    def _set_client_host(
        self,
        event_client_host: Optional[str] = None,
    ):
        if not isinstance(event_client_host, str) and event_client_host is not None:
            raise TypeError("event_client_host instance type must be str or None")

        self.event_client_host = event_client_host or os.getenv(
            "EVENT_CLIENT_HOST", EventClient.HOST
        )

    def _set_metadata(
        self,
        metadata: Dict[str, Any] = {},
    ):
        self.metadata = metadata

        if not isinstance(metadata, Dict) and metadata is not None:
            raise TypeError("metadata instance type must be Dict[str, Any]")

    def _log(self, msg: str):
        if self.use_logger:
            logger.info(msg)
        else:
            print(msg)

    def start(
        self,
        application_name: str,
        license_key: Optional[str] = None,
        metadata: Dict[str, Any] = {},
        event_client_host: Optional[str] = None,
        parent_span_id_callback: Optional[Callable] = None,
        metadata_callback: Optional[Callable] = None,
    ):
        """모니터링 시작"""
        if not self.initialized:
            self.application_name = application_name
            self._set_license_key(license_key)
            self._set_metadata(metadata)
            self._set_client_host(event_client_host)
            self.parent_span_id_callback = parent_span_id_callback
            self.metadata_callback = metadata_callback
            self._start()
            self.initialized = True

    # initialize event thread
    def _start(self):
        """이벤트 클라이언트 및 스팬 클라이언트 초기화"""
        self.event_client = EventClient(
            self.license_key,
            host=self.event_client_host,
        )
        self.event_batch = EventBatch()

        # Background thread that flushes the batch
        self.event_harvester = Harvester(self.event_client, self.event_batch)

        # This starts the thread
        self.event_harvester.start()

        # When the process exits, run the harvester.stop() method before terminating the process
        atexit.register(self.event_harvester.stop)

        self.span_client = SpanClient(
            self.license_key,
            host=self.event_client_host,
        )

        self.span_batch = SpanBatch()

        # Background thread that flushes the batch
        self.span_harvester = Harvester(self.span_client, self.span_batch)
        self.span_harvester.start()

        atexit.register(self.span_harvester.stop)

    def record_event(
        self,
        event_dict: dict,
        table: str = BEDROCK_EVENT_NAME,
    ):
        """이벤트 기록"""
        event_dict["applicationName"] = self.application_name
        event_dict["provider"] = "aws_bedrock"
        event_dict.update(self.metadata)
        event = Event(table, event_dict)
        if self.metadata_callback:
            try:
                metadata = self.metadata_callback(event)
                if metadata:
                    event.update(metadata)
            except Exception as ex:
                logger.warning(f"Failed to run metadata callback: {ex}")
        self.event_batch.record(event)

    def record_span(self, span: Span):
        """스팬 기록"""
        span["attributes"]["applicationName"] = self.application_name
        span["attributes"]["instrumentation.provider"] = "nr_bedrock_observability_sdk"
        span["attributes"]["provider"] = "aws_bedrock"
        span.update(self.metadata)
        self.span_batch.record(span)

    def create_span(
        self,
        name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        guid: Optional[str] = None,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        start_time_ms: Optional[int] = None,
        duration_ms: Optional[int] = None,
    ):
        """스팬 생성"""
        if parent_id is None and self.parent_span_id_callback:
            parent_id = self.parent_span_id_callback()

        span = Span(
            name,
            tags,
            guid,
            trace_id,
            parent_id,
            start_time_ms,
            duration_ms,
        )
        return span


bedrock_monitor = BedrockMonitoring() 