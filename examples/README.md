# AWS Bedrock NewRelic 모니터링 예제

이 예제는 AWS Bedrock API 호출을 NewRelic에서 모니터링하는 방법을 보여줍니다.

## 설치 방법

1. 필요한 패키지 설치:

```bash
pip install boto3 newrelic-telemetry-sdk newrelic-bedrock-observability
```

2. AWS 자격 증명 설정:

```bash
export AWS_ACCESS_KEY_ID=<your-access-key>
export AWS_SECRET_ACCESS_KEY=<your-secret-key>
export AWS_REGION=us-east-1
```

3. NewRelic 라이센스 키 설정:

```bash
export NEW_RELIC_LICENSE_KEY=<your-license-key>
```

## 사용 방법

AWS Bedrock을 모니터링하기 위해서는 다음과 같은 코드를 추가하면 됩니다:

```python
import boto3
from newrelic_bedrock_observability import monitor

# 모니터링 초기화
monitor.initialization(
    application_name="AWS Bedrock Observability Example",
    metadata={"environment": "development"},
)

# Bedrock 클라이언트 설정 (패치가 자동으로 적용됨)
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

# 일반적인 방식으로 Bedrock API 사용
response = bedrock_runtime.invoke_model(
    modelId="anthropic.claude-v2",
    body=json.dumps({
        "prompt": "\n\nHuman: Hello, Claude!\n\nAssistant:",
        "max_tokens_to_sample": 500,
    })
)
```

## 지원하는 기능

- InvokeModel 모니터링
- InvokeModelWithResponseStream 모니터링
- CreateEmbedding 모니터링
- 응답 시간 측정
- 토큰 사용량 추정
- 오류 추적
- NewRelic 대시보드 통합

## 예제 파일

- `example.py`: 기본 사용 예제

## EU 계정 사용자

EU 리전 계정을 사용하는 경우 다음과 같이 이벤트 클라이언트 호스트를 설정해주세요:

```python
monitor.initialization(
    application_name="AWS Bedrock Observability Example",
    event_client_host="insights-collector.eu01.nr-data.net"
)
```

또는 환경 변수를 설정할 수도 있습니다:

```bash
export EVENT_CLIENT_HOST="insights-collector.eu01.nr-data.net"
``` 