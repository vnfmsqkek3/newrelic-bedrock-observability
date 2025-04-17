# newrelic-bedrock-observability

AWS Bedrock API 호출을 NewRelic에서 모니터링하기 위한 라이브러리입니다.

## 설치 방법

```bash
pip install newrelic-bedrock-observability
```

## 사용 방법

AWS Bedrock을 모니터링하기 위해서는 다음과 같은 코드를 추가하면 됩니다:

```python
import boto3
import json
from newrelic_bedrock_observability import monitor

# 모니터링 초기화
monitor.initialization(
    application_name="AWS Bedrock Observability Example"
)

# Bedrock 클라이언트 설정 (패치가 자동으로 적용됨)
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

# Claude 모델 호출 예제
response = bedrock_runtime.invoke_model(
    modelId="anthropic.claude-v2",
    body=json.dumps({
        "prompt": "\n\nHuman: Write a short poem about observability.\n\nAssistant:",
        "max_tokens_to_sample": 500,
        "temperature": 0.7,
    })
)
```

## 지원하는 기능

- InvokeModel API 모니터링
- InvokeModelWithResponseStream API 모니터링
- Embedding API 모니터링
- 응답 시간 측정
- 토큰 사용량 추정
- 오류 추적
- NewRelic 대시보드 통합

## 예제

`examples` 디렉토리에서 실행 예제를 확인할 수 있습니다.
