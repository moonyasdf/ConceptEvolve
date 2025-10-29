import pytest
from pydantic import BaseModel
from types import SimpleNamespace

from src import llm_utils


class StructuredSchema(BaseModel):
    value: str


class FakeModel:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, contents, system_instruction, generation_config, request_options):
        self.calls.append(
            {
                "contents": contents,
                "system_instruction": system_instruction,
                "generation_config": generation_config,
                "request_options": request_options,
            }
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self):
        self.embed_calls = []
        self.embed_responses = {}
        self.models = {}
        self.requested_models = []

    @staticmethod
    def _normalize_content(content):
        if isinstance(content, list):
            return tuple(content)
        return content

    def set_embed_response(self, model, content, embedding):
        key = (model, self._normalize_content(content))
        self.embed_responses[key] = {"embedding": embedding}

    def embed_content(self, model, content):
        key = (model, self._normalize_content(content))
        self.embed_calls.append((model, content))
        return self.embed_responses[key]

    def set_model(self, name, model):
        self.models[name] = model

    def get_model(self, name):
        self.requested_models.append(name)
        return self.models[name]


class MemoryCache:
    def __init__(self):
        self.data = {}
        self.get_calls = []
        self.set_calls = []

    @staticmethod
    def _key(prompt, system_message, model, temperature):
        return (prompt, system_message, model, temperature)

    def get(self, prompt, system_message, model, temperature):
        key = self._key(prompt, system_message, model, temperature)
        self.get_calls.append(key)
        return self.data.get(key)

    def set(self, prompt, system_message, model, temperature, response):
        key = self._key(prompt, system_message, model, temperature)
        self.set_calls.append((key, response))
        self.data[key] = response


def make_structured_response(data):
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(data=data)])
            )
        ]
    )


@pytest.fixture(autouse=True)
def fake_types(monkeypatch):
    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.thinking_config = None

    class FakeThinkingConfig:
        def __init__(self, thinking_budget):
            self.thinking_budget = thinking_budget

    monkeypatch.setattr(
        llm_utils,
        "types",
        SimpleNamespace(
            GenerationConfig=FakeGenerationConfig,
            ThinkingConfig=FakeThinkingConfig,
        ),
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_utils.time, "sleep", lambda *_: None)


@pytest.fixture(autouse=True)
def memory_cache(monkeypatch):
    cache = MemoryCache()
    monkeypatch.setattr(llm_utils, "_cache", cache)
    return cache


@pytest.fixture
def mock_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(llm_utils.app_config, "get_client", lambda: client)
    return client


def test_embedding_client_returns_single_embedding(mock_client):
    mock_client.set_embed_response(
        "models/text-embedding-004", "hello", [0.1, 0.2, 0.3]
    )
    client = llm_utils.EmbeddingClient(model_name="text-embedding-004")

    result = client.get_embedding("hello")

    assert result == [0.1, 0.2, 0.3]
    assert mock_client.embed_calls == [("models/text-embedding-004", "hello")]


def test_embedding_client_handles_batch_requests(mock_client):
    embeddings = {
        "first": [0.11, 0.22],
        "second": [0.33, 0.44],
    }
    for text, embedding in embeddings.items():
        mock_client.set_embed_response("models/text-embedding-004", text, embedding)

    client = llm_utils.EmbeddingClient(model_name="models/text-embedding-004")

    results = [client.get_embedding(text) for text in embeddings]

    assert results == list(embeddings.values())
    assert mock_client.embed_calls == [
        ("models/text-embedding-004", "first"),
        ("models/text-embedding-004", "second"),
    ]


def test_query_structured_happy_path(mock_client):
    response_obj = StructuredSchema(value="ok")
    model = FakeModel([make_structured_response(response_obj)])
    mock_client.set_model("models/gemini-pro", model)

    result = llm_utils.query_gemini_structured(
        prompt="prompt",
        system_message="system",
        response_schema=StructuredSchema,
        model_name="gemini-pro",
        temperature=0.4,
        use_cache=False,
    )

    assert isinstance(result, StructuredSchema)
    assert result.value == "ok"
    assert mock_client.requested_models == ["models/gemini-pro"]
    call = model.calls[0]
    assert call["request_options"] == {"response_schema": StructuredSchema}
    assert call["generation_config"].kwargs["temperature"] == 0.4


def test_query_structured_retries_on_transient_error(mock_client, monkeypatch):
    response_obj = StructuredSchema(value="recovered")
    model = FakeModel(
        [Exception("boom"), make_structured_response(response_obj)]
    )
    mock_client.set_model("models/gemini-pro", model)
    monkeypatch.setattr(
        llm_utils, "_handle_api_error", lambda: pytest.fail("unexpected call")
    )

    result = llm_utils.query_gemini_structured(
        prompt="retry",
        system_message="system",
        response_schema=StructuredSchema,
        model_name="gemini-pro",
        temperature=0.3,
        use_cache=False,
        max_retries=2,
    )

    assert result.value == "recovered"
    assert len(model.calls) == 2
    assert mock_client.requested_models == ["models/gemini-pro", "models/gemini-pro"]


def test_query_structured_handles_malformed_candidates(mock_client, monkeypatch):
    model = FakeModel([SimpleNamespace(candidates=[])])
    mock_client.set_model("models/gemini-pro", model)
    handle_calls = []

    def fake_handle():
        handle_calls.append(None)
        return False

    monkeypatch.setattr(llm_utils, "_handle_api_error", fake_handle)

    result = llm_utils.query_gemini_structured(
        prompt="broken",
        system_message="system",
        response_schema=StructuredSchema,
        model_name="gemini-pro",
        use_cache=False,
        max_retries=1,
    )

    assert result is None
    assert len(handle_calls) == 1
    assert len(model.calls) == 1


def test_query_structured_refreshes_cache_after_validation_error(
    mock_client, memory_cache
):
    prompt = "cached"
    system = "system"
    temperature = 0.2
    key = (prompt, system, "gemini-pro", temperature)
    memory_cache.data[key] = "not json"

    response_obj = StructuredSchema(value="fresh")
    model = FakeModel([make_structured_response(response_obj)])
    mock_client.set_model("models/gemini-pro", model)

    result = llm_utils.query_gemini_structured(
        prompt=prompt,
        system_message=system,
        response_schema=StructuredSchema,
        model_name="gemini-pro",
        temperature=temperature,
        use_cache=True,
    )

    assert result.value == "fresh"
    assert mock_client.requested_models == ["models/gemini-pro"]
    assert len(memory_cache.set_calls) == 1
    assert memory_cache.data[key] == response_obj.model_dump_json()


def test_query_structured_uses_cache_when_available(mock_client, memory_cache):
    prompt = "prompt"
    system = "system"
    temperature = 0.5
    schema_obj = StructuredSchema(value="cached")
    key = (prompt, system, "gemini-pro", temperature)
    memory_cache.data[key] = schema_obj.model_dump_json()

    result = llm_utils.query_gemini_structured(
        prompt=prompt,
        system_message=system,
        response_schema=StructuredSchema,
        model_name="gemini-pro",
        temperature=temperature,
        use_cache=True,
    )

    assert isinstance(result, StructuredSchema)
    assert result.value == "cached"
    assert mock_client.requested_models == []
    assert memory_cache.set_calls == []


def test_query_structured_applies_thinking_config_when_enabled(mock_client, monkeypatch):
    monkeypatch.setattr(llm_utils.app_config.model_config, "use_thinking", True)
    monkeypatch.setattr(llm_utils.app_config.model_config, "thinking_budget", 256)
    response_obj = StructuredSchema(value="think")
    model = FakeModel([make_structured_response(response_obj)])
    mock_client.set_model("models/gemini-2.5-pro", model)

    result = llm_utils.query_gemini_structured(
        prompt="prompt",
        system_message="system",
        response_schema=StructuredSchema,
        model_name="gemini-2.5-pro",
        temperature=0.1,
        use_cache=False,
    )

    assert result.value == "think"
    gen_config = model.calls[0]["generation_config"]
    assert gen_config.thinking_config is not None
    assert gen_config.thinking_config.thinking_budget == 256


@pytest.mark.parametrize(
    "model_name,use_thinking",
    [
        ("gemini-2.5-pro", False),
        ("gemini-2.0-flash", True),
    ],
)
def test_query_structured_skips_thinking_config_when_disabled_or_non_pro(
    mock_client, monkeypatch, model_name, use_thinking
):
    monkeypatch.setattr(llm_utils.app_config.model_config, "use_thinking", use_thinking)
    monkeypatch.setattr(llm_utils.app_config.model_config, "thinking_budget", 512)
    response_obj = StructuredSchema(value="plain")
    model = FakeModel([make_structured_response(response_obj)])
    mock_client.set_model(f"models/{model_name}", model)

    result = llm_utils.query_gemini_structured(
        prompt="prompt",
        system_message="system",
        response_schema=StructuredSchema,
        model_name=model_name,
        temperature=0.7,
        use_cache=False,
    )

    assert result.value == "plain"
    gen_config = model.calls[0]["generation_config"]
    assert gen_config.thinking_config is None
