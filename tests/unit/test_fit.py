import httpx
import pytest

from colabinet.fit import estimate, hf_file_gb


def test_measured_l4_cases():

    q4 = estimate("L4", 16.8, 262144, "q4_0", 65536)
    assert q4.verdict == "tight" and 21 < q4.total_gb < 22.5
    f16 = estimate("L4", 16.8, 262144, "f16", 65536)
    assert f16.verdict == "spills" and "offload" in f16.message
    assert estimate("L4", 16.8, 32768, "f16", 65536).verdict == "fits"
    assert estimate("A100-80", 16.8, 262144, "f16", 65536).verdict == "fits"


def test_unknown_kv_and_cpu():
    assert estimate("L4", 16.8, 8192, "f16", None).verdict == "fits"
    assert "KV size unknown" in estimate("L4", 16.8, 8192, "f16", None).message
    assert estimate("L4", None, 8192, "f16").verdict == "unknown"
    assert estimate("CPU", 1.0, 2048, "f16").verdict == "cpu"


def test_weights_exceeding_vram_spill_even_with_unknown_kv():
    fit = estimate("A100", 111.33, 8192, "f16")
    assert fit.verdict == "spills" and "offload" in fit.message
    assert "KV size unknown" in fit.message


async def test_split_hf_size_counts_all_shards():
    sizes = {"Q4/model-00001-of-00003.gguf": "10000000",
             "Q4/model-00002-of-00003.gguf": "49000000000",
             "Q4/model-00003-of-00003.gguf": "12000000000"}

    def handler(request):
        path = request.url.path.split("/resolve/pinned/", 1)[1]
        return httpx.Response(302, headers={"x-linked-size": sizes[path]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        size = await hf_file_gb(client, "org/repo", "Q4/model-00001-of-00003.gguf", "pinned")
        assert size == pytest.approx(61.01)


async def test_split_hf_size_is_unknown_if_any_shard_is_missing():
    def handler(request):
        if request.url.path.endswith("00001-of-00002.gguf"):
            return httpx.Response(302, headers={"x-linked-size": "10000000"})
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await hf_file_gb(client, "org/repo", "model-00001-of-00002.gguf") is None


def test_vision_on_gpu_counts():
    off = estimate("L4", 16.8, 262144, "q4_0", 65536, mmproj_gb=0.93, mmproj_gpu=False)
    on = estimate("L4", 16.8, 262144, "q4_0", 65536, mmproj_gb=0.93, mmproj_gpu=True)
    assert on.total_gb - off.total_gb == pytest.approx(0.93)


async def test_hf_size_from_redirect_headers():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("model.gguf"):
            return httpx.Response(302, headers={"x-linked-size": "16810000000", "location": "https://cdn/x"})
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await hf_file_gb(client, "org/repo", "model.gguf") == pytest.approx(16.81)
        assert await hf_file_gb(client, "org/repo", "missing.gguf") is None


@pytest.mark.parametrize("status,headers", [(401, {}), (404, {}), (200, {}),
                                           (302, {"x-linked-size": "invalid"})])
async def test_unavailable_metadata_is_unknown(status, headers):
    def handler(request):
        return httpx.Response(status, headers=headers)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        size = await hf_file_gb(client, "org/repo", "model.gguf")
        assert size is None
        assert estimate("L4", size, 8192, "f16").verdict == "unknown"


async def test_transport_failure_is_unknown():
    def handler(request):
        raise httpx.ConnectError("private credential must not escape", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await hf_file_gb(client, "org/repo", "model.gguf") is None
