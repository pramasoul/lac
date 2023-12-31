import pytest
import psutil


def pytest_addoption(parser):
    parser.addoption("--skip-compressed", action="store_true", help="Skip tests that use precomputed compressed data")
    parser.addoption("--model", action="store", default="internal", help="What LLM to use")
    parser.addoption("--device", action="store", default="cpu", help="What device to use")
    parser.addoption("--threads", action="store", default=0, help="How many threads to use with cpu device")

def pytest_collection_modifyitems(config, items):
    #print("\nChecking for skip-compressed option")
    if config.getoption("--skip-compressed"):
        #print("Skipping compressed data tests")
        skip_compressed = pytest.mark.skip(reason="Skipping tests with precomputed compressed data")
        for item in items:
            if "compressed_data" in item.keywords:
                #print(f"Skipping {item.name}")
                item.add_marker(skip_compressed)

@pytest.fixture(scope="session")
def model_name(request):
    return request.config.getoption("--model")

@pytest.fixture(scope="session")
def device(request):
    return request.config.getoption("--device")

@pytest.fixture(scope="session")
def threads(request):
    v = int(request.config.getoption("--threads"))
    if v <= 0:
        v = psutil.cpu_count(logical=False)
    return v
