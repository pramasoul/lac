import pytest



def pytest_addoption(parser):
    parser.addoption("--skip-compressed", action="store_true", help="Skip tests that use precomputed compressed data")

def pytest_collection_modifyitems(config, items):
    #print("\nChecking for skip-compressed option")
    if config.getoption("--skip-compressed"):
        #print("Skipping compressed data tests")
        skip_compressed = pytest.mark.skip(reason="Skipping tests with precomputed compressed data")
        for item in items:
            if "compressed_data" in item.keywords:
                #print(f"Skipping {item.name}")
                item.add_marker(skip_compressed)

