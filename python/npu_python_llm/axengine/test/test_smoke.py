from axengine import _C as m


def test_main():
    assert m.__version__ == "dev"
    assert m.add(1, 2) == 3
    assert m.subtract(1, 2) == -1
