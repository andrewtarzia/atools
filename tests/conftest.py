import os
import logging
import sys


logging.basicConfig(
    format='\n\n%(levelname)s:%(module)s:%(message)s',
    stream=sys.stdout
)
logging.getLogger('atools').setLevel(logging.DEBUG)


# Run tests in a directory so that that generated files are easy to
# delete.
output_dir = 'tests_output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
os.chdir(output_dir)


def pytest_addoption(parser):
    pass


def pytest_generate_tests(metafunc):
    # if 'xtb_path' in metafunc.fixturenames:
    #     xtb_path = metafunc.config.getoption('xtb_path')
    #     metafunc.parametrize('xtb_path', [xtb_path])

    pass

# @pytest.fixture(scope='session')
# def mae_path():
    # return join('..', 'data', 'molecule.mae')
