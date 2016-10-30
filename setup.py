from pip.req import parse_requirements
from distutils.core import setup

install_reqs = parse_requirements("requirements.txt")
reqs = [str(ir.req) for ir in install_reqs]

setup(name='stacker',
      version='0.0.1',
      install_reqs=reqs,
      py_modules=['stacker'])
