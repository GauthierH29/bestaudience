#appeler la fonction setup de setuptools
#récupérer le contenu de requirements.txt

from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

# à modifier
setup(name='bestaudience',
      version="0.0.1",
      description="Best audience - plateforme",
      license="MIT",
      author="GauthierH29",
      author_email="g.haicault@gmail.com",
      url="https://github.com/GauthierH29/bestaudience",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="test")
