from setuptools import setup, find_packages

setup(name='uslime',
      version='0.1',
      description='Uncertainity-Sampling LIME for Model Explanation',
      url='https://github.com/Benyaminghn/uslime',
      author='Benyamin Ghahremani Nezhad',
      author_email='benyamin.ghahremani@aut.ac.ir',
      license='BSD',
      packages=find_packages(exclude=['js', 'node_modules', 'tests']),
      python_requires='>=3.8',
      install_requires=[
          'lime',
          'matplotlib',
          'numpy',
          'scipy',
          'tqdm >= 4.29.1',
          'scikit-learn>=0.18',
          'scikit-image>=0.12',
          'pyDOE2==1.3.0'
      ],
      extras_require={
          'dev': ['pytest', 'flake8'],
      },
      include_package_data=True,
      zip_safe=False)
