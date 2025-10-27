from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='connection',
      version='0.0.1',
      description='Implementation of the simulations in our ICASSP2026 paper',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
      ],
      keywords='multiset canonical correlation analysis, independent vector analysis',
      url='https://github.com/SSTGroup/connection_mcca_genvar_iva_g',
      author='Isabell Lehmann',
      author_email='isabell.lehmann@sst.upb.de',
      license='LICENSE',
      packages=['connection'],
      python_requires='>=3.11',
      install_requires=[
          'numpy',
          'scipy',
          'pathlib',
          'pytest',
          'matplotlib',
          'independent_vector_analysis',
          'multiset_canonical_correlation_analysis',
          'matplot2tikz'
      ],
      include_package_data=True,  # to include non .py-files listed in MANIFEST.in
      zip_safe=False)
