from setuptools import setup, find_packages

setup(
    name='reco_utils',
    version='0.1.0',
    description='Outils pour la détection de la qualité des données et l\'évaluation de modèles',
    author='Votre Nom',
    author_email='votre.email@example.com',
    url='https://github.com/votre-utilisateur/reco_utils',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0',
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
