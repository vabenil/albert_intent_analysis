from setuptools import setup


setup(
    name="intent_analysis",
    version="1.0.0",
    description="intent classification for virtual assistants using albert",
    url='https://github.com/vabenil/albert_intent_analysis',
    author='Victor Abenil Fernandez Rodriguez',
    author_email='vabtracker@gmail.com',
    license='MIT',
    packages=['intent_analysis'],
    install_requires=[
        'pandas',
        'scikit-learn',
        'transformers',
        'numpy',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  
    ],
)
