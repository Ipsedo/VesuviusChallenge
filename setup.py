# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="vesuvius_challenge",
    version="1.0.0",
    author="Samuel Berrien",
    packages=find_packages(
        include=["vesuvius_challenge", "vesuvius_challenge.*"]
    ),
)
