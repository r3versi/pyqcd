import pickle
import typing

import numpy as np


class Logger:
    def __init__(self, file: str, live_update: bool = True) -> None:
        self.vars = {}
        self.file = file
        self.live_update = live_update

    def add_variables(self, *args):
        for label in args:
            if label in self.vars:
                print(label, "already in", self.vars)
                raise Exception
            self.vars[label] = []

    def register(self, **kwargs) -> None:
        for label in kwargs:
            if label not in self.vars:
                print(label, "not in", self.vars)
                raise Exception

            self.vars[label].append(kwargs[label])

            if self.live_update:
                self.dump()

    def dump(self) -> None:
        with open(self.file, "wb") as f:
            pickle.dump(self.vars, f)
