from .cartpoleENV_temp import *


class CartPoleEnvPoleM08l07(CartPoleEnv_template):
    def __init__(self):
        super().__init__( masscart =1.0, masspole=0.8, polelength=0.7)