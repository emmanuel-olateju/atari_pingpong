import numpy as np
from itertools import product

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def aqn_set_state(environment,boxes):
  __xb__, __yb__, __yp__, __bxv__, __byv__ = environment.observe()

  # set __yp__ (y position of paddle)
  __yp__ = __yp__ // (environment.HEIGHT // boxes)

  # set __xb__ (x position of ball)
  __xb__ = __xb__ // (environment.WIDTH // boxes)

  # set __yb__ (y position of the ball)
  __yb__ = __yb__ // (environment.HEIGHT // boxes)

  # set __bxv__ (x velocity of ball)
  if environment.BALL_SPEED_X>0:
      __bxv__ = 1
  else:
      __bxv__ = 0

  # set __byv__ (y velocity of ball)
  if environment.BALL_SPEED_Y>0:
      __byv__ = 1
  else:
      __byv__ = 0

  __xb__ = int(__xb__)
  __yb__ = int(__yb__)
  __yp__ = int(__yp__)
  __bxv__ = int(__bxv__)
  __byv__ = int(__byv__)

  return (__xb__, __yb__, __yp__, __bxv__, __byv__)