from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

from pycolab import plot
from pycolab import rendering
from pycolab import things
from pycolab import engine

import six


class Engine_Marius(engine.Engine):
  def __init__(self, rows, cols, the_plot, occlusion_in_layers=True):
    self._rows = rows
    self._cols = cols
    self._occlusion_in_layers = occlusion_in_layers

    # This game's Plot object
    self._the_plot = the_plot

    # True iff its_showtime() has been called and the game is underway.
    self._showtime = False
    # True iff the game has terminated. (It's still "showtime", though.)
    self._game_over = False

    # This game's Backdrop object.
    self._backdrop = None

    # This game's collection of Sprites and Drapes. The ordering of this dict
    # is the game's z-order, from back to front.
    self._sprites_and_drapes = collections.OrderedDict()

    # The collection of update groups. Before the its_showtime call, this is a
    # dict keyed by update group name, whose values are lists of Sprites and
    # Drapes in the update group. After the call, this becomes a dict-like list
    # of tuples that freezes the ordering implied by the update-group keys.
    self._update_groups = collections.defaultdict(list)

    # The current update group---used by add(). Will be set to None once the
    # game is underway.
    self._current_update_group = ''

    # This slot will hold the observation renderer once the game is underway.
    self._renderer = None

    # And this slot will hold the last observation rendered by the renderer.
    # It is not intended that this member be available to the user directly.
    # Code should not keep local references to this object or its members.
    self._board = None

  # def set_plot(self, plot):
  #   self.the_plot = plot

  # def _replace(self, the_plot):
  #   self.the_plot = the_plot