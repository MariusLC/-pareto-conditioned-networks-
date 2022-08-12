from pycolab import human_ui
import six
import collections
import copy
import curses
import datetime
import textwrap
from pycolab import cropping
from pycolab.protocols import logging as plab_logging

# import Setuptools
# export TERM=xterm-256color


class CursesUi_Marius(human_ui.CursesUi):
    def __init__(self, policy, 
                keys_to_actions, delay=None, repainter=None, colour_fg=None, colour_bg=None, croppers=None):
        # making ing the basis curseUi
        super().__init__(keys_to_actions, delay, repainter, colour_fg, colour_bg, croppers)

        # # adding discrim to arguments for testing
        # self.discrim_to_test = discrim

        # # file to write on
        # self.filename = filename

        # agent's policy that will choose the actions
        self.policy = policy

        self.last_obs = None

    def fct(self, screen):
      keycode = screen.getch()
      if keycode == curses.KEY_PPAGE:    # Page Up? Show the game console.
        paint_console = True
      elif keycode == curses.KEY_NPAGE:  # Page Down? Hide the game console.
        paint_console = False
      elif keycode in self._keycodes_to_actions:
        action = self._keycodes_to_actions[keycode]
        if action == "eval_discrim" :
          observation = self.policy.eval_discrim(self._keycodes_to_actions, self.crop_and_repaint, env)
          # self.crop_and_repaint(observation)
        else:
          if action == None:
            observation, reward = self.policy.act()          
          else:
            observation, reward = self.policy.act(action)
          self.last_obs = observation

          observations = self.crop_and_repaint(observation)
          if self._total_return is None:
            self._total_return = reward
          elif reward is not None:
            self._total_return += reward

    def crop_and_repaint(self, observation):
          # Helper for game display: applies all croppers to the observation, then
          # repaints the cropped subwindows. Since the same repainter is used for
          # all subwindows, and since repainters "own" what they return and are
          # allowed to overwrite it, we copy repainted observations when we have
          # multiple subwindows.
          observations = [cropper.crop(observation) for cropper in self._croppers]
          if self._repainter:
            if len(observations) == 1:
              return [self._repainter(observations[0])]
            else:
              return [copy.deepcopy(self._repainter(obs)) for obs in observations]
          else:
            return observations


    def _init_curses_and_play(self, screen):
        """Set up an already-running curses; do interaction loop.
        This method is intended to be passed as an argument to `curses.wrapper`,
        so its only argument is the main, full-screen curses window.
        Args:
          screen: the main, full-screen curses window.
        Raises:
          ValueError: if any key in the `keys_to_actions` dict supplied to the
              constructor has already been reserved for use by `CursesUi`.
        """
        # See whether the user is using any reserved keys. This check ought to be in
        # the constructor, but it can't run until curses is actually initialised, so
        # it's here instead.
        for key, action in six.iteritems(self._keycodes_to_actions):
          if key in (curses.KEY_PPAGE, curses.KEY_NPAGE):
            raise ValueError(
                'the keys_to_actions argument to the CursesUi constructor binds '
                'action {} to the {} key, which is reserved for CursesUi. Please '
                'choose a different key for this action.'.format(
                    repr(action), repr(curses.keyname(key))))

        # If the terminal supports colour, program the colours into curses as
        # "colour pairs". Update our dict mapping characters to colour pairs.
        self._init_colour()
        curses.curs_set(0)  # We don't need to see the cursor.
        if self._delay is None:
          screen.timeout(-1)  # Blocking reads
        else:
          screen.timeout(self._delay)  # Nonblocking (if 0) or timing-out reads

        # Create the curses window for the log display
        rows, cols = screen.getmaxyx()
        console = curses.newwin(rows // 2, cols, rows - (rows // 2), 0)

        # By default, the log display window is hidden
        paint_console = False

        # Kick off the game---get first observation, crop and repaint as needed,
        # initialise our total return, and display the first frame.
        observation, reward, _ = self._game.its_showtime()
        observations = self.crop_and_repaint(observation)
        self._total_return = reward
        self._display(
            screen, observations, self._total_return, elapsed=datetime.timedelta())

        # Oh boy, play the game!

        # initialize state in policy
        self.policy.initial_state(observation)
        #####################################


        while not self._game.game_over:
          # Wait (or not, depending) for user input, and convert it to an action.
          # Unrecognised keycodes cause the game display to repaint (updating the
          # elapsed time clock and potentially showing/hiding/updating the log
          # message display) but don't trigger a call to the game engine's play()
          # method. Note that the timeout "keycode" -1 is treated the same as any
          # other keycode here.

          #####
          # FUNCTION TO EXECUTE
          self.fct(screen)
          ##########

          # Update the game display, regardless of whether we've called the game's
          # play() method.
          elapsed = datetime.datetime.now() - self._start_time
          self._display(screen, observations, self._total_return, elapsed)

          # Update game console message buffer with new messages from the game.
          self._update_game_console(
              plab_logging.consume(self._game.the_plot), console, paint_console)

          # Show the screen to the user.
          curses.doupdate()