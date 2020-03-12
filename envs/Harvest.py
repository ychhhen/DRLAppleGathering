import random
import numpy as np
import envs.rendering as rendering
from pycolab import ascii_art
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

TWO_PLAYER_MAP = [
                 list('                    A'),
                 list('           @        C'),
                 list('          @@@        '),
                 list('         @@@         '),
                 list('          @          '),
                 list('                     '),
                 list('                    B'),
                 list('                    D')]

agent_colours = [(0, 0, 999),  # Blue
                 (999, 0, 0),  # Red
                 (999, 999, 0),  # Yellow
                 (0, 999, 999),  # Cyan
                 (500, 0, 999),  # Purple
                 (999, 500, 0),  # Orange
                 (999, 0, 999),  # Pink
                 (400, 200, 0)]  # Brown

agents = ""
agent_o_h = 0
agent_o_w = 0
h = 0
w = 0


def two_player_map():
    global h, w
    h = 8
    w = 21
    """Returns a two player map."""
    game_field = np.array(TWO_PLAYER_MAP, dtype='|S1')

    def pad_with(vector, pad_width, iaxis, kwargs):
        del iaxis
        pad_value = kwargs.get('padder', ' ')
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector
    # Put walls
    game_field = np.pad(game_field, 1, pad_with, padder='=')
    # Put void
    game_field = np.pad(game_field, pad, pad_with, padder=' ')
    game_field = [row.tostring() for row in game_field]
    return game_field


class PlayerSprite(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character):
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable=['='] + list(agents.replace(character, '')), confined_to_board=True)
        self.orientation = 4
        self.init_pos = position
        self.timeout = 0

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if actions is not None:
            a = actions[agents.index(self.character)]
        else:
            return
        if self._visible:
            if things['I'].curtain[self.position[0], self.position[1]]:
                self.timeout = 25
                self._visible = False
            else:
                if a == 0:  # go upward?
                    if self.orientation == 1:
                        self._north(board, the_plot)
                    elif self.orientation == 2:
                        self._east(board, the_plot)
                    elif self.orientation == 3:
                        self._south(board, the_plot)
                    elif self.orientation == 4:
                        self._west(board, the_plot)
                elif a == 1:  # go downward?
                    if self.orientation == 1:
                        self._south(board, the_plot)
                    elif self.orientation == 2:
                        self._west(board, the_plot)
                    elif self.orientation == 3:
                        self._north(board, the_plot)
                    elif self.orientation == 4:
                        self._east(board, the_plot)
                elif a == 2:  # go leftward?
                    if self.orientation == 1:
                        self._west(board, the_plot)
                    elif self.orientation == 2:
                        self._north(board, the_plot)
                    elif self.orientation == 3:
                        self._east(board, the_plot)
                    elif self.orientation == 4:
                        self._south(board, the_plot)
                elif a == 3:  # go rightward?
                    if self.orientation == 1:
                        self._east(board, the_plot)
                    elif self.orientation == 2:
                        self._south(board, the_plot)
                    elif self.orientation == 3:
                        self._west(board, the_plot)
                    elif self.orientation == 4:
                        self._north(board, the_plot)
                elif a == 4:  # turn right?
                    if self.orientation == 4:
                        self.orientation = 1
                    else:
                        self.orientation = self.orientation + 1
                elif a == 5:  # turn left?
                    if self.orientation == 1:
                        self.orientation = 4
                    else:
                        self.orientation = self.orientation - 1
                elif a == 6:  # do nothing?
                    self._stay(board, the_plot)
        else:
            if self.timeout == 0:
                self._teleport(self.init_pos)
                self._visible = True
            else:
                self.timeout -= 1


class ScopeDrape(plab_things.Drape):
    """Scope of agent Drap"""
    def update(self, actions, board, layers, backdrop, things, the_plot):
        np.logical_and(self.curtain, False, self.curtain)
        ags = [things[c] for c in agents]
        for agent in ags:
            if agent.visible:
                pos = agent.position
                if agent.orientation == 1:
                    self.curtain[pos[0] - 1, pos[1]] = True
                elif agent.orientation == 2:
                    self.curtain[pos[0], pos[1] + 1] = True
                elif agent.orientation == 3:
                    self.curtain[pos[0] + 1, pos[1]] = True
                elif agent.orientation == 4:
                    self.curtain[pos[0], pos[1] - 1] = True
                self.curtain[:pad + 1, :] = False
                self.curtain[pad + h + 1:, :] = False
                self.curtain[:, :pad + 1] = False
                self.curtain[:, pad + w + 1:] = False


class ShotDrape(plab_things.Drape):
    """Tagging ray Drap"""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        ray_w = 0
        ray_h = agent_o_h
        np.logical_and(self.curtain, False, self.curtain)
        if actions is not None:
            for i, a in enumerate(actions):
                if a == 7:
                    agent = things[agents[i]]
                    if agent.visible:
                        pos = agent.position
                        if agent.orientation == 1:
                            self.curtain[pos[0] - ray_h:pos[0], pos[1] - ray_w:pos[1] + ray_w + 1] = True
                        elif agent.orientation == 2:
                            self.curtain[pos[0] - ray_w:pos[0] + ray_w + 1,
                                         pos[1] + 1:pos[1] + ray_h + 2] = True
                        elif agent.orientation == 3:
                            self.curtain[pos[0] + 1:pos[0] + ray_h + 2,
                                         pos[1] - ray_w:pos[1] + ray_w + 1] = True
                        elif agent.orientation == 4:
                            self.curtain[pos[0] - ray_w:pos[0] + ray_w + 1, pos[1] - ray_h:pos[1]] = True
                        self.curtain[:pad + 1, :] = False
                        self.curtain[pad + h + 1:, :] = False
                        self.curtain[:, :pad + 1] = False
                        self.curtain[:, pad + w + 1:] = False
        else:
            return


class AppleDrape(plab_things.Drape):
    """Coins Drap"""
    def __init__(self, curtain, character):
        super().__init__(curtain, character)
        self.apples = np.copy(curtain)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        rewards = []
        ags_map = np.ones(self.curtain.shape, dtype=bool)
        for i in range(len(agents)):
            rew = self.curtain[things[agents[i]].position[0], things[agents[i]].position[1]]
            if rew:
                self.curtain[things[agents[i]].position[0], things[agents[i]].position[1]] = False
            rewards.append(rew*1)
            ags_map[things[agents[i]].position[0], things[agents[i]].position[1]] = False
        the_plot.add_reward(rewards)
        # Matrix of local stock of apples
        kernel = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        L = convolve(self.curtain[pad + 1:-pad-1, pad + 1:-pad-1] * 1, kernel, mode='constant')
        probs = np.zeros(L.shape)
        probs[(L > 0) & (L <= 2)] = 0.01
        probs[(L > 2) & (L <= 4)] = 0.05
        probs[(L > 4)] = 0.1
        apple_idxs = np.argwhere(np.logical_and(np.logical_xor(self.apples, self.curtain), ags_map))
        for i, j in apple_idxs:
            self.curtain[i, j] = np.random.choice([True, False], p=[probs[i - pad - 1, j - pad - 1],
                                                                    1 - probs[i - pad - 1, j - pad - 1]])


def make_game(level_art):
    agents_order = list(agents)
    random.shuffle(agents_order)
    return ascii_art.ascii_art_to_game(
        level_art,
        what_lies_beneath=' ',
        sprites=dict(
            [(a, PlayerSprite) for a in agents]),
        drapes={'@': AppleDrape,
                'Y': ScopeDrape,
                'I': ShotDrape},
        update_schedule=['I'] + agents_order + ['Y'] + ['@'],
        z_order=['Y'] + list(agents) + ['I'] + ['@']
    )


class ObservationToArrayWithRGB(object):
    def __init__(self, colour_mapping):
        self._colour_mapping = colour_mapping
        # Rendering functions for the `board` representation and `RGB` values.
        self._renderers = {
            'RGB': rendering.ObservationToArray(value_mapping=colour_mapping)
        }

    def __call__(self, observation):
        # Perform observation rendering for agent and for video recording.
        result = {}
        for key, renderer in self._renderers.items():
            result[key] = renderer(observation)
        # Convert to [0, 255] RGB values.
        result['RGB'] = (result['RGB'] / 999.0 * 255.0).astype(np.uint8)
        return result


class HarvestGame(object):
    def __init__(self, n_agents, agent_view_h, agent_view_w):
        global agents, agent_o_h, agent_o_w, pad
        self.n_agents = n_agents
        self.action_dim = 8
        agents = "ABCDEFGHiJKLMNOPQRSTUVWXYZ"[0:n_agents]
        agent_o_h = agent_view_h
        agent_o_w = agent_view_w
        self.observation_dim = (agent_o_h, agent_o_w * 2 + 1, 3)
        pad = max(agent_o_h, agent_o_w)
        self.stp_cnt = 0
        self.game_field = two_player_map()
        self._game = make_game(self.game_field)
        self.state = None
        # Rendering params:
        COLOURS = dict([(a, (999, 0, 0)) for i, a in enumerate(agents)]  # Agents
                       + [('=', (705, 705, 705))]  # Steel Impassable wall
                       + [(' ', (0, 0, 0))]  # Black background
                       + [('@', (0, 999, 0))]  # Green Apples
                       + [('I', (750, 750, 0))]   # Yellow beam
                       + [('Y', (200, 200, 200))])  # Grey scope
        self.array_converter = ObservationToArrayWithRGB(colour_mapping=COLOURS)

    def reset(self):
        self._game = make_game(self.game_field)
        self.state, _, _ = self._game.its_showtime()
        obs, _ = self.get_observation()
        return np.asarray(obs)

    def step(self, actions):
        self.state, rewards, _ = self._game.play(actions)
        stock = np.sum(self.state.board == 64)
        obs, done = self.get_observation()
        self.stp_cnt += 1
        return np.array(obs), np.array(rewards), done, stock

    def get_observation(self):
        done = not (np.logical_or.reduce(self.state.layers['@'], axis=None))
        ags = [self._game.things[c] for c in agents]
        obs = []
        board = self.state.board
        for a in ags:
            if a.visible or a.timeout == 25:
                if a.orientation == 1:
                    ob = board[a.position[0] - agent_o_h + 1:a.position[0] + 1,
                               a.position[1] - agent_o_w:a.position[1] + agent_o_w + 1]
                    ob = np.asarray(self.array_converter(ob)['RGB']).transpose([1, 2, 0])
                elif a.orientation == 2:
                    ob = np.flip(board[a.position[0] - agent_o_w:a.position[0] + agent_o_w + 1,
                                 a.position[1]:a.position[1] + agent_o_h].T, 0)
                    ob = np.asarray(self.array_converter(ob)['RGB']).transpose([1, 2, 0])
                elif a.orientation == 3:
                    ob = np.flip(
                        np.flip(board[a.position[0]:a.position[0] + agent_o_h,
                                a.position[1] - agent_o_w:a.position[1] + agent_o_w + 1], 0), 1)
                    ob = np.asarray(self.array_converter(ob)['RGB']).transpose([1, 2, 0])
                elif a.orientation == 4:
                    ob = np.flip(board[a.position[0] - agent_o_w:a.position[0] + agent_o_w + 1,
                                 a.position[1] - agent_o_h + 1:a.position[1] + 1].T, 1)
                    ob = np.asarray(self.array_converter(ob)['RGB']).transpose([1, 2, 0])
                else:
                    ob = []
                if a.visible:
                    ob[agent_o_h - 1, agent_o_w, :] = [0, 0, 255]
            else:
                ob = np.ones((agent_o_h, agent_o_w * 2 + 1, 3), dtype=np.uint8) * 255
            obs.append(ob)
        return obs, done

    def close(self):
        self._game.the_plot.terminate_episode()

    def render(self, save_snapshots=False):
        board = self.state.board[pad:pad + h + 2, pad:pad + w + 2]
        board = np.asarray(self.array_converter(board)['RGB'])
        plt.figure(1)
        plt.imshow(board.transpose([1, 2, 0]))
        plt.axis("off")
        plt.show(block=False)
        if save_snapshots:
            plt.savefig("./frames_drqn/snap_{:0>4d}.png".format(self.stp_cnt), transparent=True, format="png",
                        bbox_inches='tight', pad_inches=0)
        plt.pause(0.001)
        plt.clf()

    @staticmethod
    def seed(seed):
        np.random.seed(seed)
        random.seed(seed)
