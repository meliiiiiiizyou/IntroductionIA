from pacman_module.game import Agent, Directions
from pacman_module.util import manhattanDistance


class PacmanAgent(Agent):
    def __init__(self, depth=2):
        super().__init__()
        self.depth = depth

    def evaluation_function(self, state):
        pacman_pos = state.getPacmanPosition()
        ghost_positions = state.getGhostPositions()
        food_positions = state.getFood().asList()

        ghost_distances = [manhattanDistance(
            pacman_pos, ghost) for ghost in ghost_positions]
        food_distances = [manhattanDistance(
            pacman_pos, food) for food in food_positions] if food_positions else [0]

        closest_ghost = min(
            ghost_distances) if ghost_distances else float('inf')
        closest_food = min(food_distances) if food_distances else float('inf')

        return state.getScore() - 2 / (closest_ghost + 1) + 1 / (closest_food + 1)

    def hminimax(self, state, depth, agent_index):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluation_function(state), None

        next_agent = (agent_index + 1) % 2
        is_pacman = (agent_index == 0)
        best_value = float('-inf') if is_pacman else float('inf')
        best_action = None

        if is_pacman:
            successors = state.generatePacmanSuccessors()
        else:
            successors = state.generateGhostSuccessors(agent_index)

        for successor, action in successors:
            if depth == 1:
                value = self.evaluation_function(successor)
            else:
                value, _ = self.hminimax(
                    successor, depth - (next_agent == 0), next_agent)

            if (is_pacman and value > best_value) or (not is_pacman and value < best_value):
                best_value, best_action = value, action

        return best_value, best_action

    def get_action(self, state):
        _, best_action = self.hminimax(state, self.depth, 0)
        return best_action if best_action else Directions.STOP
