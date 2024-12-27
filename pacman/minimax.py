from pacman_module.game import Agent, Directions


class PacmanAgent(Agent):
    def __init__(self, depth=2):
        super().__init__()
        self.depth = depth

    def evaluation_function(self, state):
        return state.getScore()

    def minimax(self, state, depth, agent_index):
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
            value, _ = self.minimax(
                successor,
                depth - (next_agent == 0),
                next_agent
            )

            if (is_pacman and value > best_value) or (not is_pacman and value < best_value):
                best_value, best_action = value, action

        return best_value, best_action

    def get_action(self, state):
        _, best_action = self.minimax(state, self.depth, 0)
        return best_action if best_action else Directions.STOP
