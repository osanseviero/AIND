"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def proportion_heuristic(game, player):
    """Divide the number of moves and get a proportion"""
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return own_moves/opp_moves

def center_heuristic(game, player):
    """Higher value if player is closer to the center"""
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_pos = game.get_player_location(player)
    opp_pos = game.get_player_location(game.get_opponent(player))

    own_dist_from_center = math.sqrt((own_pos[0] - game.width/2 )**2 + (own_pos[1] - game.height/2)**2)
    
    return 10 - own_dist_from_center

def weighted_moves(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 2 * opp_moves)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    return proportion_heuristic(game, player)




class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=70.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        
        # Check if there are no valid moves
        if len(legal_moves) == 0:
            return (-1, -1)

        move = (-1, -1)
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.method == 'minimax':
                if not self.iterative:
                    move = self.minimax(game, self.search_depth)[1]
                else:
                    x = 1
                    while True:
                        move = self.minimax(game, x)[1]
                        x = x + 1

            if self.method == 'alphabeta':
                if not self.iterative:
                    move = self.alphabeta(game, self.search_depth)[1]
                else:
                    x = 1
                    while True:
                        move = self.alphabeta(game, self.search_depth)[1]
                        x = x + 1
            pass

        except Timeout:
            pass

        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0 or depth == 0:
            return self.score(game, self), (-1.0, -1.0)

        if maximizing_player:
            max_score_move = [-math.inf, (-1, -1)]
            max_score = max_score_move[0]
            for legal_move in legal_moves:
                minimaxResult = self.minimax(game.forecast_move(legal_move), depth-1, not maximizing_player)
                if minimaxResult[0] > max_score:
                    max_score = minimaxResult[0]
                    max_score_move = minimaxResult[0], legal_move
            return max_score_move

        if not maximizing_player:
            min_score_move = [math.inf, (-1, -1)]
            min_score = min_score_move[0]
            for legal_move in game.get_legal_moves():
                minimaxResult = self.minimax(game.forecast_move(legal_move), depth-1, not maximizing_player)
                if minimaxResult[0] < min_score:
                    min_score  = minimaxResult[0]
                    min_score_move = minimaxResult[0], legal_move
            return min_score_move
       

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing  the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0 or depth == 0:
            return self.score(game, self), (-1.0, -1.0)

        if maximizing_player:
            max_score_move = ([-math.inf, (-1, -1)])
            max_score = max_score_move[0]
            for legal_move in legal_moves:
                minimaxResult = self.alphabeta(game.forecast_move(legal_move), depth-1, alpha, beta, not maximizing_player)

                if minimaxResult[0] > max_score:
                    max_score = minimaxResult[0]
                    max_score_move = minimaxResult[0], legal_move

                # Ignore the rest
                if minimaxResult[0] >= beta:
                    return minimaxResult[0], legal_move
                alpha = max(alpha, minimaxResult[0])
            return max_score_move

        if not maximizing_player:
            min_score_move = [math.inf, (-1, -1)]
            min_score = min_score_move[0]
            for legal_move in game.get_legal_moves():
                minimaxResult = self.alphabeta(game.forecast_move(legal_move), depth-1, alpha, beta, not maximizing_player)
                if minimaxResult[0] < min_score:
                    min_score  = minimaxResult[0]
                    min_score_move = minimaxResult[0], legal_move
                # Ignore the rest
                if minimaxResult[0] <= alpha:
                    return minimaxResult[0], legal_move
                beta = min(beta, minimaxResult[0])
            return min_score_move










