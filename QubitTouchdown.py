import pennylane as qml
import numpy as np
import random
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import math
from PIL import Image, ImageDraw, ImageTk

# =============================================================================
# Ten quantum-computing facts.
# One is shown after each AI move.
# They can be toggled off in-game.
# TODO: Add more facts
# =============================================================================
QUANTUM_FACTS = [
    "A qubit can be in a superposition of |0⟩ and |1⟩ at once, until it is measured.",
    "Measuring a qubit collapses its superposition to a definite 0 or 1.",
    "The Bloch sphere pictures any single-qubit pure state as a point on a sphere.",
    "The Hadamard (H) gate turns |0⟩ into |+⟩, an equal mix of |0⟩ and |1⟩.",
    "Entangled qubits share correlations stronger than any classical bits can.",
    "Every quantum gate is reversible — apply its inverse and you get the state back.",
    "The no-cloning theorem proves an unknown quantum state cannot be perfectly copied.",
    "The Pauli-X gate is the quantum version of the classical NOT gate.",
    "Shor's algorithm can factor large integers far faster than known classical methods.",
    "Grover's search finds an item in an unsorted list with a quadratic speedup.",
]

class QubitTouchdown:
    def __init__(self, is_ai=False):
        self._ai_after_id = None
        self.device = qml.device('default.qubit', wires=1)
        # Randomize initial state
        self.current_state = random.choice(['|0⟩', '|1⟩'])
        self.player_scores = [0, 0]
        self.current_player = 0
        self.game_states = ['|0⟩', '|1⟩', '|+⟩', '|-⟩', '|i⟩', '|-i⟩']

        # AI flag. Player index 1 == "Player 2" is the AI when this is True.
        self.is_ai = is_ai

        self.deck = self.create_deck()
        random.shuffle(self.deck)

        self.player_hands = [[], []]
        self.deal_cards()

    def create_deck(self):
        cards = {
            'I': 3, 'X': 4, 'Y': 9, 'Z': 7,
            'H': 7, 'S': 7, '√X': 12, 'Measurement': 3
        }
        """
        # Smaller deck for debugging
        cards = {
            'I': 1, 'X': 1, 'Y': 1, 'Z': 1,
            'H': 1, 'S': 1, '√X': 1, 'Measurement': 1
        }
        """
        deck = []
        for card, count in cards.items():
            deck.extend([card] * count)
        return deck

    def deal_cards(self):
        # 4 cards each
        for _ in range(4):
            # 2 players
            for player in range(2):
                if self.deck:
                    self.player_hands[player].append(self.deck.pop())

    def is_game_over(self):
        # Check if both players have run out of cards
        return len(self.player_hands[0]) == 0 and len(self.player_hands[1]) == 0 and len(self.deck) == 0

    def get_winner(self):
        if self.player_scores[0] > self.player_scores[1]:
            return "Player 1"
        elif self.player_scores[1] > self.player_scores[0]:
            return "Player 2"
        else:
            return "Tie"

    def get_state_vector(self, state_name):
        # Convert state name to state vector using PennyLane

        @qml.qnode(self.device)
        def prepare_state():
            """
            Checks the requested state name and applies
            the necessary gates to go from the default state |0⟩
            to the desired state.
            """
            if state_name == '|0⟩':
                pass
            elif state_name == '|1⟩':
                qml.PauliX(wires=0)
            elif state_name == '|+⟩':
                qml.Hadamard(wires=0)
            elif state_name == '|-⟩':
                qml.PauliZ(wires=0)
                qml.Hadamard(wires=0)
            elif state_name == '|i⟩':
                qml.Hadamard(wires=0)
                # S gate adds +i phase
                qml.S(wires=0)
            elif state_name == '|-i⟩':
                qml.Hadamard(wires=0)
                # adjoint S adds -i phase
                qml.adjoint(qml.S)(wires=0)
            return qml.state()

        return prepare_state()

    def apply_gate_pennylane(self, current_state, gate):
        # Apply quantum gate using PennyLane simulation

        @qml.qnode(self.device)
        def circuit():
            if current_state == '|0⟩':
                pass
            elif current_state == '|1⟩':
                qml.PauliX(wires=0)
            elif current_state == '|+⟩':
                qml.Hadamard(wires=0)
            elif current_state == '|-⟩':
                qml.PauliZ(wires=0)
                qml.Hadamard(wires=0)
            elif current_state == '|i⟩':
                qml.Hadamard(wires=0)
                qml.S(wires=0)
            elif current_state == '|-i⟩':
                qml.Hadamard(wires=0)
                qml.adjoint(qml.S)(wires=0)

            if gate == 'I':
                pass
            elif gate == 'X':
                qml.PauliX(wires=0)
            elif gate == 'Y':
                qml.PauliY(wires=0)
            elif gate == 'Z':
                qml.PauliZ(wires=0)
            elif gate == 'H':
                qml.Hadamard(wires=0)
            elif gate == 'S':
                qml.S(wires=0)
            elif gate == '√X':
                # a rotation on the X-axis by 90 degrees is
                # equivalent to the square root of the Pauli-X gate
                qml.RX(np.pi / 2, wires=0)

            return qml.state()

        final_state = circuit()

        # Convert the state vector back to one of the six game states
        return self.identify_closest_state(final_state)

    def identify_closest_state(self, state_vector):
        """
        Identify which of the game states the quantum state is closest to
        This is necessary in simulations
        """
        # Normalize the state vector first
        state_vector = state_vector / np.linalg.norm(state_vector)

        # Define the six game states as state vectors
        game_state_vectors = {
            '|0⟩': np.array([1, 0]),
            '|1⟩': np.array([0, 1]),
            '|+⟩': np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]),
            '|-⟩': np.array([1 / np.sqrt(2), -1 / np.sqrt(2)]),
            '|i⟩': np.array([1 / np.sqrt(2), 1j / np.sqrt(2)]),
            '|-i⟩': np.array([1 / np.sqrt(2), -1j / np.sqrt(2)])
        }

        best_fidelity = 0
        best_state = None

        for state_name, target_vector in game_state_vectors.items():
            # Fidelity = |<ψ|φ>|^2
            fidelity = np.abs(np.vdot(state_vector, target_vector)) ** 2

            if abs(fidelity - 1.0) < 1e-10:
                return state_name

            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_state = state_name

        if best_fidelity > 0.999:
            return best_state

        print(f"DEBUG: No exact match found. Best: {best_state} with fidelity {best_fidelity:.10f}")
        return best_state

    def get_bloch_coordinates(self, state):
        coordinates = {
            '|0⟩': (0, 0, 1),
            '|1⟩': (0, 0, -1),
            '|+⟩': (1, 0, 0),
            '|-⟩': (-1, 0, 0),
            '|i⟩': (0, 1, 0),
            '|-i⟩': (0, -1, 0)
        }
        return coordinates.get(state, (0, 0, 0))

    def measurement(self):
        # Collapse to |0⟩ or |1⟩ using PennyLane

        @qml.qnode(self.device)
        def measure_circuit():
            if self.current_state == '|0⟩':
                pass
            elif self.current_state == '|1⟩':
                qml.PauliX(wires=0)
            elif self.current_state == '|+⟩':
                qml.Hadamard(wires=0)
            elif self.current_state == '|-⟩':
                qml.PauliZ(wires=0)
                qml.Hadamard(wires=0)
            elif self.current_state == '|i⟩':
                qml.Hadamard(wires=0)
                qml.S(wires=0)
            elif self.current_state == '|-i⟩':
                qml.Hadamard(wires=0)
                qml.adjoint(qml.S)(wires=0)

            return qml.probs(wires=0)

        probabilities = measure_circuit()
        # Collapse to |0⟩ or |1⟩ based on probabilities
        result = random.choices([0, 1], weights=probabilities)[0]
        return '|0⟩' if result == 0 else '|1⟩'

    def play_card(self, card_index):
        if card_index >= len(self.player_hands[self.current_player]):
            return "Invalid card selection", False, None

        card = self.player_hands[self.current_player].pop(card_index)
        touchdown = False
        scoring_player = None

        print(f"DEBUG: Player {self.current_player + 1} playing {card} from state {self.current_state}")

        if card == 'Measurement':
            if self.current_state in ['|0⟩', '|1⟩']:
                result = "Measurement: Ball was already at |0⟩ or |1⟩ - no change"
            else:
                old_state = self.current_state
                self.current_state = self.measurement()
                result = f"Measurement collapsed state from {old_state} to {self.current_state}"
        else:
            old_state = self.current_state
            # Use PennyLane to apply the gate
            self.current_state = self.apply_gate_pennylane(old_state, card)
            result = f"Applied {card} gate. Ball moved from {old_state} to {self.current_state}"

        # Check for touchdown & store current player
        current_player_before_switch = self.current_player
        touchdown = self.check_touchdown(current_player_before_switch)

        print(
            f"DEBUG: Touchdown check for Player {current_player_before_switch + 1} in state {self.current_state}: {touchdown}")

        if touchdown:
            scoring_player = current_player_before_switch + 1
            result += f"\nTOUCHDOWN! Player {scoring_player} scores!"
            self.player_scores[current_player_before_switch] += 1
            # Reset ball position randomly
            self.current_state = random.choice(['|0⟩', '|1⟩'])
            print(f"DEBUG: Reset ball to {self.current_state} after touchdown")

        # Draw new card if available
        if self.deck:
            self.player_hands[self.current_player].append(self.deck.pop())

        # Switch players
        if not touchdown:
            self.current_player = 1 - self.current_player
        else:
            self.current_player = 1 - self.current_player  # Change possession after touchdown

        print(f"DEBUG: Next player: {self.current_player + 1}")
        return result, touchdown, scoring_player

    def check_touchdown(self, player):
        print(f"DEBUG: Checking touchdown for Player {player + 1} in state {self.current_state}")

        # Players score in their OPPONENT'S endzone, not their own
        # Player 1's endzone is |-⟩, so Player 2 scores there
        # Player 2's endzone is |+⟩, so Player 1 scores there

        if player == 0 and self.current_state == '|+⟩':  # Player 1 in Player 2's endzone
            print("DEBUG: Player 1 touchdown in Player 2's endzone (|+⟩) confirmed!")
            return True
        elif player == 1 and self.current_state == '|-⟩':  # Player 2 in Player 1's endzone
            print("DEBUG: Player 2 touchdown in Player 1's endzone (|-⟩) confirmed!")
            return True

        print(f"DEBUG: No touchdown - Player {player + 1} in {self.current_state}")
        return False

    # =========================================================================
    # The AI is Player 2 and therefore scores in Player 1's endzone,
    # which is the |-⟩ state. This simulates every card in hand and
    # picks the one that lands the ball closest to (or exactly on) |-⟩.
    # =========================================================================

    # The AI wants to move the ball to this state to score.
    AI_TARGET_STATE = '|-⟩'

    def _simulate_card_result(self, card):
        """
        Return the resulting board state if card were played from the current
        state, without mutating the real game. Used only for AI strategy.

        For gate cards use apply_gate_pennylane simulation so the AI's
        prediction always matches the real rules. The Measurement card
        is probabilistic, so the AI cannot reliably predict it. AI treats
        it as "no useful change" & returns the current state.
        """
        if card == 'Measurement':
            return self.current_state
        return self.apply_gate_pennylane(self.current_state, card)

    def _distance_to_target(self, state_name):
        """
        Straight line distance on the Bloch sphere between a state
        and the AI's target endzone state (|-⟩ at coordinates (-1, 0, 0)).
        Smaller = closer to scoring. A distance of 0 means the ball is exactly
        on the AI's endzone (a touchdown).
        """
        a = self.get_bloch_coordinates(state_name)
        b = self.get_bloch_coordinates(self.AI_TARGET_STATE)
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    def ai_select_card(self):
        """
        Decide which card index the AI should play.
        TODO: Fix logic for gates on |-⟩ & |+⟩.
        Only the Identity gate leaves |-⟩ unchanged (unless we count global phases)
        For |+⟩, I, X, & sqrtX gates leave |+⟩ unchanged.

        Strategy:
          1. If any card lands the ball exactly on |-⟩, play it immediately
             NOTE: this handles the hard case where the human has parked the ball
             in the AI's endzone (|-⟩): the AI automatically chooses a card that lands on |-⟩.
             Because the simulation uses apply_gate_pennylane, the
             AI always agrees with the game about which cards score.
             Which gates keep |-⟩ unchanged aren't hard-coded.
             In the game the gates that score from a ball labelled |-⟩ are Y and Z,
             not I/X/√X, because of how |-⟩ is prepared. TODO: fix
          2. Otherwise, pick the card that minimises the Bloch-sphere distance
             to |-⟩ (moves the ball as close to the endzone as possible).
          3. If no card actually gets the AI closer than it already is ("no card
             helps"), fall back to a random card so the AI still makes a move.
        """
        hand = self.player_hands[self.current_player]
        if not hand:
            return 0

        print(f"DEBUG[AI]: Hand: {hand}")

        best_idx = None
        best_dist = float('inf')

        for i, card in enumerate(hand):
            result_state = self._simulate_card_result(card)

            # 1. Immediate scoring move wins outright.
            if result_state == self.AI_TARGET_STATE:
                print(f"DEBUG[AI]: Found scoring card {card} at index {i}")
                return i

            # 2. Track the card that gets AI closest to the endzone.
            dist = self._distance_to_target(result_state)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # 3. Only commit to the best card if it is a genuine improvement over
        # just standing still. Otherwise, play randomly.
        # TODO: Prevent AI from playing measurement inside target endzone
        current_dist = self._distance_to_target(self.current_state)
        if best_idx is None or best_dist >= current_dist - 1e-9:
            choice = random.randrange(len(hand))
            print(f"DEBUG[AI]: No improving card; random choice index {choice} ({hand[choice]})")
            return choice

        print(f"DEBUG[AI]: Best card {hand[best_idx]} at index {best_idx} (dist {best_dist:.3f})")
        return best_idx

    def get_game_state(self):
        return {
            'current_state': self.current_state,
            'player_scores': self.player_scores,
            'current_player': self.current_player,
            'player_hands': self.player_hands,
            'cards_remaining': len(self.deck),
            'bloch_coords': self.get_bloch_coordinates(self.current_state),
            'is_game_over': self.is_game_over(),
            'winner': self.get_winner() if self.is_game_over() else None
        }


class BlochSphere3D:
    def __init__(self, parent):
        # CHANGED: slightly smaller figure so the 3D sphere and the new 2D board
        # fit side by side without overflowing the window.
        self.figure = Figure(figsize=(4.3, 4.3), dpi=100)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.draw()

    def plot_sphere(self, current_state, coords):
        self.ax.clear()

        # Create a sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        self.ax.plot_wireframe(x, y, z, color='gray', alpha=0.3, linewidth=0.5)

        state_positions = {
            '|0⟩': (0, 0, 1),
            '|1⟩': (0, 0, -1),
            '|+⟩': (1, 0, 0),
            '|-⟩': (-1, 0, 0),
            '|i⟩': (0, 1, 0),
            '|-i⟩': (0, -1, 0)
        }

        # Plot state positions
        for state, pos in state_positions.items():
            color = 'red' if state == current_state else 'blue'
            size = 100 if state == current_state else 50
            self.ax.scatter(*pos, color=color, s=size, alpha=0.8)
            offset = 0.15
            label_pos = (pos[0] + offset, pos[1] + offset, pos[2] + offset)
            self.ax.text(*label_pos, state, fontsize=12, ha='center')

        # Draw axes
        axis_length = 1.2
        self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2)
        self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, linewidth=2)
        self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, linewidth=2)

        # Label axes
        self.ax.text(axis_length, 0, 0, 'X', fontsize=12, color='r')
        self.ax.text(0, axis_length, 0, 'Y', fontsize=12, color='g')
        self.ax.text(0, 0, axis_length, 'Z', fontsize=12, color='b')

        theta = np.linspace(0, 2 * np.pi, 100)

        # Equator
        x_eq = np.cos(theta)
        y_eq = np.sin(theta)
        z_eq = np.zeros_like(theta)
        self.ax.plot(x_eq, y_eq, z_eq, 'k-', alpha=0.3, linewidth=1)

        # YZ-plane
        x_yz = np.zeros_like(theta)
        y_yz = np.cos(theta)
        z_yz = np.sin(theta)
        self.ax.plot(x_yz, y_yz, z_yz, 'k-', alpha=0.3, linewidth=1)

        # XZ-plane
        x_xz = np.cos(theta)
        y_xz = np.zeros_like(theta)
        z_xz = np.sin(theta)
        self.ax.plot(x_xz, y_xz, z_xz, 'k-', alpha=0.3, linewidth=1)

        self.ax.set_box_aspect([1, 1, 1])

        self.ax.set_xlim([-1.2, 1.2])
        self.ax.set_ylim([-1.2, 1.2])
        self.ax.set_zlim([-1.2, 1.2])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Set title with current coordinates
        self.ax.set_title(f'Current State: {current_state}\n({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f})')

        self.canvas.draw()


# =============================================================================
# NEW: BoardView
# A 2D schematic of the six quantum states arranged exactly like the printed
# Qubit Touchdown board:
#
#                 |+⟩            <- top: Player 2's endzone (Player 1 scores)
#               /     \
#           |0⟩         |-i⟩
#            |   (cross) |
#           |i⟩          |1⟩
#               \     /
#                 |-⟩            <- bottom: Player 1's endzone (Player 2 scores)
#
# It draws static gate arrows for visual reference (X, Y, Z, H, S, √X) and
# highlights the current ball position with a semi-transparent red dot.
# =============================================================================
class BoardView(tk.Canvas):
    # Logical pixel positions of each state node on the canvas (diamond layout
    # matching Fig. 2.2a in the rulebook).
    NODE_POS = {
        '|+⟩':  (190, 60),
        '|0⟩':  (95, 195),
        '|-i⟩': (285, 195),
        '|i⟩':  (95, 350),
        '|1⟩':  (285, 350),
        '|-⟩':  (190, 485),
    }
    NODE_R = 27          # node circle radius
    WIDTH = 380
    HEIGHT = 545

    # Perimeter gate moves (source, destination, label). Directions/labels were
    # verified against the rulebook's board and the game's own gate simulation.
    PERIMETER_EDGES = [
        ('|0⟩', '|+⟩', 'H'),
        ('|-i⟩', '|+⟩', 'S'),
        ('|0⟩', '|-i⟩', '√X'),
        ('|-i⟩', '|1⟩', '√X'),
        ('|1⟩', '|i⟩', '√X'),
        ('|i⟩', '|0⟩', '√X'),
        ('|i⟩', '|-⟩', 'S'),
        ('|1⟩', '|-⟩', 'H'),
    ]

    # The two crossing diagonals through the center
    # TODO: Tilt gates along the diagonals
    DIAGONAL_EDGES = [
        ('|0⟩', '|1⟩', 'X,Y'),       # X|0⟩=|1⟩, Y|0⟩=|1⟩ (and the reverse)
        ('|-i⟩', '|i⟩', 'X,Z,H'),    # X,Z,H all map |-i⟩↔|i⟩
    ]

    def __init__(self, parent):
        super().__init__(parent, width=self.WIDTH, height=self.HEIGHT, highlightthickness=0, bg='#0b3d0b')
        self._highlight_photo = None
        self._highlight_cache = {}

    # ---- geometry -------------------------------------------
    def _shrink_to_edge(self, x1, y1, x2, y2, r):
        """Pull line's endpoints in by radius r so arrows touch node's edge,
        not node center."""
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy) or 1
        ux, uy = dx / dist, dy / dist
        return (x1 + ux * r, y1 + uy * r, x2 - ux * r, y2 - uy * r)

    def _draw_arrow(self, src, dst, label, double=False):
        x1, y1 = self.NODE_POS[src]
        x2, y2 = self.NODE_POS[dst]
        sx1, sy1, sx2, sy2 = self._shrink_to_edge(x1, y1, x2, y2, self.NODE_R + 2)
        arrow = tk.BOTH if double else tk.LAST
        self.create_line(sx1, sy1, sx2, sy2, fill='white', width=2, arrow=arrow, arrowshape=(10, 12, 4))
        # Place the gate label near the midpoint of the arrow,
        # also place it next to the arrow rather than on top of it.
        mx, my = (sx1 + sx2) / 2, (sy1 + sy2) / 2
        dx, dy = sx2 - sx1, sy2 - sy1
        dist = math.hypot(dx, dy) or 1
        px, py = -dy / dist, dx / dist
        self.create_text(mx + px * 12, my + py * 12, text=label,
                         fill='#ffd23f', font=('Arial', 11, 'bold'))

    def draw(self, current_state):
        """Redraw the whole board and highlight current_state"""
        self.delete('all')

        # Top endzone (blue) = Player 2's endzone, where Player 1 scores (|+⟩).
        self.create_rectangle(0, 0, self.WIDTH, 120, fill='#3a1c5a', outline='')
        # Bottom endzone (orange) = Player 1's endzone, where Player 2 scores (|-⟩).
        self.create_rectangle(0, self.HEIGHT - 120, self.WIDTH, self.HEIGHT, fill='#cd7f32', outline='')
        # Decorative yard lines
        for yy in (120, 220, 320, 425):
            self.create_line(0, yy, self.WIDTH, yy, fill='#2f7d36', width=1)

        # --- gate arrows ---
        for src, dst, label in self.PERIMETER_EDGES:
            self._draw_arrow(src, dst, label, double=False)
        for src, dst, label in self.DIAGONAL_EDGES:
            self._draw_arrow(src, dst, label, double=True)

        # --- the six state nodes ---
        for state, (x, y) in self.NODE_POS.items():
            self.create_oval(x - self.NODE_R, y - self.NODE_R,
                             x + self.NODE_R, y + self.NODE_R,
                             fill='#ffcf33', outline='#7a5c00', width=2)
            self.create_text(x, y, text=state, font=('Arial', 12, 'bold'), fill='#1a1a1a')

        # --- semi-transparent red highlight on the current state ---
        if current_state in self.NODE_POS:
            self._draw_highlight(*self.NODE_POS[current_state])

        # --- tiny caption noting the gates that don't move the qubit ---
        # TODO: Change this once the endzone logic is fixed
        self.create_text(self.WIDTH / 2, self.HEIGHT - 8,
                         text="Identity (I) gate keeps the ball still.",
                         fill='#000000', font=('Arial', 8))

    def _draw_highlight(self, x, y):
        r = self.NODE_R + 6
        size = r * 2
        if size not in self._highlight_cache:
            img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            # alpha = 120
            d.ellipse([0, 0, size - 1, size - 1], fill=(255, 0, 0, 120))
            self._highlight_cache[size] = ImageTk.PhotoImage(img)
        self._highlight_photo = self._highlight_cache[size]
        self.create_image(x, y, image=self._highlight_photo)


class QubitTouchdownGUI:
    # Can now reuse an existing root so the start screen can
    # close without creating a second Tk instance, and accepts the
    # is_ai flag that turns Player 2 into the computer.
    def __init__(self, root=None, is_ai=False):
        self._exiting = False
        self._ai_after_id = None
        self._owns_root = root is None
        self.root = root if root is not None else tk.Tk()
        self.root.title("Qubit Touchdown")

        # Window geometry
        window_width = 1320
        window_height = 860
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")  # Check if this resolution is good

        self.is_ai = is_ai
        self.game = QubitTouchdown(is_ai=is_ai)

        # quantum facts state
        self.facts_enabled = True
        self.fact_index = 0

        self.setup_gui()
        self.update_display()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Game info & controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        title_label = ttk.Label(left_frame, text="Qubit Touchdown",
                                font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Game now acknowledges the AI opponent when in single-player.
        mode_line = "Single Player vs. AI\n" if self.is_ai else ""
        instructions = ttk.Label(left_frame,
                                 text=mode_line +
                                      "Score by moving the ball into your OPPONENT'S endzone.\n"
                                      "Player 1 scores in |+⟩\n"
                                      "Player 2 scores in |-⟩",
                                 font=("Arial", 10),
                                 justify=tk.CENTER)
        instructions.pack(pady=5)

        # Scores
        self.score_label = ttk.Label(left_frame, text="", font=("Arial", 12))
        self.score_label.pack(anchor=tk.W, pady=5)

        # Current state
        self.state_label = ttk.Label(left_frame, text="", font=("Arial", 12,))
        self.state_label.pack(anchor=tk.W, pady=5)

        # Current player
        self.player_label = ttk.Label(left_frame, text="", font=("Arial", 14, "bold"))
        self.player_label.pack(anchor=tk.W, pady=5)

        # Cards remaining
        self.cards_label = ttk.Label(left_frame, text="", font=("Arial", 10))
        self.cards_label.pack(anchor=tk.W, pady=5)

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Player hand
        hand_label = ttk.Label(left_frame, text="Your Hand:", font=("Arial", 11, "bold"))
        hand_label.pack(anchor=tk.W, pady=(10, 5))

        self.hand_frame = ttk.Frame(left_frame)
        self.hand_frame.pack(fill=tk.X, pady=5)

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Quantum fact area. The fact label updates after each AI move.
        # Button lets player toggle off/on facts.
        fact_header = ttk.Frame(left_frame)
        fact_header.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(fact_header, text="Quantum Fact:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.facts_button = ttk.Button(fact_header, text="Hide Facts", command=self.toggle_facts, width=12)
        self.facts_button.pack(side=tk.RIGHT)

        self.fact_label = ttk.Label(left_frame, text="(facts appear after the AI moves)",
                                    font=("Arial", 9, "italic"), wraplength=300,
                                    justify=tk.LEFT, foreground="#1a4d8f")
        self.fact_label.pack(anchor=tk.W, pady=(0, 8))

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Game log
        log_label = ttk.Label(left_frame, text="Game Log:", font=("Arial", 11, "bold"))
        log_label.pack(anchor=tk.W, pady=(10, 5))

        self.log_text = tk.Text(left_frame, height=12, width=40, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Scrollbar for log
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # The right side now holds the 3D Bloch sphere & the 2D
        # board side by side, with the legend underneath.
        viz_frame = ttk.Frame(right_frame)
        viz_frame.pack(fill=tk.BOTH, expand=True)

        # 3D Bloch sphere
        self.bloch_3d = BlochSphere3D(viz_frame)
        self.bloch_3d.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)

        # 2D board
        board_holder = ttk.Frame(viz_frame)
        board_holder.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 10), pady=10)
        ttk.Label(board_holder, text="Game Board", font=("Arial", 11, "bold")).pack(pady=(0, 4))
        self.board_view = BoardView(board_holder)
        self.board_view.pack()

        legend_frame = ttk.Frame(right_frame)
        legend_frame.pack(fill=tk.X, pady=10)

        states_info = [
            ("|0⟩", "North Pole - Start Position"),
            ("|1⟩", "South Pole - Start Position"),
            ("|+⟩", "Player 2's Endzone - Player 1 scores here"),
            ("|-⟩", "Player 1's Endzone - Player 2 scores here"),
            ("|i⟩", "+Y Axis"),
            ("|-i⟩", "-Y Axis")
        ]

        for i, (state, desc) in enumerate(states_info):
            lbl = ttk.Label(legend_frame, text=f"{state}: {desc}", font=("Arial", 9))
            lbl.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)

    # Toggle handler for the quantum facts
    def toggle_facts(self):
        self.facts_enabled = not self.facts_enabled
        if self.facts_enabled:
            self.facts_button.config(text="Hide Facts")
            self.fact_label.config(text="(facts will appear after the AI moves)")
        else:
            self.facts_button.config(text="Show Facts")
            self.fact_label.config(text="Quantum facts hidden.")

    # Show the next fact (called after each AI move, only if enabled).
    def show_next_fact(self):
        if not self.facts_enabled:
            return
        try:
            if self.fact_label.winfo_exists():
                fact = QUANTUM_FACTS[self.fact_index % len(QUANTUM_FACTS)]
                self.fact_index += 1
                self.fact_label.config(text=f"💡 {fact}")
        except tk.TclError:
            pass

    def update_display(self):
        state = self.game.get_game_state()

        # Determine whose turn it is
        cp = state['current_player']
        ai_now = self.is_ai and cp == 1  # True while it's the AI's (index 1) turn

        # Update labels
        self.score_label.config(text=f"Scores: Player 1: {state['player_scores'][0]} | Player 2: {state['player_scores'][1]}")
        self.state_label.config(text=f"Current State: {state['current_state']}")

        # Clearer current-player text, including an AI "thinking" notice.
        if ai_now:
            self.player_label.config(text="AI (Player 2) is thinking…")
        else:
            who = "Player 1 (You)" if (self.is_ai and cp == 0) else f"Player {cp + 1}"
            self.player_label.config(text=f"Current Player: {who}")

        self.cards_label.config(text=f"Cards remaining: {state['cards_remaining']}")

        # Update 3D Bloch sphere
        self.bloch_3d.plot_sphere(state['current_state'], state['bloch_coords'])

        # Update the 2D board schematic too.
        self.board_view.draw(state['current_state'])

        # Update hand buttons
        for widget in self.hand_frame.winfo_children():
            widget.destroy()

        hand = state['player_hands'][cp]
        for i, card in enumerate(hand):
            btn = ttk.Button(self.hand_frame, text=card,
                             command=lambda idx=i: self.play_card(idx))
            # While it's the AI's turn, show its cards but disable them so
            # the player cannot play on the AI's behalf.
            if ai_now:
                btn.state(['disabled'])
            btn.grid(row=0, column=i, padx=2, pady=2)

        # Check for game over
        if state['is_game_over']:
            self.show_game_over_popup(state['winner'])
            # Check: Do GUI updates happen after user responds
            self.ask_restart_or_exit()

    def show_game_over_popup(self, winner):
        score1, score2 = self.game.player_scores

        if winner == "Tie":
            message = f"Game Over!\n\nIt's a tie!\n\nFinal Score:\nPlayer 1: {score1}\nPlayer 2: {score2}"
        else:
            message = f"Game Over!\n\n{winner} wins!\n\nFinal Score:\nPlayer 1: {score1}\nPlayer 2: {score2}"

        # Add game over message to log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"\n\n{message.replace('Game Over!\\n\\n', '')}")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

        # Show popup
        messagebox.showinfo("Game Over!", message)

    # Restart logic: Clear logs, reset fact index, call update_display()
    def restart_game(self):
        self.cancel_ai_turn()
        self._exiting = False
        self.game = QubitTouchdown(is_ai=self.is_ai)
        self.fact_index = 0
        self.log_text.delete("1.0", tk.END)
        self.update_display()
        self._ai_after_id = None
        print("Restarting game.")


    # Prompt the user to restart or exit
    def ask_restart_or_exit(self):
        self.cancel_ai_turn()
        response = messagebox.askquestion("Restart?", "Select 'Yes' to restart or 'No' to exit.")
        if response == "yes":
            self.restart_game()
        else:
            self._exiting = True
            self.root.destroy()

    # This is necessary for restarting because of the turn scheduler
    def cancel_ai_turn(self):
        if hasattr(self, '_ai_after_id') and self._ai_after_id is not None:
            self.root.after_cancel(self._ai_after_id)
            self._ai_after_id = None

    def play_card(self, card_index):
        # Make sure the user can't play cards on the AI's turn
        if self.is_ai and self.game.current_player == 1:
            return

        result, touchdown, scoring_player = self.game.play_card(card_index)

        # Cancel any AI turn
        self.cancel_ai_turn()

        # Schedule AI move if it's now the AI's turn and game isn't over
        if (self.is_ai and not self.game.is_game_over()
                and self.game.current_player == 1):
            # Store the after id so it can be canceled later
            self._ai_after_id = self.root.after(500, self.ai_turn)

        # Add to log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"\n{result}")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

        self.update_display()

        if touchdown and scoring_player is not None:
            messagebox.showinfo("Touchdown!", f"Player {scoring_player} scored!")


    # =========================================================================
    # GUI-side of AI moves.
    # Gets which card to play, plays it, logs the result,
    # refreshes the display, shows a touchdown popup if needed, and reveals the
    # next quantum fact. The actual card choice lives in QubitTouchdown.ai_select_card.
    # =========================================================================
    def ai_turn(self):
        # Stop the AI if the game is over
        # We mustn't unleash the beast upon the world... yet.
        if self._exiting or self.game.is_game_over() or not self.root.winfo_exists():
            return
        # Only act if it really is the AI's turn and the game isn't over.
        if not self.is_ai or self.game.current_player != 1 or self.game.is_game_over():
            return

        card_index = self.game.ai_select_card()       # decide using the heuristic
        chosen_card = self.game.player_hands[1][card_index] if card_index < len(self.game.player_hands[1]) else "?"
        result, touchdown, scoring_player = self.game.play_card(card_index)

        # Log the AI's move
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"\n[AI plays {chosen_card}] {result}")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

        self.update_display()

        if touchdown and scoring_player is not None:
            messagebox.showinfo("Touchdown!", f"AI (Player {scoring_player}) scored!")

        # Surface a quantum fact after the AI's move.
        self.show_next_fact()

    def run(self):
        # Only start a mainloop if we created the root ourselves. When launched
        # from the start screen, that screen owns/started the loop already.
        if self._owns_root:
            self.root.mainloop()


# =============================================================================
# StartScreen
# A simple launcher window with two buttons:
#   - "Single Player"           -> starts the game with the AI enabled
#   - "Multiplayer (Coming Soon!)" -> disabled placeholder
# Choosing single player clears this window and builds the main game in the same Tk root
# =============================================================================
class StartScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Qubit Touchdown")
        # self.root.geometry("420x320") #TODO: Make StartScreen bigger
        self.root.geometry("600x320")

        # A container to destroy when the game starts
        self.container = ttk.Frame(self.root, padding="30")
        self.container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self.container, text="🏈  Qubit Touchdown",
                  font=("Arial", 22, "bold")).pack(pady=(20, 6))
        ttk.Label(self.container, text="A quantum computing board game",
                  font=("Arial", 11, "italic")).pack(pady=(0, 30))

        # Button: starts single-player vs. AI
        ttk.Button(self.container, text="Single Player",
                   command=self.start_single_player).pack(fill=tk.X, pady=6, ipady=6)

        # Disabled placeholder for future multiplayer mode
        multiplayer_btn = ttk.Button(self.container, text="Multiplayer (Coming Soon!)")
        multiplayer_btn.state(['disabled'])
        multiplayer_btn.pack(fill=tk.X, pady=6, ipady=6)

    def start_single_player(self):
        # Remove the start-screen widgets, then build the game in this same root
        # with the AI enabled. The mainloop started in run() keeps running.
        self.container.destroy()
        QubitTouchdownGUI(root=self.root, is_ai=True)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    StartScreen().run()