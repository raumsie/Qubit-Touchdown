import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
import random
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import math


class QubitTouchdown:
    def __init__(self):
        self.device = qml.device('default.qubit', wires=1)
        # Randomize initial state
        self.current_state = random.choice(['|0⟩', '|1⟩'])
        self.player_scores = [0, 0]
        self.current_player = 0
        self.game_states = ['|0⟩', '|1⟩', '|+⟩', '|-⟩', '|i⟩', '|-i⟩']

        self.deck = self.create_deck()
        random.shuffle(self.deck)

        self.player_hands = [[], []]
        self.deal_cards()

    def create_deck(self):
        cards = {
            'I': 3, 'X': 4, 'Y': 9, 'Z': 7,
            'H': 7, 'S': 7, '√X': 12, 'Measurement': 3
        }

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
        # Determine the winner based on scores
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
                qml.S(wires=0)
            elif state_name == '|-i⟩':
                qml.Hadamard(wires=0)
                qml.adjoint(qml.S)(wires=0)
            return qml.state()

        return prepare_state()

    def apply_gate_penrylane(self, current_state, gate):
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
            self.current_state = self.apply_gate_penrylane(old_state, card)
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
        self.figure = Figure(figsize=(5, 5), dpi=100)
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

        # Highlight endzones
        self.ax.scatter(1, 0, 0, color='purple', s=300, alpha=0.2)
        self.ax.scatter(-1, 0, 0, color='green', s=300, alpha=0.2)

        # Endzone labels
        self.ax.text(1.3, 0, .4, 'P2 Endzone\n( P1 score here)', color='purple', fontsize=10, ha='center')
        self.ax.text(-1.3, 0, .4, 'P1 Endzone\n(P2 score here)', color='green', fontsize=10, ha='center')

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


class QubitTouchdownGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Qubit Touchdown")
        self.root.geometry("1200x800")

        self.game = QubitTouchdown()

        self.setup_gui()
        self.update_display()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Game info and controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        title_label = ttk.Label(left_frame, text="Qubit Touchdown",
                                font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Game instructions
        instructions = ttk.Label(left_frame,
                                 text="Score by moving the ball into your OPPONENT'S endzone.\n"
                                      "Player 1 scores in |+⟩ (purple)\n"
                                      "Player 2 scores in |-⟩ (green)",
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

        # Game log
        log_label = ttk.Label(left_frame, text="Game Log:", font=("Arial", 11, "bold"))
        log_label.pack(anchor=tk.W, pady=(10, 5))

        self.log_text = tk.Text(left_frame, height=15, width=40, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Scrollbar for log
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # 3D Bloch sphere
        self.bloch_3d = BlochSphere3D(right_frame)
        self.bloch_3d.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        legend_frame = ttk.Frame(right_frame)
        legend_frame.pack(fill=tk.X, pady=10)

        states_info = [
            ("|0⟩", "North Pole - Start Position"),
            ("|1⟩", "South Pole - Start Position"),
            ("|+⟩", "Player 2's Endzone (Purple) - Player 1 scores here"),
            ("|-⟩", "Player 1's Endzone (Green) - Player 2 scores here"),
            ("|i⟩", "+Y Axis"),
            ("|-i⟩", "-Y Axis")
        ]

        for i, (state, desc) in enumerate(states_info):
            lbl = ttk.Label(legend_frame, text=f"{state}: {desc}", font=("Arial", 9))
            lbl.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)

    def update_display(self):
        state = self.game.get_game_state()

        # Update labels
        self.score_label.config(
            text=f"Scores: Player 1: {state['player_scores'][0]} | Player 2: {state['player_scores'][1]}")
        self.state_label.config(text=f"Current State: {state['current_state']}")
        self.player_label.config(
            text=f"Current Player: Player {state['current_player'] + 1}")
        self.cards_label.config(text=f"Cards remaining: {state['cards_remaining']}")

        # Update 3D Bloch sphere
        self.bloch_3d.plot_sphere(state['current_state'], state['bloch_coords'])

        # Update hand buttons
        for widget in self.hand_frame.winfo_children():
            widget.destroy()

        hand = state['player_hands'][state['current_player']]
        for i, card in enumerate(hand):
            btn = ttk.Button(self.hand_frame, text=card,
                             command=lambda idx=i: self.play_card(idx))
            btn.grid(row=0, column=i, padx=2, pady=2)

        # Check for game over and show popup if needed
        if state['is_game_over']:
            self.show_game_over_popup(state['winner'])

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
        tk.messagebox.showinfo("Game Over!", message)

    def play_card(self, card_index):
        result, touchdown, scoring_player = self.game.play_card(card_index)

        # Add to log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"\n{result}")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

        self.update_display()

        if touchdown and scoring_player is not None:
            tk.messagebox.showinfo("Touchdown!", f"Player {scoring_player} scored!")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    game_gui = QubitTouchdownGUI()
    game_gui.run()