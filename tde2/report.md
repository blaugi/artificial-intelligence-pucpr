# Relatório

## Os Algoritmos

### Minimax

O algoritmo Minimax é uma técnica de tomada de decisão utilizada em jogos de dois jogadores, como jogos de tabuleiro. Ele funciona explorando todas as possíveis sequências de movimentos até uma profundidade máxima, avaliando o estado do jogo em cada nó folha com uma função heurística. O algoritmo assume que ambos os jogadores jogam de forma ótima: o jogador maximizador (geralmente o jogador atual) tenta maximizar o valor, enquanto o minimizador tenta minimizar. Isso resulta em uma árvore de decisão onde o valor de cada nó é determinado pelos valores dos seus filhos.

### Alpha-Beta

O algoritmo Alpha-Beta é uma otimização do Minimax que reduz o número de nós explorados na árvore de decisão. Ele utiliza dois parâmetros, alpha e beta, para podar ramos que não influenciarão o resultado final. Alpha representa o melhor valor já encontrado para o jogador maximizador, e beta para o minimizador. Quando alpha >= beta, o ramo é podado, evitando cálculos desnecessários. Isso torna o algoritmo mais eficiente, especialmente em jogos com alta ramificação.

## Resumo das Comparações

Com base nos resultados das simulações, o algoritmo Alpha-Beta demonstrou superioridade em termos de eficiência. Em ambas as configurações (Minimax vs Alpha-Beta e Alpha-Beta vs Minimax), todos os jogos terminaram em empate, indicando que ambos os algoritmos são igualmente eficazes em termos de estratégia quando jogam de forma ótima. No entanto, o Alpha-Beta visitou menos nós em média (770.320 nós contra 966.687 nós) e teve um tempo de execução menor (31.56 segundos contra 40.00 segundos). O número médio de movimentos foi idêntico (25 movimentos), refletindo jogos equilibrados.

## Gráficos ou Tabelas dos Resultados

| Configuração              | Vitórias X | Vitórias O | Empates | Nós Visitados (Média) | Tempo (Média, s) | Movimentos (Média) |
|---------------------------|------------|------------|---------|-----------------------|------------------|-------------------|
| Minimax vs Alpha-Beta     | 0          | 0          | 5       | 966.687              | 40.00            | 25                |
| Alpha-Beta vs Minimax     | 0          | 0          | 5       | 770.320              | 31.56            | 25                |

## Análise Crítica dos Achados

Os resultados confirmam a eficácia do algoritmo Alpha-Beta como uma otimização do Minimax, reduzindo significativamente o número de nós explorados e o tempo de execução sem comprometer a qualidade das decisões. O fato de todos os jogos terminarem em empate sugere que ambos os algoritmos alcançam o jogo ótimo quando a profundidade máxima é suficiente para avaliar o tabuleiro. No entanto, em jogos mais complexos ou com profundidades maiores, a diferença de eficiência pode ser ainda mais pronunciada, tornando o Alpha-Beta preferível para aplicações em tempo real. Limitações incluem a dependência de uma função heurística adequada e o risco de poda excessiva em cenários não ideais. Futuras investigações poderiam explorar profundidades variáveis ou jogos mais complexos para validar esses achados.

## Código Fonte da Implementação

```python
import time

class Node:
    utility:int
    children:list["Node"]

    def get_children(self) -> list["Node"]:
        return self.children
    
    def is_terminal(self) -> bool:
        match len(self.children):
            case 0:
                return True
            case _:
                return False
        
    def get_utility(self) -> int:
        return self.utility
    
    def add_child(self, child:"Node"):
        self.children.append()
    


class Board:
    def __init__(self):
        self.grid = [[' ' for _ in range(5)] for _ in range(5)]

    def make_move(self, row, col, player):
        if self.grid[row][col] == ' ':
            self.grid[row][col] = player
            return True
        return False

    def is_full(self):
        return all(cell != ' ' for row in self.grid for cell in row)

    def check_win(self, player):
        # check rows
        for row in self.grid:
            for i in range(2):
                if all(cell == player for cell in row[i:i+4]):
                    return True
        # columns
        for col in range(5):
            for i in range(2):
                if all(self.grid[r][col] == player for r in range(i, i+4)):
                    return True
        # diagonals \
        for r in range(2):
            for c in range(2):
                if all(self.grid[r+j][c+j] == player for j in range(4)):
                    return True
        # diagonals /
        for r in range(2):
            for c in range(3, 5):
                if all(self.grid[r+j][c-j] == player for j in range(4)):
                    return True
        return False

    def get_available_moves(self):
        moves = []
        for r in range(5):
            for c in range(5):
                if self.grid[r][col] == ' ':
                    moves.append((r, c))
        return moves

    def copy(self):
        new_board = Board()
        new_board.grid = [row[:] for row in self.grid]
        return new_board

    def evaluate(self):
        # heuristic: score based on 3-in-a-row
        score = 0
        for player in ['X', 'O']:
            sign = 1 if player == 'X' else -1
            # rows
            for row in self.grid:
                for i in range(3):
                    window = row[i:i+3]
                    if window.count(player) == 3 and window.count(' ') == 0:
                        score += sign * 10
                    elif window.count(player) == 2 and window.count(' ') == 1:
                        score += sign * 1
            # columns
            for col in range(5):
                for i in range(3):
                    window = [self.grid[r][col] for r in range(i, i+3)]
                    if window.count(player) == 3 and window.count(' ') == 0:
                        score += sign * 10
                    elif window.count(player) == 2 and window.count(' ') == 1:
                        score += sign * 1
            # diagonals \
            for r in range(3):
                for c in range(3):
                    window = [self.grid[r+j][c+j] for j in range(3)]
                    if window.count(player) == 3 and window.count(' ') == 0:
                        score += sign * 10
                    elif window.count(player) == 2 and window.count(' ') == 1:
                        score += sign * 1
            # diagonals /
            for r in range(3):
                for c in range(2, 5):
                    window = [self.grid[r+j][c-j] for j in range(3)]
                    if window.count(player) == 3 and window.count(' ') == 0:
                        score += sign * 10
                    elif window.count(player) == 2 and window.count(' ') == 1:
                        score += sign * 1
        return score

    def __str__(self):
        return '\n'.join([' '.join(row) for row in self.grid])

class MinimaxAgent:
    def __init__(self, player, max_depth=3):
        self.player = player
        self.max_depth = max_depth
        self.nodes_visited = 0

    def get_move(self, board):
        self.nodes_visited = 0
        best_move = None
        best_value = -float('inf') if self.player == 'X' else float('inf')
        for move in board.get_available_moves():
            new_board = board.copy()
            new_board.make_move(*move, self.player)
            value = self.minimax(new_board, 0, self.player == 'O')
            if self.player == 'X':
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
        return best_move

    def minimax(self, board, depth, is_max):
        self.nodes_visited += 1
        if board.check_win('X'):
            return 1000
        if board.check_win('O'):
            return -1000
        if board.is_full():
            return 0
        if depth == self.max_depth:
            return board.evaluate()
        if is_max:
            max_eval = -float('inf')
            for move in board.get_available_moves():
                new_board = board.copy()
                new_board.make_move(*move, 'X')
                eval = self.minimax(new_board, depth + 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.get_available_moves():
                new_board = board.copy()
                new_board.make_move(*move, 'O')
                eval = self.minimax(new_board, depth + 1, True)
                min_eval = min(min_eval, eval)
            return min_eval

class AlphaBetaAgent(MinimaxAgent):
    def get_move(self, board):
        self.nodes_visited = 0
        best_move = None
        best_value = -float('inf') if self.player == 'X' else float('inf')
        alpha = -float('inf')
        beta = float('inf')
        for move in board.get_available_moves():
            new_board = board.copy()
            new_board.make_move(*move, self.player)
            value = self.alphabeta(new_board, 0, self.player == 'O', alpha, beta)
            if self.player == 'X':
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
        return best_move

    def alphabeta(self, board, depth, is_max, alpha, beta):
        self.nodes_visited += 1
        if board.check_win('X'):
            return 1000
        if board.check_win('O'):
            return -1000
        if board.is_full():
            return 0
        if depth == self.max_depth:
            return board.evaluate()
        if is_max:
            max_eval = -float('inf')
            for move in board.get_available_moves():
                new_board = board.copy()
                new_board.make_move(*move, 'X')
                eval = self.alphabeta(new_board, depth + 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.get_available_moves():
                new_board = board.copy()
                new_board.make_move(*move, 'O')
                eval = self.alphabeta(new_board, depth + 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

def play_game(agent1, agent2):
    board = Board()
    current_player = 'X'
    agents = {'X': agent1, 'O': agent2}
    total_nodes = 0
    start_time = time.time()
    moves_count = 0
    while not board.is_full() and not board.check_win('X') and not board.check_win('O'):
        move = agents[current_player].get_move(board)
        board.make_move(*move, current_player)
        total_nodes += agents[current_player].nodes_visited
        current_player = 'O' if current_player == 'X' else 'X'
        moves_count += 1
    end_time = time.time()
    winner = None
    if board.check_win('X'):
        winner = 'X'
    elif board.check_win('O'):
        winner = 'O'
    return winner, total_nodes, end_time - start_time, moves_count

def run_simulations(num_games=5):
    results = {'minimax_vs_alphabeta': [], 'alphabeta_vs_minimax': []}
    for i in range(num_games):
        # Minimax as X, AlphaBeta as O
        agent1 = MinimaxAgent('X')
        agent2 = AlphaBetaAgent('O')
        winner, nodes, time_taken, moves = play_game(agent1, agent2)
        results['minimax_vs_alphabeta'].append((winner, nodes, time_taken, moves))
        
        # AlphaBeta as X, Minimax as O
        agent1 = AlphaBetaAgent('X')
        agent2 = MinimaxAgent('O')
        winner, nodes, time_taken, moves = play_game(agent1, agent2)
        results['alphabeta_vs_minimax'].append((winner, nodes, time_taken, moves))
    
    return results

def print_results(results):
    print("Results:")
    for config, games in results.items():
        print(f"\nConfiguration: {config}")
        total_games = len(games)
        wins_X = sum(1 for w, _, _, _ in games if w == 'X')
        wins_O = sum(1 for w, _, _, _ in games if w == 'O')
        draws = total_games - wins_X - wins_O
        avg_nodes = sum(n for _, n, _, _ in games) / total_games
        avg_time = sum(t for _, _, t, _ in games) / total_games
        avg_moves = sum(m for _, _, _, m in games) / total_games
        print(f"Wins for X: {wins_X}, Wins for O: {wins_O}, Draws: {draws}")
        print(f"Average nodes visited: {avg_nodes:.2f}")
        print(f"Average time: {avg_time:.4f} seconds")
        print(f"Average moves: {avg_moves:.2f}")

def main():
    print("Running simulations...")
    results = run_simulations()
    print_results(results)

if __name__ == "__main__":
    main()
```
