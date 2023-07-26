import chess.pgn
import chess
import copy
import tensorflow as tf
from tensorflow import keras
import numpy as np

class DeepChessEngine:
    def __init__(self, depth):
        self.__depth = depth
        self.__move = None
        self.siamese_model = tf.keras.models.load_model("siamese_final_500000.h5")
    
    def __fen2bitstring(self, fen):
        board = chess.Board(fen)
        bitboard = np.zeros(64*6*2+5)

        piece_idx = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}

        for i in range(64):
            if board.piece_at(i):
                color = int(board.piece_at(i).color) + 1
                bitboard[(piece_idx[board.piece_at(i).symbol().lower()] + i * 6) * color] = 1

        bitboard[-1] = int(board.turn)
        bitboard[-2] = int(board.has_kingside_castling_rights(True))
        bitboard[-3] = int(board.has_kingside_castling_rights(False))
        bitboard[-4] = int(board.has_queenside_castling_rights(True))
        bitboard[-5] = int(board.has_queenside_castling_rights(False))

        return bitboard

    def preprocess_for_siamese_input(self, fen1, fen2):
        bitstring1 = np.array([self.__fen2bitstring(fen1)])
        bitstring2 = np.array([self.__fen2bitstring(fen2)])
        siamese_input = [bitstring1, bitstring2]
        return siamese_input

    def game_predict(self, game1, game2):
        fen1 = game1.fen()
        fen2 = game2.fen()
        fin = self.preprocess_for_siamese_input(fen1, fen2)
        pred = self.siamese_model.predict(fin)
        if game1.is_checkmate():
            return (game1, game2)
        elif game2.is_checkmate():
            return (game2, game1)
        elif pred[0][0] > pred[0][1]:
            return (game1, game2)
        elif pred[0][0] <= pred[0][1]:
            return (game2, game1)
    
    def is_game_over(self, board):
        return (
            board.is_checkmate()
            or board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
            or board.is_variant_draw()
        )


    def alphabeta(self, pos, depth, alpha, beta, maximizingPlayer):
        if depth == 0:
            return pos
        if maximizingPlayer:
            ini = -1
            for move in pos.legal_moves:
                curr = copy.copy(pos)
                curr.push(move)
                if ini == -1:
                    ini = self.alphabeta(curr, depth-1, alpha, beta, False) 
                if alpha == -1:
                    alpha = ini

                ini = self.game_predict(ini, self.alphabeta(curr, depth-1, alpha, beta, False))[0]
                alpha = self.game_predict(alpha, ini)[0] 
                if beta != 1:
                    if self.game_predict(alpha, beta)[0] == alpha:
                        break
            return ini 
        else:
            ini = 1
            for move in pos.legal_moves:
                curr = copy.copy(pos)
                curr.push(move)
                if ini == 1:
                    ini = self.alphabeta(curr, depth-1, alpha, beta, True) 
                if beta == 1:
                    beta = ini

                ini = self.game_predict(ini, self.alphabeta(curr, depth-1, alpha, beta, True))[1]
                beta = self.game_predict(beta, ini)[1] 
                if alpha != -1:
                    if self.game_predict(alpha, beta)[0] == alpha:
                        break
            return ini 


    def computerMove(self, board, depth):
        try:
            alpha = -1
            beta = 1
            ini = -1
            bestMove = None
            for move in board.legal_moves:
                curr = copy.copy(board)
                curr.push(move)
                if ini == -1:
                    ini = self.alphabeta(curr, depth-1, alpha, beta, False)
                    bestMove = move
                    if alpha == -1:
                        alpha = ini
                else:
                    new_ini = self.alphabeta(curr, depth-1, alpha, beta, False)
                    pred_ini = self.game_predict(new_ini, ini)[0]
                    if pred_ini != ini:
                        bestMove = move
                        ini = pred_ini
                    alpha = self.game_predict(alpha, ini)[0]

            print(bestMove)
            board.push(bestMove)
            return board
        except AttributeError:
            depth = 1
            return self.computerMove(board, depth)


    def playerMove(self, board):
        while True:
            try:
                move = input("Enter your move \n")
                board.push_san(move)
                break
            except ValueError:
                print("Illegal move, please try again")

        return board

    def playGame(self):
        moveTotal = 0;
        board = chess.Board()
        depth = input("Enter search depth \n")
        depth = int(depth)
        while board.is_game_over() == False:
            print(board)
            if moveTotal % 2 == 1:
                board = self.playerMove(board)
            else:
                board =	self.computerMove(board, depth)
            moveTotal = moveTotal+1

        print(board)
        print("Game is over")
