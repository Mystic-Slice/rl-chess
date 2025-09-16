from rewards import format_reward_sample
response = """<reasoning>
The position shows a typical endgame with a king and pawn structures. The black king is on e8, and white has pawns on the queenside and center. The key threat is the white pawn on d6 and e5, which can advance to promote. However, 
black's knight on d5 controls d4 and e4, and the king is near the center. Since white has a passed pawn on e5 and black has no immediate reply, the best action is to support the e5 pawn with a push to e6, threatening to push to e7
 and e8. But white's king is on d1, and the knight on d5 is a strong defender. The most immediate and safe move is to advance the pawn to e6, gaining space and pressure.  
</reasoning>
<best_move>e5xe6</best_move><|im_end|>"""

print(format_reward_sample(response))