import math

opps = {
    'A': {'bull': 0.12, 'base': 0.06, 'bear': -0.08, 'horizon': 2},
    'B': {'bull': 0.30, 'base': 0.15, 'bear': -0.20, 'horizon': 1},
    'C': {'bull': 0.06, 'base': 0.04, 'bear': 0.00, 'horizon': 3}
}

probs = [
    {'bull': 0.2, 'base': 0.6, 'bear': 0.2, 'name': 'P1 (Neutral)'},
    {'bull': 0.1, 'base': 0.5, 'bear': 0.4, 'name': 'P2 (Bearish)'},
    {'bull': 0.3, 'base': 0.4, 'bear': 0.3, 'name': 'P3 (Volatile)'}
]

print(f"{'Opp':<4} | {'Prob':<15} | {'Exp Ret':<8} | {'Vol':<8} | {'Ann Ret':<8}")
print("-" * 55)

for p in probs:
    for name, data in opps.items():
        er = p['bull']*data['bull'] + p['base']*data['base'] + p['bear']*data['bear']
        var = p['bull']*(data['bull'] - er)**2 + p['base']*(data['base'] - er)**2 + p['bear']*(data['bear'] - er)**2
        vol = math.sqrt(var)
        ann_ret = (1 + er)**(1/data['horizon']) - 1
        print(f"{name:<4} | {p['name']:<15} | {er:>8.2%} | {vol:>8.2%} | {ann_ret:>8.2%}")
