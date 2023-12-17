from collections import OrderedDict
def change_pd(c: int, l_coins: List[int]) -> np.ndarray:
    
    arr = np.full((len(l_coins) + 1, c + 1), np.inf)
    
    
    arr[:, 0] = 0
    
    
    for i in range(1, len(l_coins) + 1):
        for j in range(1, c + 1):
            
            if l_coins[i - 1] <= j:
                
                arr[i][j] = min(arr[i - 1][j], 1 + arr[i][j - l_coins[i - 1]])
            else:
                
                arr[i][j] = arr[i - 1][j]
    
    return arr





def optimal_change_pd(c: int, l_coins: List[int])-> Dict:

    arr = change_pd(c, l_coins)
    sol = int(arr[-1,-1])
    monedas = {elemento: 0 for elemento in l_coins}

    for _ in range(0, sol):
        for i in range(1, len(arr[:,-1])):
            if arr[-i,-1] < min(arr[:-i,-1]):
                monedas[l_coins[-i]] += 1
                c = c-l_coins[-i]
                arr = change_pd(c, l_coins)
                break
    
    return dict(OrderedDict(sorted(monedas.items())))

optimal_change_pd(3, [1,2,4])
