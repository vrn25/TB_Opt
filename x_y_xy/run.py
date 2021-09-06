import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from optimizers_xy import linear_linear, bilinear_bilinear, linear_bilinear, bilinear_linear

class policy(nn.Module):
    def __init__(self, init_val):
        super(policy, self).__init__()
        self.move = nn.Parameter(init_val)

    def forward(self):
        return self.move
"""
Python function for x^m + x^n + x^p y^q
seed: seed for experiment
lr1: player 1 learning rate
lr2: player 2 learning rate
batch_size: batch size used
train_rounds: number of iterations for which game is played and params updated
level1: player 1's level
level2: player 2's level
game: a list containing powers of x and y | example - game = [1,2,3,4] means player 1 maximizes x + y2 + x3y4 and player 2 minimizes x + y2 + x3y4
game = [None, 1, 2, 4] meas player 1 maximizes y + x2y4 and player 2 minimizes y + x2y4
plot_figures: boolean - whether to plot figures or not
p1_first: True if player 1 plays first, false if player 2 plays first
"""
def run_game(seed_idx, lr1, lr2, train_rounds, level1, level2, game, batch_size=1, plot_figures=False, p1_first=True, print_vals=True, min_max=None, optimizer='LL', exp=None):

    lookback = 10
    var_threshold = 1e-8
    converged_rounds = 0
    converged = False
    
    # curr_x and curr_y are the values initialized for x and y
    curr_x, curr_y = torch.tensor([[init_val[0]]]), torch.tensor([[init_val[1]]])
    init_x = curr_x[:,0].item()
    init_y = curr_y[:,0].item()
    if print_vals:
        print('init_x = {}, init_y = {}\n'.format(init_x, init_y))
        print('seed={}, lr1={}, lr2={}, batch_size={}, train_rounds={}, level1={}, level2={}, game={}\n'.format(seeds[seed_idx], lr1, lr2, batch_size, train_rounds, level1, level2, game))

    p1 = policy(curr_x)
    p2 = policy(curr_y)
    #for p in p1.parameters():
    #    print(p)
    #for p in p2.parameters():
    #    print(p)

    if optimizer == 'L-L':
        optim1 = linear_linear(p1.parameters(), p2.parameters(), lr1=lr1, lr2=lr2, collect_info=True, exp=exp)
        optim2 = linear_linear(p2.parameters(), p1.parameters(), lr1=lr2, lr2=lr1, collect_info=True, exp=exp)
    elif optimizer == 'BL-BL':
        optim1 = bilinear_bilinear(p1.parameters(), p2.parameters(), lr1=lr1, lr2=lr2, collect_info=True, exp=exp)
        optim2 = bilinear_bilinear(p2.parameters(), p1.parameters(), lr1=lr2, lr2=lr1, collect_info=True, exp=exp)
    elif optimizer == 'L-BL':
        optim1 = linear_bilinear(p1.parameters(), p2.parameters(), lr1=lr1, lr2=lr2, collect_info=True, exp=exp)
        optim2 = linear_bilinear(p2.parameters(), p1.parameters(), lr1=lr2, lr2=lr1, collect_info=True, exp=exp)
    elif optimizer == 'BL-L':
        optim1 = bilinear_linear(p1.parameters(), p2.parameters(), lr1=lr1, lr2=lr2, collect_info=True, exp=exp)
        optim2 = bilinear_linear(p2.parameters(), p1.parameters(), lr1=lr2, lr2=lr1, collect_info=True, exp=exp)
    else:
        raise Exception('Unknown optimizer!')

    level_dict1 = {0: optim1.level0_step, 1: optim1.level1_step, 2: optim1.level2_step, 3: optim1.level3_step}
    level_dict2 = {0: optim2.level0_step, 1: optim2.level1_step, 2: optim2.level2_step, 3: optim2.level3_step}

    def construct_objective(curr_x, curr_y):
        objective = torch.zeros_like(curr_x)
        objective += pow(curr_x, game[0]) if game[0] is not None else 0
        objective += pow(curr_y, game[1]) if game[1] is not None else 0
        objective += pow(curr_x, game[2])*pow(curr_y, game[3]) if game[2] is not None and game[3] is not None else 0
        objective = objective.mean()
        return objective

    x_values = []
    y_values = []
    game_values = []
    x_values.append(curr_x.mean(dim=0)[0].item())
    y_values.append(curr_y.mean(dim=0)[0].item())
    game_values.append(construct_objective(curr_x, curr_y).detach().item())
    
    curr_x, curr_y = p1(), p2()

    norms_p1, norms_p2 = [], []
    #a, b, c = 1, 1, 1
    
    i=1 if p1_first else 0
    for round in range(train_rounds):
        if p1_first:
            eta1 = construct_objective(curr_x, curr_y)
            eta2 = -construct_objective(curr_x, curr_y)
            optim1.zero_grad()
            level_dict1[level1](eta1, eta2)
            norms_p1.append(optim1.get_info())
            curr_x = p1()

            eta1 = construct_objective(curr_x, curr_y)
            eta2 = -construct_objective(curr_x, curr_y)
            optim2.zero_grad()
            level_dict2[level2](eta2, eta1)
            norms_p2.append(optim2.get_info())
            curr_y = p2()
        else:
            eta1 = construct_objective(curr_x, curr_y)
            eta2 = -construct_objective(curr_x, curr_y)
            optim2.zero_grad()
            level_dict2[level2](eta2, eta1)
            norms_p2.append(optim2.get_info())
            curr_y = p2()

            eta1 = construct_objective(curr_x, curr_y)
            eta2 = -construct_objective(curr_x, curr_y)
            optim1.zero_grad()
            level_dict1[level1](eta1, eta2)
            norms_p1.append(optim1.get_info())
            curr_x = p1()

        x = p1()
        y = p2()
        x_values.append(x.detach().mean(dim=0)[0].item())
        y_values.append(y.detach().mean(dim=0)[0].item())
        game_values.append(construct_objective(x, y).detach().item())

        if not converged and round > lookback:
            if np.var(np.array(x_values)[-lookback:]) < var_threshold and np.var(np.array(y_values)[-lookback:]) < var_threshold:
                converged_rounds = round
                converged = True

    if plot_figures:
        #print('X = {}, Y = {}'.format(x_values[-1], y_values[-1]))
        pass
        #plot_curve(x_values, y_values, lr1, lr2, level1, level2, game_values)
    if print_vals:
        print('final_x = {}, final_y = {}\n'.format(x_values[-1], y_values[-1]))
        print('##########################################################################################################')
    return x_values, y_values, norms_p1, norms_p2, converged_rounds-lookback

def plot_figures(level1, level2, lr1, lr2, x, y, case='LL'):

    plt.subplot(4, 4, level1*len(levels1) + level2 + 1)
    plt.plot(np.array(x), label='X')
    plt.plot(np.array(y), label='Y')
    #plt.plot(x[0], y[0], '*', color='r', label='Start point')
    #plt.plot(x[-1], y[-1], '*', color='g', label='End point')
    #plt.plot(np.array(y), label='Y')
    #plt.plot(np.array(x)+np.array(y), label='X+Y')
    plt.ylabel('X,Y values')
    plt.xlabel('Rounds')
    #plt.xlabel('X values')
    plt.legend(loc='upper right')
    precision=5
    plt.title('X={}|Y={}'.format(round(x[-1],precision), round(y[-1],precision)))
    plt.grid()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)

#lrs = [0.1]
save_fig=False#True
save_xys=False#True

seeds=[99, 7, 87, 16, 11] # generated randomly
print(seeds)
c_rounds_list = []
for i in range(len(seeds)):
    torch.manual_seed(seeds[i])
    use_seed=True

    if use_seed:
        #n = -1/pow(2,1/3)
        #r=1
        #x = np.random.uniform(n-r,n+r,1)[0]
        #y = np.sqrt(r**2 - (x-n)**2) + n
        mean, std = 0., 1.#-1/pow(2,1/3), 0.3
        init_val = [torch.normal(mean, std, (1,)).item(), torch.normal(mean, std, (1,)).item()]
        #init_val = [torch.randn(1).item(), torch.randn(1).item()]
    else:
        init_val=[-0.7,-0.7]
    print(init_val)

    levels1 = [0, 1, 2, 3]
    levels2 = [0, 1, 2, 3]
    lr1, lr2 = 0.1, 0.1
    c_rounds = []
    #print('\033[1m LEARNING RATE = {}'.format(lr))
    plt.figure(figsize=(40,30))
    plt.suptitle('Seed={} ({}) | L-L'.format(seeds[i], init_val))
    for level1 in levels1:
        for level2 in levels2:
            case, exp = 'L-L', 2
            x, y, n1, n2, cr = run_game(seed_idx=i, lr1=lr1, lr2=lr2, train_rounds=2000, level1=level1, level2=level2, game=[1,1,1,1], plot_figures=False, print_vals=True, p1_first=True, optimizer=case, exp=exp)
            c_rounds.append(cr)
            plot_figures(level1, level2, lr1, lr2, x, y, case)
            if save_xys:
                np.savez('plots/xys/seed{}/T{}{}_x.npz'.format(seeds[i], level1, level2), x)
                np.savez('plots/xys/seed{}/T{}{}_y.npz'.format(seeds[i], level1, level2), y)
    if save_fig:
        plt.savefig('plots/Linear_Linear/seed{}.png'.format(seeds[i]))
    c_rounds_list.append(c_rounds)

for i in range(len(c_rounds_list)):
    print(np.array(c_rounds_list[i]).reshape(4,4))

a = np.array(c_rounds_list).reshape(len(seeds),4,4)
print(np.mean(a, axis=0))

plt.show()