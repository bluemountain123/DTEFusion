import os
import time
import argparse
import matplotlib.pyplot as plt
from typing import Callable, Iterator
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from metric_new import calc_EN, calc_MS_SSIM
from config import TrainConfig, load_config, load_dataset, load_model, load_optimizer
from dataset import parse_directory, Tensor_to_PIL, PIL_to_Tensor
from models.models2 import DBTFuse1, FuseInput, FuseOutput
from models.loss import ALL_Loss



def parse_commandline():
    parser = argparse.ArgumentParser(description='train/test/evolve model')
    parser.add_argument('--train', action='store_true', default=False, help='train model (normal full training)')
    parser.add_argument('--test', action='store_true', default=False, help='test model')
    parser.add_argument('--evolve', action='store_true', default=False, help='run small-scale training + WOA evolve hyperparams, then do final training')
    return parser.parse_args()


def timer(fn: Callable):
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = fn(*args, **kwargs)
        used_time = round(time.time() - start_time)
        minutes, seconds = divmod(used_time, 60)
        print(f'{fn.__name__} completed in {minutes}m {seconds}s')
        return ret
    return wrapper

#plot the images of loss curves
def plot_loss(train_losses, valid_losses, save_path):

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(valid_losses, label="Valid Loss", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_convergence(best_fitness_per_gen, save_path):

    plt.figure(figsize=(12, 8))
    generations = range(1, len(best_fitness_per_gen) + 1)
    plt.plot(generations, best_fitness_per_gen, marker='o', linestyle='-', color='green', label='Best Fitness')
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Validation Loss - EN - MS_SSIM)")
    plt.title("PSO Convergence Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Convergence plot saved to {save_path}")


############################
# 
############################

HYP_SPACE = {
    'a': (1,10),    
    'b': (1,10),     #
    'c': (1,10),
    'd': (1,10),
}


############################
# PSO 
############################
class PSO:
    def __init__(self, opt_func, constraints, nsols, w=0.5, c1=1.5, c2=1.5, maximize=False):
        """
        Parameters:
            opt_func (callable): The objective function to be optimized (minimized or maximized).
            constraints (list of tuples): A list of [(min, max), ...] tuples defining the boundaries for each hyperparameter.
            nsols (int): Population size (number of particles or solutions).
            w (float): Inertia weight.
            c1 (float): Cognitive coefficient (personal learning factor).
            c2 (float): Social coefficient (global learning factor).
            maximize (bool): If True, perform maximization; otherwise, perform minimization.
        """
        self._opt_func = opt_func
        self._constraints = constraints
        self._dim = len(constraints)  
        self._sols = self._init_solutions(nsols)
        self._w = w
        self._c1 = c1
        self._c2 = c2
        self._maximize = maximize
        self._best_solutions = []

    def _init_solutions(self, nsols):
        """Initialize particle positions and velocities."""
        sols = np.array([  
            np.random.uniform(low, high, nsols)
            for (low, high) in self._constraints
        ]).T  # (nsols, dim) 
        velocities = np.zeros_like(sols)  
        return np.hstack((sols, velocities))  

    def get_best_solution(self):
        """Return the best solution."""
        if not self._best_solutions:
            return None
        return self._best_solutions[-1]

    def optimize(self):
        fitness = self._opt_func(self._sols[:, :-self._dim])   # Get (nsols,) array
        sol_positions = self._sols[:, :-self._dim]             # Only positions, not velocities
        sol_fitness   = list(zip(fitness, sol_positions))      # Each element is (fitness, [a,b,c,d])

        sol_fitness = sorted(sol_fitness, key=lambda x: x[0], reverse=self._maximize)
        self._best_solutions.append(sol_fitness[0])

        # Update particle velocities and positions
        for i in range(self._sols.shape[0]):
            print("Optimizing")
            r1 = np.random.rand(self._dim)
            r2 = np.random.rand(self._dim)

            # Update velocity
            self._sols[i, self._dim:] = (
                self._w * self._sols[i, self._dim:] +
                self._c1 * r1 * (self._best_solutions[-1][1] - self._sols[i, :-self._dim]) +
                self._c2 * r2 * (self._best_solutions[-1][1] - self._sols[i, :-self._dim])
            )

            # Update position
            self._sols[i, :-self._dim] += self._sols[i, self._dim:]
            # Constrain particle solutions
            self._sols[i, :-self._dim] = self._constrain_solution(self._sols[i, :-self._dim])

    def _constrain_solution(self, sol: np.ndarray) -> np.ndarray:
        """
        Force: Constrain hyperparameters to meet specified ranges.
        a + b = 10
        c + d = 10
        """
        # 1) a + b = 10
        sol[0] = np.clip(sol[0], 1, 9)
        sol[1] = 10 - sol[0]  # b

        # 2) c + d = 10
        sol[2] = np.clip(sol[2], 1, 9)
        sol[3] = 10 - sol[2]  # d

        return np.clip(sol, 1, 9)  # Limit each hyperparameter to [1, 10] range

    def print_best_solutions(self):
        """Print the best solution of each generation."""
        print('Generation Best Solution History:')
        print('([Fitness], [Solution])')
        for s in self._best_solutions:
            print(s)
        print('\nBest Solution:')
        print(self._best_solutions[-1])


############################
# 4. Training/Validation Main Flow Example
############################

@timer
def train_loop(model: DBTFuse1, criterion: nn.Module,
               trainloader: DataLoader, optimizer: Optimizer,
               accumulation_steps: int, device: torch.device) -> float:# Small-scale training, called by train and train_small for hyperparameter and network training
    train_loss = 0.0
    model.train()
    
    for step, (vi, ir) in enumerate(tqdm(trainloader, desc="Training", leave=False), 1):
        inputs = FuseInput(vi.to(device), ir.to(device))
        outputs: FuseOutput = model(inputs)
        loss: Tensor = criterion(inputs, outputs)
        train_loss += loss.item()

        loss = loss / accumulation_steps
        loss.backward()

        if (step % accumulation_steps) == 0 or (step == len(trainloader)):
            optimizer.step()
            optimizer.zero_grad()
    
    average_loss = train_loss / len(trainloader)
    print(f"\nCompleted Train Loop: Average Loss = {average_loss:.4f}")
    return average_loss


def normalize_metrics(metrics: list) -> list:
    """
    Perform Min-Max normalization on the given metrics to scale them to [0, 1] range.
    """
    min_val = min(metrics)
    max_val = max(metrics)
    return [(metric - min_val) / (max_val - min_val) for metric in metrics]

@timer
def valid_loop(model: DBTFuse1, criterion: nn.Module,
               validloader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    valid_loss = 0.0
    total_EN = 0.0
    total_MS_SSIM = 0.0
    count = 0
    model.eval()
    
    en_values = []  # Store original EN values
    ms_ssim_values = []  # Store original MS_SSIM values
    loss_values = []  # Store original valid_loss values

    with torch.no_grad():
        for step, (vi, ir) in enumerate(tqdm(validloader, desc="Validation", leave=False), 1):
            inputs = FuseInput(vi.to(device), ir.to(device))
            outputs: FuseOutput = model(inputs)
            loss: Tensor = criterion(inputs, outputs)
            valid_loss += loss.item()

            # Calculate new evaluation metrics
            fu = outputs.fusion  # Assume fusion is the output fused image
            EN = calc_EN(fu)
            MS_SSIM_val = calc_MS_SSIM(vi.to(device), ir.to(device), fu)

            # Save original EN, MS_SSIM and loss values
            en_values.append(EN)
            ms_ssim_values.append(MS_SSIM_val)
            loss_values.append(loss.item())

            total_EN += EN
            total_MS_SSIM += MS_SSIM_val
            count += 1

    # Perform normalization
    normalized_EN = normalize_metrics(en_values)
    normalized_MS_SSIM = normalize_metrics(ms_ssim_values)
    normalized_loss = normalize_metrics(loss_values)

    # Calculate normalized averages
    average_loss = sum(normalized_loss) / count
    average_EN = sum(normalized_EN) / count
    average_MS_SSIM = sum(normalized_MS_SSIM) / count

    print(f"\nCompleted Validation Loop: Normalized Average Loss = {average_loss:.4f}, "
          f"Normalized Average EN = {average_EN:.4f}, Normalized Average MS_SSIM = {average_MS_SSIM:.4f}")
    
    return average_loss, average_EN, average_MS_SSIM






def train(config: TrainConfig, device: torch.device, a: float, b: float, c: float, d: float, save_loss_plot_path: str):

    print(f"[TRAIN] Starting complete training with hyperparameters: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d}")

    model: DBTFuse1 = load_model(config.model, device)
    # print(f'Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # ALL_Loss supports passing lamda + betas
    criterion = ALL_Loss(a=a, b=b, c=c, d=d).to(device)

    trainloader, validloader = load_dataset(config.dataset)[:2]
    optim = load_optimizer(config.train.optimizer, model.parameters())
    lr_scheduler = StepLR(optim, step_size=50, gamma=0.5)
    accumulation_steps = config.train.accumulation_steps
    best_valid_fitness = float('inf')  # fitness = loss - (EN + MS_SSIM)

    print(f'Training <{model._get_name()}> on {device.type}')

    train_losses = []
    valid_fitness = []
    valid_losses = []  # Ensure this list exists
    for epoch in range(config.train.epochs):
        print(f"\nStarting Epoch {epoch+1}/{config.train.epochs}")
        train_loss = train_loop(model, criterion, trainloader, optim, accumulation_steps, device)
        train_losses.append(train_loss)

        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch+1}/{config.train.epochs}], LR={current_lr:.6f}, train_loss={train_loss:.4f}')

        valid_loss, valid_EN, valid_MS_SSIM = valid_loop(model, criterion, validloader, device)
        # Calculate comprehensive fitness
        fitness = valid_loss - (valid_EN + valid_MS_SSIM)
        valid_fitness.append(fitness)
        valid_losses.append(valid_loss)  # Add this line, append validation loss to the list
        print(f'Epoch [{epoch+1}/{config.train.epochs}], Fitness={fitness:.4f} '
              f'(loss={valid_loss:.4f} - EN={valid_EN:.4f} - MS_SSIM={valid_MS_SSIM:.4f}), '
              f'a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d}')

        # Save the best
        if fitness < best_valid_fitness:
            best_valid_fitness = fitness
            checkpoint(model, a, b, c, d, mode="evolve")
            print(f'  => New best fitness={fitness:.4f}, model saved!')

    # # Print training and validation losses
    # print("train_losses:", train_losses)
    # print("valid_losses:", valid_losses)  # Correct the print variable name here

    # Check and plot loss curves
    if len(train_losses) and len(valid_losses):
        last_epoch = config.train.epochs
        fig_path = os.path.join(save_loss_plot_path, f'loss_plot_epoch_{last_epoch}.png')
        plot_loss(train_losses, valid_losses, fig_path)
        print(f"Loss plot saved to {fig_path}")


############################
# 5. Small-Scale Training + Evolution Process
############################
@timer
def train_small(
    config: TrainConfig,
    device: torch.device,
    a: float,
    b: float,
    c: float,
    d: float,
    small_epochs: int = 3
) -> float:
    """
    Train for small_epochs epochs for quick evaluation during evolution search.
    Return comprehensive fitness (validation_loss - (EN + MS_SSIM)), the smaller the better.
    """
    model: DBTFuse1 = load_model(config.model, device)
    criterion = ALL_Loss(a=a, b=b, c=c, d=d).to(device)

    trainloader, validloader = load_dataset(config.dataset)[:2]
    optim = load_optimizer(config.train.optimizer, model.parameters())
    lr_scheduler = StepLR(optim, step_size=50, gamma=0.5)
    accumulation_steps = config.train.accumulation_steps

    for epoch in range(small_epochs):
        print(f"\nSmall-Train Epoch {epoch+1}/{small_epochs}")
        train_loop(model, criterion, trainloader, optim, accumulation_steps, device)
        lr_scheduler.step()

    valid_loss, valid_EN, valid_MS_SSIM = valid_loop(model, criterion, validloader, device)
    fitness = valid_EN + valid_MS_SSIM
    print(f"EN + MS_SSIM = {valid_EN + valid_MS_SSIM:.4f}")
    print(f"Small Training Completed. Validation Fitness: {fitness:.4f} (loss={valid_loss:.4f} - EN={valid_EN:.4f} - MS_SSIM={valid_MS_SSIM:.4f})")
    return fitness, valid_loss, valid_EN, valid_MS_SSIM


def objective_function(solutions, config, device):
    nsols = solutions.shape[0]
    fitness_arr = np.zeros(nsols, dtype=np.float64)

    for i in range(nsols):
        a, b, c, d = solutions[i]
        # ... small-scale training ...
        _, valid_loss, valid_EN, valid_MS_SSIM = train_small(
            config, device, a, b, c, d, small_epochs=2
        )
        # For example, maximize EN+MS_SSIM
        fitness_arr[i] = valid_EN + valid_MS_SSIM

    return fitness_arr
def evolve(config, device):
    """
    Use PSO for hyperparameter optimization, finally return (best_alpha, best_mu, best_lamda, best_betas).
    """
    constraints = [
        HYP_SPACE['a'],
        HYP_SPACE['b'],
        HYP_SPACE['c'],
        HYP_SPACE['d']
    ]

    pop_size = 50 # Increase population size to improve search effectiveness
    pso = PSO(
        opt_func = lambda sols: objective_function(sols, config, device),
        constraints = constraints,
        nsols = pop_size,
        w=0.5,
        c1=1.5,
        c2=1.5,
        maximize=True  # Now we want to maximize the sum of EN and MS_SSIM
    )
    n_iterations = 13 # Increase iterations to improve optimization effectiveness
    best_fitness_per_gen = []

    for gen in range(n_iterations):
        print(f"\n=== PSO_Generation {gen+1}/{n_iterations} ===")
        pso.optimize()
        best_solution = pso.get_best_solution()
        if best_solution:
            best_fitness, best_params = best_solution
            best_fitness_per_gen.append(best_fitness)
            print(f"  [Gen{gen+1}] Best Fitness={best_fitness:.4f}, Params={best_params}")
        else:
            print("  No solution found.")

    # Best solution
    final_best = pso.get_best_solution()
    if final_best is None:
        raise ValueError("No best solution found during PSO optimization.")
    
    best_fitness, best_params = final_best
    best_a  = best_params[0]
    best_b   = best_params[1]
    best_c  = best_params[2]
    best_d  =  best_params[3]

    print(f"\nEvolution finished. Best Fitness={best_fitness:.4f}, "
          f"a={best_a:.4f}, b={best_b:.4f}, c={best_c:.4f}, d={best_d}\n")

    # Plot convergence curve
    result_path = '/root/autodl-tmp/ESWAcode/result1'
    os.makedirs(result_path, exist_ok=True)
    plot_convergence(best_fitness_per_gen, os.path.join(result_path, 'PSO_convergence_curve.png'))

    return best_a, best_b, best_c, best_d
############################
# 6. Test Function
############################
def pad(vi: Tensor, ir: Tensor, patch_size: int):
    H, W = vi.shape[-2:]
    pad_h = 0 if (mod_h := H % patch_size) == 0 else patch_size - mod_h
    pad_w = 0 if (mod_w := W % patch_size) == 0 else patch_size - mod_w
    left, top = pad_w >> 1, pad_h >> 1
    right, down = pad_w - left, pad_h - top
    box = (left, right, top, down)
    vi = F.pad(vi, box, mode='reflect')
    ir = F.pad(ir, box, mode='reflect')
    return FuseInput(vi, ir), torch.Size((H, W))


def tensor_to_image(img: Tensor, size: torch.Size):
    H, W = size
    pad_h = img.size(-2) - H
    pad_w = img.size(-1) - W
    sh, sw = pad_h >> 1, pad_w >> 1
    cropped = img.squeeze(0)[:, sh:sh+H, sw:sw+W]
    return Tensor_to_PIL(cropped)


def test(config: TrainConfig, path: str, device: torch.device):
    save_path = Path(path)
    model: DBTFuse1 = load_model(config.model, device)
    model_name = config.model.pretrained.split('/', 1)[-1].replace('/', '_').removesuffix('.pth')
    save_path = save_path.with_name(save_path.name + '-Test').joinpath(model_name)
    print(f'{save_path=}')
    os.makedirs(save_path, exist_ok=True)
    model = model.eval()
    with torch.no_grad():
        for (vi, ir, filename) in load_data(path):
            inputs, size = pad(vi.to(device), ir.to(device), config.model.params['patch_size'])
            output: FuseOutput = model(inputs)

            image: Image.Image = tensor_to_image(output.fusion, size)

            save_file = os.path.join(save_path, filename)
            image.save(save_file)
            print(f'saved: {save_file}{" " * 20}', end='\r')


def load_data(dir: str) -> Iterator[tuple[Tensor, Tensor, str]]:
    if 'MSRS' in dir:
        msrs = Path(dir)
        with open(msrs.joinpath('labels.txt'), 'r') as fp:
            filenames = fp.read().splitlines()
            for filename in filenames:
                vi_file = msrs.joinpath('vi', filename)
                ir_file = msrs.joinpath('ir', filename)
                vi = PIL_to_Tensor(Image.open(vi_file).convert('L')).unsqueeze(0)  # type: ignore
                ir = PIL_to_Tensor(Image.open(ir_file).convert('L')).unsqueeze(0)  # type: ignore
                yield vi, ir, filename
    elif 'TNO' in dir:
        labels = parse_directory(dir)
        for item in labels:
            vi = PIL_to_Tensor(Image.open(item['vi']).convert('L')).unsqueeze(0)  # type: ignore
            ir = PIL_to_Tensor(Image.open(item['ir']).convert('L')).unsqueeze(0)  # type: ignore
            filename = os.path.basename(item['vi'])  # Use os.path.basename to get filename
            yield vi, ir, filename
    elif 'Road' in dir:
        road = Path(dir)
        with open(road.joinpath('labels.txt'), 'r') as fp:
            filenames = fp.read().splitlines()
            for filename in filenames:
                vi_file = road.joinpath('vi', filename)
                ir_file = road.joinpath('ir', filename)
                vi = PIL_to_Tensor(Image.open(vi_file).convert('L')).unsqueeze(0)  # type: ignore
                ir = PIL_to_Tensor(Image.open(ir_file).convert('L')).unsqueeze(0)  # type: ignore
                yield vi, ir, filename
    elif 'LLVIP' in dir:
        road = Path(dir)
        with open(road.joinpath('labels.txt'), 'r') as fp:
            filenames = fp.read().splitlines()
            for filename in filenames:
                vi_file = road.joinpath('vi', filename)
                ir_file = road.joinpath('ir', filename)
                vi = PIL_to_Tensor(Image.open(vi_file).convert('L')).unsqueeze(0)  # type: ignore
                ir = PIL_to_Tensor(Image.open(ir_file).convert('L')).unsqueeze(0)  # type: ignore
                yield vi, ir, filename
    elif 'M3FD' in dir:
        road = Path(dir)
        with open(road.joinpath('labels.txt'), 'r') as fp:
            filenames = fp.read().splitlines()
            for filename in filenames:
                vi_file = road.joinpath('vi', filename)
                ir_file = road.joinpath('ir', filename)
                vi = PIL_to_Tensor(Image.open(vi_file).convert('L')).unsqueeze(0)  # type: ignore
                ir = PIL_to_Tensor(Image.open(ir_file).convert('L')).unsqueeze(0)  # type: ignore
                yield vi, ir, filename
    else:
        raise ValueError()
def checkpoint(model: nn.Module, a: float, b: float, c: float, d: tuple, mode="evolve"):
    folder_path = '/root/autodl-tmp/ESWAcode/result3'
    os.makedirs(folder_path, exist_ok=True)

    if mode == "train":
        filename = "train.pth"
    elif mode == "evolve":
        filename = "evolve.pth"
    else:
        filename = "model.pth"

    save_file = os.path.join(folder_path, filename)
    torch.save(model.state_dict(), save_file)
    print(f'Saved: {save_file} (a={a:.4f}, b={b:.4f}, c={c:.4f}, c={c}, mode={mode})')

############################
# 7. Main Program
############################
if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    args = parse_commandline()
    config_path = "/root/autodl-tmp/infaredandpyhcode/config/model2.yaml"
    config = load_config(config_path)

    # Path for saving results
    save_loss_plot_path = "/root/autodl-tmp/infaredandpyhcode/result123"
    os.makedirs(save_loss_plot_path, exist_ok=True)
    config.result_path = save_loss_plot_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    if args.evolve:
        # Evolution search
        best_a, best_b, best_c, best_d = evolve(config, device)
        # Full training with optimal hyperparameters
        train(config, device,
              a=best_a, b=best_b, c=best_c, d=best_d,
              save_loss_plot_path=save_loss_plot_path)

    elif args.train:
        # Direct normal training (given a set of manual hyperparameter examples)
        # alpha + mu = 100,  betas sum=100
        fixed_a = 1.44825699
        fixed_b   = 8.55174301
        fixed_c = 3.51766962    # lamda仅需在[1,100]
        fixed_d =6.48233038  # sum=100
        train(config, device,
              a=fixed_a, b=fixed_b, c=fixed_c, d=fixed_d,
              save_loss_plot_path=save_loss_plot_path)

    elif args.test:
        # Test
        config.model.pretrained ="/root/autodl-tmp/ESWAcode/result3/evolve.pth"
        assert os.path.isfile(config.model.pretrained), FileNotFoundError(config.model.pretrained)
        test(config, "/root/autodl-tmp/data/testdata/MSRS", device)
        test(config, "/root/autodl-tmp/data/testdata/M3FD", device)
        test(config, "/root/autodl-tmp/data/testdata/LLVIP", device)
        test(config, "/root/autodl-tmp/data/testdata/Road", device)
        print("Testing flow is not implemented in this snippet.")
    else:
        print("No action specified. Use --train, --test, or --evolve.")
