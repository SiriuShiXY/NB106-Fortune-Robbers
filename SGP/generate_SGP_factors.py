#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os, random, math, operator, json, re, gc, pickle
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, gp

from utils import (
    protected_division, protected_sqrt, protected_log, safe_exp,
    compute_topR, compute_monotonicity, compute_rank_ic,
    # Alpha operators
    ts_returns_mean, ts_returns_std, ts_rank_mean, ts_rank_std,
    ts_ma, ts_ema, ts_wma, ts_momentum, ts_rsi, ts_beta,
    ts_autocorr, ts_half_life, ts_volatility, ts_downside_std,
    ts_skew, ts_kurt,
    # Constants
    ARITY
)

# ===== Configuration  =====
 
random.seed(42) # random seed  
np.random.seed(42) # random seed  
lag_period   = 12 # lag 

# file paths
data_path    = "your/base/directory/Cleaned Data"  # replace with your path
output_dir   = "your/base/directory/SGP_Final_Results"  # replace with your path
ckpt_dir     = os.path.join(output_dir, 'checkpoints')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ckpt_dir,  exist_ok=True)

# time spans for 5-fold cross-validation
spans = [("2015/10/8","2018/12/21"),
         ("2017/5/5","2020/8/16"),
         ("2018/12/19","2022/4/1"),
         ("2020/8/13","2023/11/25"),
         ("2021/3/14","2024/6/25")]
spans = [(datetime.strptime(s, "%Y/%m/%d"),
          datetime.strptime(e, "%Y/%m/%d")) for s, e in spans]

# SGP parameters
LAMBDA1 = 0.4
LAMBDA2 = 2.0
fitness_cache = {}

# Preprocessing function
def preprocess(df: pd.DataFrame):
    exclude = set(df.select_dtypes(exclude=[np.number]).columns) \
            | set(df.columns[df.isnull().any()]) \
            | {'date', 'code', 'rank', 'rank_if'}
    feats = [c for c in df.columns if c not in exclude]
    if not feats:
        return np.empty(0), *([np.empty(0)]*4)

    Xnum = StandardScaler().fit_transform(df[feats])
    if len(Xnum) <= lag_period:
        return np.empty(0), *([np.empty(0)]*4)

    n_rows, n_cols = len(Xnum) - lag_period, len(feats) * lag_period
    Xlag = np.zeros((n_rows, n_cols))
    for i in range(lag_period, len(Xnum)):
        idx = 0
        for k in range(1, lag_period + 1):
            Xlag[i - lag_period, idx: idx + len(feats)] = Xnum[i - k]
            idx += len(feats)

    names = [f'{f}_lag{k}'
             for k in range(1, lag_period + 1)
             for f in feats]

    return (Xlag,
            df['涨跌幅(%)'].values[lag_period:],  # assuming '涨跌幅(%)' is the target
            df['date'].values[lag_period:],
            df['code'].values[lag_period:],
            names)

def split_data(csv_path):
    """Split data into 5 folds based on predefined spans"""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    out = []
    for s, e in spans:
        d = df[(df['date'] >= s) & (df['date'] <= e)].copy()
        if d.empty:
            out.append(pd.DataFrame())
            continue
        d.sort_values('date', inplace=True)
        keep = int(len(d) * 0.7)
        out.append(d.iloc[:keep])
    return out

# ============== GP ==============

def batch_feats(Xb, funcs):
    """ Calculate features in batches for better memory efficiency """
    F = np.zeros((len(Xb), len(funcs)))
    for j, f in enumerate(funcs):
        try:
            F[:, j] = np.array([f(*row) for row in Xb], dtype=np.float64)
            F[:, j] = np.nan_to_num(F[:, j], nan=0.0, posinf=1e10, neginf=-1e10)
            F[:, j] = np.clip(F[:, j], -1e10, 1e10)
        except Exception as err:
            print(f"Error in feature {j}: {err}")
            F[:, j] = 0.0
    return F

def build_sgp_feature_dict(feature_names, out_dir=None):
    pat, mapping = re.compile(r'^(.*)_lag(\d+)$'), {}
    for idx, f in enumerate(feature_names):
        m = pat.match(f)
        mapping[f'feature_{idx}'] = {
            'original': m.group(1) if m else f,
            'lag': int(m.group(2)) if m else 0
        }
    if out_dir:
        with open(os.path.join(out_dir, 'sgp_feature_mapping.json'),
                  'w', encoding='utf-8') as fp:
            json.dump(mapping, fp, indent=4, ensure_ascii=False)
    return mapping

global_x, global_y, global_dates, global_toolbox = None, None, None, None

def init_worker(x, y, dates, toolbox):
    """Initialize worker with shared data"""
    global global_x, global_y, global_dates, global_toolbox
    global_x, global_y, global_dates, global_toolbox = x, y, dates, toolbox

def eval_fitness_cs(individual, X=None, y=None, dates=None, toolbox=None, n_groups=5):
    """
    Evaluate fitness of an individual in a cross-sectional manner.
    This function computes the top R-squared, monotonicity, and rank IC
    for each date group, and returns a combined fitness score.
    """
    global global_x, global_y, global_dates, global_toolbox
    if X is None:      X      = global_x
    if y is None:      y      = global_y
    if dates is None:  dates  = global_dates
    if toolbox is None:toolbox = global_toolbox

    func = toolbox.compile(expr=individual)

    preds = np.array([func(*row) for row in X], dtype=np.float64)
    preds = np.nan_to_num(preds, nan=0.0, posinf=1e10, neginf=-1e10)

    uniq_dates = np.unique(dates)
    top_list, mono_list, ic_list = [], [], []

    for d in uniq_dates:
        m = dates == d
        if m.sum() < n_groups:
            continue
        p, t = preds[m], y[m]
        top_list.append(compute_topR(p, t, n_groups))
        mono_list.append(compute_monotonicity(p, t, n_groups))
        ic_list.append(compute_rank_ic(p, t))

    if not top_list:                        
        return (-np.inf,)

    fitness = (np.mean(top_list)
               + LAMBDA1 * np.mean(mono_list)
               + LAMBDA2 * np.mean(ic_list))
    return (fitness,)

def parallel_evaluate(individuals, x, y, dates, toolbox, n_jobs=None):
    """Multi-process evaluation of individuals"""
    import multiprocessing as mp
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    chunk = max(1, len(individuals) // (n_jobs * 4))
    with mp.Pool(processes=n_jobs,
                 initializer=init_worker,
                 initargs=(x, y, dates, toolbox)) as pool:
        fitnesses = list(
            tqdm(pool.imap(eval_fitness_cs, individuals, chunksize=chunk),
                 total=len(individuals), desc="Evaluating", unit="ind")
        )
    return fitnesses

def cached_eval(ind, X, y, dates, toolbox):
    k = str(ind)
    if k in fitness_cache:
        return fitness_cache[k]
    fit = eval_fitness_cs(ind, X, y, dates, toolbox)
    fitness_cache[k] = fit
    return fit

def ensure_unique_current(pop, toolbox, target):
    seen, uniq = set(), []
    for ind in pop:
        s = str(ind)
        if s not in seen:
            seen.add(s)
            uniq.append(ind)
    while len(uniq) < target:
        ind = toolbox.individual()
        if str(ind) not in seen:
            uniq.append(ind)
            seen.add(str(ind))
    return uniq


if __name__ == "__main__":

    # fetch data and preprocess 
    folds, global_names = [[] for _ in range(5)], None

    for file in os.listdir(data_path):
        if not file.endswith('.csv'):
            continue
        for i, part in enumerate(split_data(os.path.join(data_path, file))):
            if part.empty:
                continue
            X, y, dates, _, names = preprocess(part)
            if X.size:
                folds[i].append((X, y, dates))
            if global_names is None and names:
                global_names = names

    merged = []
    for fold in folds:
        if not fold:
            continue
        Xc = np.concatenate([t[0] for t in fold])
        yc = np.concatenate([t[1] for t in fold])
        dc = np.concatenate([t[2] for t in fold])
        merged.append((Xc, yc, dc))

    m = min(len(t[0]) for t in merged)
    X    = np.concatenate([t[0][:m] for t in merged])
    y    = np.concatenate([t[1][:m] for t in merged])
    dates = np.concatenate([t[2][:m] for t in merged])

    n_feat = X.shape[1]
    print(f"data: {len(X):,} features {n_feat}")

    # SGP PrimitiveSet 
    pset = gp.PrimitiveSet("MAIN", n_feat)
    
    # Basic arithmetic operators
    for op, ar in [(operator.add, 2), (operator.sub, 2), (operator.mul, 2),
                   (protected_division, 2),
                   (operator.neg, 1), (math.sin, 1), (math.cos, 1),
                   (protected_sqrt, 1), (protected_log, 1),
                   (abs, 1), (safe_exp, 1)]:
        pset.addPrimitive(op, ar)

    # Add Alpha operators
    for op in [ts_returns_mean, ts_returns_std, ts_rank_mean, ts_rank_std,
               ts_ma, ts_wma, ts_momentum, ts_rsi, ts_beta,
               ts_autocorr, ts_half_life, ts_volatility, ts_downside_std,
               ts_skew, ts_kurt]:
        pset.addPrimitive(op, ARITY) 

    # Special case for ts_ema 
    pset.addPrimitive(ts_ema, ARITY)

    for i in range(n_feat):
        pset.renameArguments(**{f'ARG{i}': f'feature_{i}'})

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("expr", gp.genFull, pset=pset, min_=1, max_=2)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("compile", gp.compile, pset=pset)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp.genFull, min_=0, max_=2)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)
    tb.register("select", tools.selTournament, tournsize=3)
    tb.register("evaluate", cached_eval, X=X, y=y, dates=dates, toolbox=tb)

    # Evolution parameters
    POP, NGEN, CXPB, MUTPB = 300, 70, 0.4, 0.10
    REPL, ELITE = int(POP * 0.4), int(POP * 0.1)

    # checkpoint
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_gen_')]
    if ckpts:
        latest = max(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        with open(os.path.join(ckpt_dir, latest), 'rb') as f:
            chk = pickle.load(f)
        pop, start_gen, stats_hist = (chk['population'], chk['generation'],
                                      chk['stats_history'])
        fitness_cache.update(chk['fitness_cache'])
        print(f"⚡ 载入存档 {latest} (generation {start_gen})")
    else:
        pop, start_gen, stats_hist = tb.population(POP), 0, []

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean); stats.register("std", np.std)
    stats.register("min", np.min); stats.register("max", np.max)

    # Evolution 
    for gen in range(start_gen, NGEN):
        print(f"\n=== Generation {gen + 1} / {NGEN} ===")

        invalid = [ind for ind in pop if not ind.fitness.valid]
        if invalid:
            new_ind = [i for i in invalid if str(i) not in fitness_cache]
            if new_ind:
                fits = parallel_evaluate(new_ind, X, y, dates, tb, n_jobs=6)
                for ind, fit in zip(new_ind, fits):
                    fitness_cache[str(ind)] = fit
            for ind in invalid:
                ind.fitness.values = fitness_cache[str(ind)]

        cur_stats = stats.compile(pop)
        stats_hist.append(cur_stats)
        print(f"Stats: {cur_stats}")

        if (gen + 1) % 5 == 0:
            ck_file = os.path.join(ckpt_dir, f'checkpoint_gen_{gen + 1}.pkl')
            with open(ck_file, 'wb') as f:
                pickle.dump({'population': pop, 'generation': gen + 1,
                             'fitness_cache': fitness_cache,
                             'stats_history': stats_hist}, f)
            print(f"💾 checkpoint saved → {ck_file}")

        uniq_seen, uniq_best = set(), []
        for ind in sorted(pop, key=lambda i: i.fitness.values[0], reverse=True):
            s = str(ind)
            if s not in uniq_seen:
                uniq_best.append(ind); uniq_seen.add(s)
            if len(uniq_best) == 5:
                break
        print("\nTop-5 expressions:")
        for k, ind in enumerate(uniq_best, 1):
            print(f"{k}. Fit = {ind.fitness.values[0]:.6f}  |  {ind}")

        elites = tools.selBest(pop, ELITE)
        to_rep = tb.select(pop, REPL)
        offspring = [tb.clone(ind) for ind in to_rep]

        for i in range(0, len(offspring), 2):
            if i + 1 < len(offspring) and random.random() < CXPB:
                tb.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values, offspring[i + 1].fitness.values

        for ind in offspring:
            if random.random() < MUTPB:
                tb.mutate(ind)
                del ind.fitness.values

        remain = tools.selBest([p for p in pop if p not in to_rep],
                               POP - len(elites) - len(offspring))
        pop = ensure_unique_current(elites + offspring + remain, tb, POP)

    fn_final = os.path.join(ckpt_dir, 'final_checkpoint.pkl')
    with open(fn_final, 'wb') as f:
        pickle.dump({'population': pop, 'generation': NGEN,
                     'fitness_cache': fitness_cache,
                     'stats_history': stats_hist}, f)
    print(f"\n🏁 Evolution finished. Final checkpoint → {fn_final}")

    # Select top-30 features
    best30 = tools.selBest(pop, 30)
    funcs = [tb.compile(expr=i) for i in best30]

    batch = 1000
    mat = []
    for i in tqdm(range((len(X) + batch - 1) // batch)):
        s, e = i * batch, min((i + 1) * batch, len(X))
        mat.append(batch_feats(X[s:e], funcs))
    feats_mat = np.vstack(mat)
    np.save(os.path.join(output_dir, 'optimal_features_top30.npy'), feats_mat)

    fmap = build_sgp_feature_dict(global_names, output_dir)
    pat  = re.compile("your_feature_(\d+)")
    def map_expr(expr: str):
        return pat.sub(lambda m: f"{fmap[m.group()]['original']}_lag{fmap[m.group()]['lag']}", expr)

    with open(os.path.join(output_dir, 'top30_individuals.txt'), 'w') as fp:
        fp.write(f"Using {lag_period}-day lagged features\n\n")
        for i, ind in enumerate(best30, 1):
            fp.write(f"{i}. Fit={ind.fitness.values[0]:.4f}\n")
            fp.write("   Expr:   " + str(ind) + "\n")
            fp.write("   Mapped: " + map_expr(str(ind)) + "\n\n")

    print("Top-30 features & expressions saved.")

    # Generate augmented CSV files
    print("Start augmenting individual CSV files ...")

    clean_dir = data_path
    aug_dir   = "youtr/base/directory/SGP_Augmented"  # replace with your path
    os.makedirs(aug_dir, exist_ok=True)

    factor_cols = [f"SGP_F{i + 1:02d}" for i in range(30)]

    for fname in tqdm([f for f in os.listdir(clean_dir) if f.endswith('.csv')],
                      desc="Augmenting"):
        df_raw = pd.read_csv(os.path.join(clean_dir, fname))
        X0, _, d0, c0, _ = preprocess(df_raw)
        if X0.size == 0:
            continue
        F = batch_feats(X0, funcs)
        df_f = pd.DataFrame(F, columns=factor_cols)
        df_f['date'] = d0
        df_f['code'] = c0
        df_f.to_csv(os.path.join(aug_dir, fname), index=False)
        gc.collect()
