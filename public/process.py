import os
import copy
import pickle
import sympy
import functools
import itertools
import sys
import json
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import mutual_info_score, auc, roc_curve, roc_auc_score, f1_score
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize as scipy_min
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, Bounds, linprog
from sympy import Symbol as sb
from sympy import lambdify
from IPython.display import display,clear_output
from random import choice
from multiprocessing import Pool

from sympy.printing.latex import LatexPrinter
class ForcePlus(LatexPrinter):
    def _print_Add(self, expr):
        terms = list(expr.args)
        latex_terms = []
        for i, term in enumerate(terms):
            if i > 0:
                if term.could_extract_minus_sign():
                    latex_terms.append(f"+ (-{super()._print(-term)})")
                else:
                    latex_terms.append(f"+ {super()._print(term)}")
            else:
                latex_terms.append(super()._print(term))
        return " ".join(latex_terms)

import importlib.util
import ginac_module

class FileFormatError(Exception):
    def __init__(self, f, *args):
        super().__init__(args)
        self.fname = f

    def __str__(self):
        return f'The file "{self.fname}" does not have a valid format! If the file is a training dataset, contact the administrator for assistance.'
    
symbol_id = -1

def create_symbol(suffix=''):
    global symbol_id
    symbol_id += 1
    name = f'e{symbol_id}_{suffix}' if suffix else f'e{symbol_id}'
    return sympy.Symbol(name=name)

def get_symbol_count(df):
    global symbol_id
    nan_count = np.isnan(df).sum()
    symbol_id = nan_count - 1

def add_red(match):
    return "\\textcolor{red}{" + match.group(0) + "}"

def add_green(match):
    return "\\textcolor{green}{" + match.group(0) + "}"

def round_coeff(expr, ndigits=6):
    new_expr = 0
    
    for term in expr.as_ordered_terms():
        coeff, symbolic = term.as_independent(*expr.free_symbols)
        coeff_rounded = round(coeff, ndigits)
        new_expr += coeff_rounded * symbolic
    
    return new_expr

def simplify_expression(expr) -> str:
    data_count = 0
    nd_count = 0
    # print(expr, file=sys.stderr)
    expr = round_coeff(expr)
    print(expr, file=sys.stderr)
    symbols_list = list(expr.free_symbols)
    force_plus = ForcePlus()
    for s in symbols_list:
        s_str = str(s)
        if 'ep' in s_str:
            nd_count += 1
        elif 'e' in s_str and 'ep' not in s_str:
            data_count += 1

    if data_count <= 2 and nd_count <= 2:
        return force_plus.doprint(expr)
    else:
        last_data = "e" + str(data_count - 1)
        last_nd = "ep" + str(nd_count - 1)
        simplified_list = []
        for term in expr.args:
            if term.is_Number:
                simplified_list.append(term)
                continue
            coefficient, symbol = term.as_coeff_Mul()
            if str(symbol) == "e0" or str(symbol) == last_data or str(symbol) == "ep0" or str(symbol) == last_nd:
                simplified_list.append(term)

        # print(simplified_list, file=sys.stderr)

        new_expr = sympy.Add(*simplified_list)
        new_latex = force_plus.doprint(new_expr)
        print(new_latex, file=sys.stderr)
        latex_parts = new_latex.split('+')
        latex_parts.sort(key=lambda term: (0 if 'e_{0}' in term else 1 if 'e_{' + str(data_count - 1) in term else 2 if 'ep_{0}' in term else 3 if 'ep_{' + str(nd_count - 1) in term else 4))

        pattern_e = r"e_{\d+}"
        pattern_ep = r"ep_{\d+}"
        print(latex_parts, file=sys.stderr)
        latex_parts = [re.sub(pattern_e, add_red, latex_parts[0]), re.sub(pattern_e, add_red, latex_parts[1]), re.sub(pattern_ep, add_green, latex_parts[2]), re.sub(pattern_ep, add_green, latex_parts[3]), latex_parts[4]]

        print(latex_parts, file=sys.stderr)
        
        return latex_parts[0] + '+ \dots +' + latex_parts[1] + '+' + latex_parts[2] + '+ \dots +' + latex_parts[3] + '+ \\textcolor{blue}{' + latex_parts[4] + '}'
    
def get_weight(expr):
    wmax = 0
    wmin = 0
    wmid = 0
    for term in expr.args:
        coefficient, symbol = term.as_coeff_Mul()
        if term.is_Number:
            wmid = coefficient
            continue
        if coefficient < 0:
            wmax += (-1) * coefficient
            wmin += coefficient
        else:
            wmax += coefficient
            wmin += (-1) * coefficient

    return [wmax, abs(wmin), wmid]

def get_expr_range_radius(expr):
    expr_range_radius = 0
    for arg in expr.args:
        if arg.free_symbols:
            expr_range_radius += abs(arg.args[0])
    return expr_range_radius

def get_expr_center(expr):
    return expr.subs(dict([(symb, 0) for symb in expr.free_symbols]))

def sympy_to_ginac_format(expr):
    """Convert SymPy expression to GiNaC-parseable string format."""
    # Replace SymPy-specific functions with GiNaC equivalents
    s = str(expr)
    s = s.replace('**', '^')  # Exponentiation
    s = s.replace('sin', 'GiNaC::sin')
    s = s.replace('cos', 'GiNaC::cos')

    return s

def robustness_report(model, model_oi, X_test, ss):
    X_test_ss = np.append(np.ones((len(X_test), 1)), ss.transform(X_test), axis=1)
    X_test_sp = sympy.Matrix(X_test_ss)
    X_test_list = list(np.array(X_test_sp, dtype=float))
    X_test_list = [list(l) for l in X_test_list]
    # test_preds = X_test_sp*model
    radius_base = 1000
    robustness_radius = [r * radius_base for r in [0.1, 0.2, 0.3, 0.5, 0.75, 1]]
    robustness_ratios = []
    pred_range_radiuses = []
    pred_centers = []
    pred_ub = []
    pred_lb = []
    pred_oi = []

    ginac_param_list = [sympy_to_ginac_format(s) for s in model]

    preds = ginac_module.pred_test(X_test_list, ginac_param_list)

    for pred_id in range(len(X_test_list)):
        # pred = test_preds[pred_id]
        pred_range_radius = preds[pred_id][1]
        pred_center = preds[pred_id][0]
        pred_centers.append(round(float(pred_center)))
        pred_ub.append(round(float(pred_center + pred_range_radius)))
        pred_lb.append(round(float(pred_center - pred_range_radius)))
        pred_range_radiuses.append(pred_range_radius)
        oi_pred = 0
        for j in range(len(model_oi)):
            if j == 0:
                oi_pred += model_oi[j]
            else:
                oi_pred += model_oi[j] * X_test_ss[pred_id, j]
        pred_oi.append(round(float(oi_pred), 2))

    for radius in robustness_radius:
        robustness_ls = []
        for pred_id in range(len(X_test_list)):
            if pred_range_radiuses[pred_id] <= radius:
                robustness_ls.append(1)
            else:
                robustness_ls.append(0)

        robustness_ratios.append(float(np.mean(robustness_ls)))

    return robustness_ratios, pred_centers, pred_ub, pred_lb, pred_oi

def getMissingDataIns(dirty_df, dirty_y):
    age = []
    children = []
    dirty_ys = []
    cage = []
    cchildren = []
    cy = []
    for i in range(len(dirty_df)):
        if pd.isna(dirty_df.iloc[i]['bmi']):
            age.append(int(dirty_df.iloc[i]['age']))
            children.append(int(dirty_df.iloc[i]['children']))
            dirty_ys.append(round(float(dirty_y.iloc[i])))
        elif pd.isna(dirty_df.iloc[i]['age']):
            age.append(int(dirty_df.iloc[i]['bmi']))
            children.append(int(dirty_df.iloc[i]['children']))
            dirty_ys.append(round(float(dirty_y.iloc[i])))
        elif pd.isna(dirty_df.iloc[i]['children']):
            age.append(int(dirty_df.iloc[i]['bmi']))
            children.append(int(dirty_df.iloc[i]['age']))
            dirty_ys.append(round(float(dirty_y.iloc[i])))
        else:
            cage.append(int(dirty_df.iloc[i]['age']))
            cchildren.append(int(dirty_df.iloc[i]['children']))
            cy.append(round(float(dirty_y.iloc[i])))

    return age, children, dirty_ys, cage, cchildren, cy

def getMissingDataGeneral(dirty_df, dirty_y, uncertain_attr: int):
    dirty_features = [[] for _ in range(dirty_df.shape[1] - 1)]
    clean_features = [[] for _ in range(dirty_df.shape[1] - 1)]
    dirty_ys = []
    cy = []
    
    columns = dirty_df.columns.tolist()

    for i in range(len(dirty_df)):
        row = dirty_df.iloc[i]
        label = round(float(dirty_y.iloc[i]))

        if row.isna().any():
            flag = True
            for j, col in enumerate(columns):
                if j == uncertain_attr:
                    flag = False
                    continue

                if flag:
                    dirty_features[j].append(float(row[col]))
                else:
                    dirty_features[j-1].append(float(row[col]))
            
            dirty_ys.append(round(float(label)))
        
        else:
            flag = True
            for j, col in enumerate(columns):
                if j == uncertain_attr:
                    flag = False
                    continue
                if flag:
                    clean_features[j].append(float(row[col]))
                else:
                    clean_features[j-1].append(float(row[col]))
                    
            cy.append(round(float(label)))
    
    return dirty_features, clean_features, dirty_ys, cy

def getFeaturesList(path_to_dataset):
    try:
        feature_df = pd.read_csv(path_to_dataset)
    except:
        try:
            feature_df = pd.read_excel(path_to_dataset)
        except:
            f_name = path_to_dataset.split('/')[-1]
            raise FileFormatError(f_name)
        
    return feature_df.columns.tolist()

def compute_closed_form(X, y):
    return np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))

def merge_small_components_pca(expr_ls, budget=10):
    if not(isinstance(expr_ls, sympy.Expr)):
        expr_ls = sympy.Matrix(expr_ls)
    if expr_ls.free_symbols:
        center = expr_ls.subs(dict([(symb, 0) for symb in expr_ls.free_symbols]))
    else:
        return expr_ls
    monomials_dict = get_generators(expr_ls)
    generators = np.array([monomials_dict[m] for m in monomials_dict])
    if len(generators) <= budget:
        return expr_ls
    monomials = [m for m in monomials_dict]
    pca = PCA(n_components=len(generators[0]))
    pca.fit(np.concatenate([generators, -generators]))
    transformed_generators = pca.transform(generators)
    transformed_generator_norms = np.linalg.norm(transformed_generators, axis=1, ord=2)
    # from largest to lowest norm
    sorted_indices = transformed_generator_norms.argsort()[::-1].astype(int)
    sorted_transformed_generators = transformed_generators[sorted_indices]
    sorted_monomials = [monomials[idx] for idx in sorted_indices]
    new_transformed_generators = np.concatenate([sorted_transformed_generators[:budget], 
                                                 np.diag(np.sum(np.abs(sorted_transformed_generators[budget:]), 
                                                                axis=0))])
    new_generators = pca.inverse_transform(new_transformed_generators)
    new_monomials = sorted_monomials[:budget] + [create_symbol() for _ in range(len(generators[0]))]
    
    processed_expr_ls = center
    for monomial_id in range(len(new_monomials)):
        processed_expr_ls += sympy.Matrix(new_generators[monomial_id])*new_monomials[monomial_id]
    
    return processed_expr_ls

def merge_small_components_pca_optimized(expr_ls, budget=10):
    # Convert to sympy.Matrix only if needed
    if not isinstance(expr_ls, sympy.Expr):
        expr_ls = sympy.Matrix(expr_ls)
    
    # Early exit if no free symbols
    free_symbols = expr_ls.free_symbols
    if not free_symbols:
        return expr_ls
    
    # Compute center more efficiently
    center = expr_ls.subs([(symb, 0) for symb in free_symbols])
    
    # Get generators
    monomials_dict = get_generators(expr_ls)
    if len(monomials_dict) <= budget:
        return expr_ls
    
    # Convert to numpy arrays once
    monomials = list(monomials_dict.keys())
    generators = np.array([monomials_dict[m] for m in monomials])
    
    # Optimize PCA fitting
    # Instead of concatenating [generators, -generators], we can use the fact that
    # PCA on X and PCA on [X, -X] give the same principal components
    pca = PCA(n_components=generators.shape[1])
    pca.fit(generators)  # Just fit on generators, not doubled
    
    # Transform and compute norms in one operation
    transformed_generators = pca.transform(generators)
    transformed_generator_norms = np.linalg.norm(transformed_generators, axis=1, ord=2)
    
    # Use argpartition instead of full sort (faster for large arrays)
    if len(generators) > budget * 2:
        # We only need the top 'budget' elements
        top_budget_indices = np.argpartition(transformed_generator_norms, -budget)[-budget:]
        sorted_indices = top_budget_indices[np.argsort(transformed_generator_norms[top_budget_indices])[::-1]]
    else:
        sorted_indices = transformed_generator_norms.argsort()[::-1]
    
    # Select top generators
    sorted_transformed_generators = transformed_generators[sorted_indices]
    sorted_monomials = [monomials[idx] for idx in sorted_indices]
    
    # Optimize the diagonal computation
    remaining_transformed = sorted_transformed_generators[budget:]
    if remaining_transformed.size > 0:
        diagonal_values = np.sum(np.abs(remaining_transformed), axis=0)
        new_transformed_generators = np.vstack([
            sorted_transformed_generators[:budget],
            np.diag(diagonal_values)
        ])
    else:
        new_transformed_generators = sorted_transformed_generators[:budget]
    
    # Inverse transform
    new_generators = pca.inverse_transform(new_transformed_generators)
    
    # Pre-create all symbols at once (if create_symbol is expensive)
    num_new_symbols = new_generators.shape[0] - budget
    new_symbols = [create_symbol() for _ in range(num_new_symbols)]
    new_monomials = sorted_monomials[:budget] + new_symbols
    
    # Optimize final expression construction
    # Instead of adding in a loop, construct all at once
    monomial_matrix = sympy.zeros(len(new_generators[0]), 1)
    for monomial_id, (generator, monomial) in enumerate(zip(new_generators, new_monomials)):
        monomial_matrix += sympy.Matrix(generator) * monomial
    
    processed_expr_ls = center + monomial_matrix
    
    return processed_expr_ls

def process_2d_pair(ij, param):
    i, j = ij
    if i == j:
        return None
    vert1, vert2 = get_vertices_group([param[i], param[j]], 0.5, budget=10)
    vert1 = [float(l) for l in vert1]
    vert2 = [float(l) for l in vert2]
    vert1.append(vert1[0])
    vert2.append(vert2[0])
    return f'f{i},f{j}', [vert1, vert2]

def process_3d_triplet(ijk, param):
    i, j, k = ijk
    if i == j or j == k or i == k:
        return None
    vert1, vert2, vert3, triangle1, triangle2, triangle3 = get_vertices_group_3([param[i], param[j], param[k]], 0.5, budget=10)
    vert1 = [float(l) for l in vert1]
    vert2 = [float(l) for l in vert2]
    vert3 = [float(l) for l in vert3]
    triangle1 = [int(l) for l in triangle1]
    triangle2 = [int(l) for l in triangle2]
    triangle3 = [int(l) for l in triangle3]
    return f'f{i},f{j},f{k}', [vert1, vert2, vert3, triangle1, triangle2, triangle3]

# take a list of expressions as input, output the list of monomials and generator vectors,
def get_generators(expr_ls):
    monomials = dict()
    for expr_id, expr in enumerate(expr_ls):
        if not(isinstance(expr, sympy.Expr)) or not(expr.free_symbols):
            continue
        expr = expr.expand()
        p = sympy.Poly(expr)
        monomials_in_expr = [sympy.prod(x**k for x, k in zip(p.gens, mon)) 
                             for mon in p.monoms() if sum(mon) >= 1]
        for monomial in monomials_in_expr:
            coef = float(p.coeff_monomial(monomial))
            if monomial in monomials:
                if len(monomials[monomial]) < expr_id:
                    monomials[monomial] = monomials[monomial] + [0 for _ in range(expr_id-len(monomials[monomial]))]
                monomials[monomial].append(coef)
            else:
                monomials[monomial] = [0 for _ in range(expr_id)] + [coef]

    for monomial in monomials:
        if len(monomials[monomial]) < len(expr_ls):
            monomials[monomial] = monomials[monomial] + [0 for _ in range(len(expr_ls)-len(monomials[monomial]))]
    
    return monomials

def plot_conretiztion(affset, alpha = 0.5, color='red', budget=-1, 
                      label='Ours', edgecolor=None, linewidth=1):
    if budget > -1:
        affset = merge_small_components_pca(affset, budget=budget)
    pts = np.array(list(map(list, get_vertices(affset))))
    hull = ConvexHull(pts)
    print(pts[hull.vertices,0])
    print(pts[hull.vertices,1])
    plt.fill(pts[hull.vertices,0], pts[hull.vertices,1], color, alpha=alpha, 
             label=label, edgecolor=edgecolor, linewidth=linewidth)

def get_vertices(affset):
    l = len(affset)
    distinct_symbols = set()
    for expr in affset:
        if not(isinstance(expr, sympy.Expr)):
            assert isinstance(expr, int) or isinstance(expr, float)
        else:
            if distinct_symbols:
                distinct_symbols = distinct_symbols.union(expr.free_symbols)
            else:
                distinct_symbols = expr.free_symbols
    distinct_symbols = list(distinct_symbols)
    # print(distinct_symbols)
    combs = [list(zip(distinct_symbols,list(l))) for l in list(itertools.product([-1, 1], repeat=len(distinct_symbols)))]
    res = set()
    for assignment in combs:
        res.add(tuple([expr.subs(assignment) for expr in affset]))
    return(res)

def get_vertices_group(affset, alpha = 0.5, budget=-1):
    if budget > -1:
        affset = merge_small_components_pca_optimized(affset, budget=budget)
    pts = np.array(list(map(list, get_vertices(affset))))
    hull = ConvexHull(pts)
    return pts[hull.vertices,0], pts[hull.vertices,1]

def get_vertices_group_3(affset, alpha = 0.5, budget=-1):
    if budget > -1:
        affset = merge_small_components_pca_optimized(affset, budget=budget)
    pts = np.array(list(map(list, get_vertices(affset))))
    hull = ConvexHull(pts)
    vertex_map = {old_index: new_index for new_index, old_index in enumerate(hull.vertices)}
    simplified_simplices = np.array([
        [vertex_map[idx] for idx in simplex]
        for simplex in hull.simplices
    ])

    return pts[hull.vertices,0], pts[hull.vertices,1], pts[hull.vertices,2], simplified_simplices[:, 0], simplified_simplices[:, 1], simplified_simplices[:, 2]

    
if __name__ == "__main__":
    dataset = sys.argv[1]
    test = sys.argv[2]
    lr = float(sys.argv[3])
    reg = float(sys.argv[4])
    features_list = []

    with open('public/trained_model.json', 'r') as json_file:
        json_str = json_file.read()
        json_body = json.loads(json_str)

    if dataset not in json_body.keys() or json_body[dataset][0] != lr or json_body[dataset][1] != reg:
        # train new model
        with open('public/custom.json', 'r') as train_json_file:
            train_str = train_json_file.read()
            train_json_body = json.loads(train_str)
            if dataset not in train_json_body.keys():
                raise FileFormatError('custom.json')
        
        feature_path = train_json_body[dataset][0]
        feature_path = 'dataset/' + dataset + '/' + feature_path
        label_path = train_json_body[dataset][1]
        label_path = 'dataset/' + dataset + '/' + label_path
        clean_path = train_json_body[dataset][2]
        clean_path = 'dataset/' + dataset + '/' + clean_path

        X_train = None
        X_clean = None
        y_train = None

        try:
            X_train = pd.read_csv(feature_path)
        except:
            try:
                X_train = pd.read_excel(feature_path)
            except:
                f_name = feature_path.split('/')[-1]
                raise FileFormatError(f_name)
            
        if X_train is None:
            f_name = feature_path.split('/')[-1]
            raise FileFormatError(f_name)
        
        try:
            y_train = pd.read_csv(label_path)  
        except:
            try:
                y_train = pd.read_excel(label_path)
            except:
                f_name = label_path.split('/')[-1]
                raise FileFormatError(f_name)
        
        if y_train is None:
            f_name = label_path.split('/')[-1]
            raise FileFormatError(f_name)
        
        try:
            X_clean = pd.read_csv(clean_path)
        except:
            try:
                X_clean = pd.read_excel(clean_path)
            except:
                f_name = clean_path.split('/')[-1]
                raise FileFormatError(f_name)
            
        if X_clean is None:
            f_name = clean_path.split('/')[-1]
            raise FileFormatError(f_name)

        all_cols = X_train.columns.tolist()
        all_cols_idx = [X_train.columns.to_list().index(c) for c in all_cols]
        X_extended = np.append(np.ones((len(X_train), 1)), X_train.to_numpy().astype(float)[:, all_cols_idx], axis=1)
        X_extended_clean = np.append(np.ones((len(X_clean), 1)), X_clean.to_numpy().astype(float)[:, all_cols_idx], axis=1)
        ss = StandardScaler()
        X_extended[:, 1:] = ss.fit_transform(X_extended[:, 1:])
        X_extended_clean[:, 1:] = ss.transform(X_extended_clean[:, 1:])

        uncertain_attr = 0
        for i in range(X_extended.shape[1]):
            if np.isnan(X_extended[:, i]).any():
                uncertain_attr = i
                break

        imputers = [KNNImputer(n_neighbors=5), KNNImputer(n_neighbors=10), IterativeImputer(random_state=42)]
        num_attrs = X_extended.shape[1]
        X_nan = X_extended.copy()
        imputed_cols = [X_extended_clean[:, uncertain_attr]]
        imputed_datasets = [X_extended_clean]

        for imp in imputers:
            imputed_dataset = imp.fit_transform(X_nan)
            imputed_datasets.append(imputed_dataset)
            imputed_cols.append(imputed_dataset[:, uncertain_attr])

        X_extended_max = X_extended.copy()
        X_extended_max[:, uncertain_attr] = np.max(imputed_cols, axis=0)
        X_extended_min = X_extended.copy()
        X_extended_min[:, uncertain_attr] = np.min(imputed_cols, axis=0)

        X_max_list = [list(l) for l in X_extended_max]
        X_min_list = [list(l) for l in X_extended_min]
        X_train_list = [list(l) for l in X_extended]
        y_train_list = [float(l) for l in y_train.to_numpy()]

        param_list = ginac_module.zorro(X_train_list, y_train_list, X_max_list, X_min_list, lr, reg)
        param = sympy.Matrix([sympy.sympify(expr) for expr in param_list])

        scaled_training = imputed_datasets[1]
        one_imp_params = compute_closed_form(scaled_training, y_train)
        one_imp_params = np.array(one_imp_params.values.flatten().tolist())
        print("one_imp_params", one_imp_params, type(one_imp_params), file=sys.stderr)

        get_symbol_count(X_extended)
        param_clean = compute_closed_form(X_extended_clean, y_train)
        param_clean = np.array(param_clean.values.flatten().tolist())
        print("param_clean", param_clean, type(one_imp_params), file=sys.stderr)

        # 2D zonotope processing
        n = X_extended.shape[1]
        pairs = [(i, j) for i in range(n) for j in range(n)]

        with Pool() as pool:
            results_2d = pool.map(functools.partial(process_2d_pair, param=param), pairs)

        json_2d = {k: v for k, v in results_2d if k is not None}
        for a in range(n):
            json_2d[f'f{a}'] = float(param_clean[a])

        os.makedirs(os.path.dirname('models/' + dataset + '/' + dataset + '_2d.json'), exist_ok=True)
        with open('models/' + dataset + '/' + dataset + '_2d.json', 'w') as file:
            json.dump(json_2d, file, indent=4)

        # 3D zonotope processing
        triplets = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]

        with Pool() as pool:
            results_3d = pool.map(functools.partial(process_3d_triplet, param=param), triplets)

        json_3d = {k: v for k, v in results_3d if k is not None}
        for a in range(n):
            json_3d[f'f{a}'] = float(param_clean[a])

        os.makedirs(os.path.dirname('models/' + dataset + '/' + dataset + '_3d.json'), exist_ok=True)
        with open('models/' + dataset + '/' + dataset + '_3d.json', 'w') as file:
            json.dump(json_3d, file, indent=4)

        # json_2d = dict()
        # for i in range(X_extended.shape[1]):
        #     for j in range(X_extended.shape[1]):
        #         if i == j:
        #             continue
        #         vert1, vert2 = get_vertices_group([param[i], param[j]], 0.5, budget=10)
        #         vert1 = [float(l) for l in vert1]
        #         vert2 = [float(l) for l in vert2]
        #         vert1.append(vert1[0])
        #         vert2.append(vert2[0])
        #         json_2d['f' + str(i) + ',f' + str(j)] = [vert1, vert2]
                
        # for a in range(X_extended.shape[1]):
        #     json_2d[f'f{a}'] = float(param_clean[a])

        # os.makedirs(os.path.dirname('models/' + dataset + '/' + dataset + '_2d.json'), exist_ok=True)
        # with open('models/' + dataset + '/' + dataset + '_2d.json', 'w') as file:
        #     json.dump(json_2d, file, indent=4)

        # json_3d = dict()
        # for i in range(X_extended.shape[1]):
        #     for j in range(X_extended.shape[1]):
        #         for k in range(X_extended.shape[1]):
        #             if i == j or j == k or i == k:
        #                 continue
        #             vert1, vert2, vert3, triangle1, triangle2, triangle3 = get_vertices_group_3([param[i], param[j], param[k]], 0.5, budget=10)
        #             vert1 = [float(l) for l in vert1]
        #             vert2 = [float(l) for l in vert2]
        #             vert3 = [float(l) for l in vert3]
        #             triangle1 = [int(l) for l in triangle1]
        #             triangle2 = [int(l) for l in triangle2]
        #             triangle3 = [int(l) for l in triangle3]
        #             json_3d['f' + str(i) + ',f' + str(j) + ',f' + str(k)] = [vert1, vert2, vert3, triangle1, triangle2, triangle3]

        # for a in range(X_extended.shape[1]):
        #     json_3d[f'f{a}'] = float(param_clean[a])

        # os.makedirs(os.path.dirname('models/' + dataset + '/' + dataset + '_3d.json'), exist_ok=True)
        # with open('models/' + dataset + '/' + dataset + '_3d.json', 'w') as file:
        #     json.dump(json_3d, file, indent=4)

        with open('models/' + dataset + '/' + dataset + '_model.pkl', 'wb') as file:
            pickle.dump(param, file)

        with open('models/' + dataset + '/' + dataset + '_model_1imp.pkl', 'wb') as file:
            pickle.dump(one_imp_params, file)

        with open('models/' + dataset + '/' + dataset + '_ss.pkl', 'wb') as file:
            pickle.dump(ss, file)

        with open('models/' + dataset + '/' + dataset + '_X_train_dirty.pkl', 'wb') as file:
            pickle.dump(X_train, file)

        with open('models/' + dataset + '/' + dataset + '_y_train_dirty.pkl', 'wb') as file:
            pickle.dump(y_train, file)

        json_body[dataset] = [lr, reg]
        with open('public/trained_model.json', 'w') as json_file:
            json_file.write(json.dumps(json_body))

    # choose dataset
    if dataset == "Insurance" and lr == 0.01 and reg == 0:
        with open('models/ins/ins_30_model.pkl', 'rb') as file:
            model = pickle.load(file)
        features_list = ["age", "bmi", "children"]

        # load model with only one imputer
        with open('models/ins/ins_30_model_1imp.pkl', 'rb') as file:
            model_one = pickle.load(file)

        # load test data
        if test == "t1":
            with open('models/ins/ins_30_X_test_s1.pkl', 'rb') as file:
                X_test = pickle.load(file)
        if test == "t2":
            with open('models/ins/ins_30_X_test_s2.pkl', 'rb') as file:
                X_test = pickle.load(file)

        # load standardscaler and dirty train data
        with open('models/ins/ins_30_ss.pkl', 'rb') as file:
            ss = pickle.load(file)
        with open('models/ins/ins_30_X_train_dirty.pkl', 'rb') as file:
            X_train_dirty = pickle.load(file)
        with open('models/ins/ins_30_y_train_dirty.pkl', 'rb') as file:
            y_train_dirty = pickle.load(file)

        X_test_lists = []
        for i in range(len(X_test)):
            X_test_lists.append([float(j) for j in X_test.values[i].flatten().tolist()])

        dage, dchildren, dy, age, children, cy = getMissingDataIns(X_train_dirty, y_train_dirty)
        missing = [dage, dchildren]
        clean_data = [age, children]
        missing_feature = 'bmi'
        missing_column = 1

    else:
        with open('models/' + dataset + '/' + dataset + '_model.pkl', 'rb') as file:
            model = pickle.load(file)

        with open('public/custom.json', 'r') as feature_json:
            feature_json_str = feature_json.read()
            feature_json_body = json.loads(feature_json_str)
            # print(feature_json_body)
            if dataset not in feature_json_body.keys():
                raise FileFormatError('custom.json')
            
            feature_set = feature_json_body[dataset][0]

        features_list = getFeaturesList('dataset/' + dataset + '/' + feature_set)

        with open('models/' + dataset + '/' + dataset + '_model_1imp.pkl', 'rb') as file:
            model_one = pickle.load(file)

        with open('public/custom_test.json', 'r') as test_json:
            test_json_str = test_json.read()
            test_json_body = json.loads(test_json_str)
            # print(feature_json_body)
            if dataset not in test_json_body.keys():
                raise FileFormatError('custom_test.json')
            
            test_set = test_json_body[dataset]

        for i in range(len(test_set)):
            if test == "t" + str(i + 1):
                X_test = None
                try:
                    X_test = pd.read_csv('dataset/' + dataset + '/' + test_set[i][0])
                except:
                    try:
                        X_test = pd.read_excel('dataset/' + dataset + '/' + test_set[i][0])
                    except:
                        f_name = test_set[i]
                        raise FileFormatError(f_name)
                if X_test is None:
                    f_name = test_set[i]
                    raise FileFormatError(f_name)
                
                break

        with open('models/' + dataset + '/' + dataset + '_ss.pkl', 'rb') as file:
            ss = pickle.load(file)
        with open('models/' + dataset + '/' + dataset + '_X_train_dirty.pkl', 'rb') as file:
            X_train_dirty = pickle.load(file)
        with open('models/' + dataset + '/' + dataset + '_y_train_dirty.pkl', 'rb') as file:
            y_train_dirty = pickle.load(file)

        X_test_lists = []
        for i in range(len(X_test)):
            X_test_lists.append([float(j) for j in X_test.values[i].flatten().tolist()])

        dirty_df = X_train_dirty.to_numpy().astype(float)
        uncertain_attr = 0
        for i in range(dirty_df.shape[1]):
            if np.isnan(dirty_df[:, i]).any():
                uncertain_attr = i
                break

        missing, clean_data, dy, cy = getMissingDataGeneral(X_train_dirty, y_train_dirty, uncertain_attr)
        missing_feature = str(X_train_dirty.columns[uncertain_attr])
        missing_column = uncertain_attr

    latex_str = "\left[\\begin{matrix}"
    weight_max = []
    weight_min = []
    weight_mid = []

    for i in range(len(model)):
        latex_str += simplify_expression(model[i])
        if i != len(model) - 1:
            latex_str += "\\\\"

        if i > 0:
            weights = get_weight(model[i])
            weight_max.append(float(weights[0]))
            weight_min.append(float(weights[1]))
            weight_mid.append(float(weights[2]))

    latex_str += "\end{matrix}\\right]"

    robustness, pred_c, ub, lb, oip = robustness_report(model, model_one, X_test, ss)

    # print(weight_max, file=sys.stderr)

    output = {
        "latex": latex_str,
        "wt_max": weight_max,
        "wt_min": weight_min,
        "wt_mid": weight_mid,
        "features": features_list,
        "robustness": robustness,
        "centers": pred_c,
        "ub": ub,
        "lb": lb,
        "X_test": X_test_lists,
        "missingy": dy,
        "cleany": cy,
        "oneimp": oip,
        "missing": missing,
        "clean": clean_data,
        "missing_feature": missing_feature,
        "missing_column": missing_column
    }

    print(json.dumps(output))