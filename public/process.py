import os
import copy
import pickle
import sympy
import functools
import itertools
import sys
import json
import re

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
    
    # def _print_Mul(self, expr):
    #     if expr.could_extract_minus_sign():
    #         return f"(-{super()._print(-expr)})"
    #     return super()._print_Mul(expr)

def add_red(match):
    return "\\textcolor{red}{" + match.group(0) + "}"

def add_green(match):
    return "\\textcolor{green}{" + match.group(0) + "}"

def simplify_expression(expr) -> str:
    data_count = 0
    nd_count = 0
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
        latex_parts = new_latex.split('+')
        latex_parts.sort(key=lambda term: (0 if 'e_{0}' in term else 1 if 'e_{' + str(data_count - 1) in term else 2 if 'ep_{0}' in term else 3 if 'ep_{' + str(nd_count - 1) in term else 4))

        pattern_e = r"e_{\d+}"
        pattern_ep = r"ep_{\d+}"
        latex_parts = [re.sub(pattern_e, add_red, latex_parts[0]), re.sub(pattern_e, add_red, latex_parts[1]), re.sub(pattern_ep, add_green, latex_parts[2]), re.sub(pattern_ep, add_green, latex_parts[3]), latex_parts[4]]

        print(latex_parts, file=sys.stderr)
        
        return latex_parts[0] + '+ \dots +' + latex_parts[1] + '+' + latex_parts[2] + '+ \dots +' + latex_parts[3] + '+ \\textcolor{blue}{' + latex_parts[4] + '}'
    
def get_weight(expr) -> [float]:
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

def robustness_report(model, X_test, y_test, ss):
    y_test_sp = sympy.Matrix(y_test.to_numpy().reshape(-1, 1))
    X_test_sp = sympy.Matrix(np.append(np.ones((len(X_test), 1)), ss.transform(X_test), axis=1))
    test_preds = X_test_sp*model
    radius_base = np.median(y_test)
    robustness_radius = [r * radius_base for r in [0.01, 0.02, 0.03, 0.05, 0.10, 0.20]]
    robustness_ratios = []
    pred_range_radiuses = []
    pred_centers = []
    pred_ub = []
    pred_lb = []

    for pred_id in range(len(test_preds)):
        pred = test_preds[pred_id]
        pred_range_radius = get_expr_range_radius(pred)
        pred_center = get_expr_center(pred)
        pred_centers.append(float(pred_center))
        pred_ub.append(float(pred_center + pred_range_radius))
        pred_lb.append(float(pred_center - pred_range_radius))
        pred_range_radiuses.append(pred_range_radius)

    for radius in robustness_radius:
        robustness_ls = []
        for pred_id in range(len(test_preds)):
            if pred_range_radiuses[pred_id] <= radius:
                robustness_ls.append(1)
            else:
                robustness_ls.append(0)

        robustness_ratios.append(float(np.mean(robustness_ls)))

    return robustness_ratios, pred_centers, pred_ub, pred_lb

    
if __name__ == "__main__":
    dataset = sys.argv[1]
    test = sys.argv[2]
    features_list = []

    if dataset == "i1":
        with open('models/ins/ins_30_model.pkl', 'rb') as file:
            model = pickle.load(file)
        features_list = ["age", "bmi", "children"]
        with open('models/ins/ins_30_X_test.pkl', 'rb') as file:
            X_test = pickle.load(file)
        with open('models/ins/ins_30_y_test.pkl', 'rb') as file:
            y_test = pickle.load(file)
        with open('models/ins/ins_30_ss.pkl', 'rb') as file:
            ss = pickle.load(file)

        X_test_lists = []
        for i in range(len(X_test)):
            X_test_lists.append([float(j) for j in X_test.values[i].flatten().tolist()])
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

    robustness, pred_c, ub, lb = robustness_report(model, X_test, y_test, ss)

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
        "X_test": X_test_lists
    }

    print(json.dumps(output))
    
