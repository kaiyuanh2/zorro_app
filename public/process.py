import os
import copy
import pickle
import sympy
import functools
import itertools
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

        print(latex_parts, file=sys.stderr)
        
        return latex_parts[0] + '+ \dots +' + latex_parts[1] + '+' + latex_parts[2] + '+ \dots +' + latex_parts[3] + '+' + latex_parts[4]
    
if __name__ == "__main__":
    dataset = sys.argv[1]
    test = sys.argv[2]

    if dataset == "i1":
        with open('models/ins/ins_30_model.pkl', 'rb') as file:
            model = pickle.load(file)

    latex_str = "\left[\\begin{matrix}"
    for i in range(len(model)):
        latex_str += simplify_expression(model[i])
        if i != len(model) - 1:
            latex_str += "\\\\"

    latex_str += "\end{matrix}\\right]"

    print(latex_str)
    
