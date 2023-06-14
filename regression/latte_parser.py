# imports
from fractions import Fraction
import os
# from latte_parser_relu import write_latte_likelihood


# helper functions 
def get_bounds(bd_0, bd_1):
    """
    input: bounds in str
    output: ordered pair of bounds in float
    """
    bd_0, bd_1 = float(bd_0), float(bd_1)
    x_min = bd_0 if bd_0 < bd_1 else bd_1
    x_max = bd_0 if bd_0 > bd_1 else bd_1
    return (x_min, x_max)


def create_line(length, contents):
    line = [0] * length
    for i in contents:
        line[i] = contents[i]
#     line = " ".join([str(entry) for entry in line])
    return line


def create_monomial(coefficient, exp_vec, precision=0):
    denominator = coefficient.denominator * 10**(precision * (sum(exp_vec) + 1))
    monomial = "["
    # monomial += f"{coefficient.numerator}/{coefficient.denominator},"
    monomial += f"{coefficient.numerator}/{denominator},"
    str_exp_vec = ",".join([str(exp) for exp in exp_vec])
    monomial += f"[{str_exp_vec}]"
    monomial += "]"
    return monomial


def get_idx(filename):
    filename = filename.split(".")[0]
    return filename.split("_")[1]


def write_latte_likelihood(output, weights, y, problem_idx, relu=None, folder=".", note=""):

    """
    Polytope
    """
    # basic components
    c = output['triangular']['center']['constant']
    y = output['query']
    r = output['triangular']['radius']

    const_m_y = c - y
    y_m_const = y - c
    radius_m_const_p_y = r - c + y
    # radius_m_const_m_y = r - c - y
    radius_p_const_m_y = r + c - y
    radius_p_const = r + c
    radius_m_const = r - c

    weight_indices = {w: i for i, w in enumerate(weights.keys())}

    for integral in ["numerator", "denominator"]:
        for side in ["left", "right"]:
            if integral == "numerator":
                d = len(weights)
                m = 2 * d + 2 if not relu else 2 * d + 3
                hlines = []
                # line from relu
                if relu:
                    relu_body, relu_v = relu[0], relu[1]
                    relu_coef = -1 if relu_v == 0 else 1
                    contents = {weight_indices[w] + 1: relu_coef * relu_body[w] for w in weights}
                    contents[0] = relu_coef * relu_body["constant"]
                    hlines.append(create_line(d + 1, contents))
                # line from triangular distribution
                contents = {weight_indices[w] + 1: output['triangular']['center'][w] for w in weights}
                contents[0] = const_m_y if side == "left" else radius_p_const_m_y
                hlines.append(create_line(d + 1, contents))
                
                contents = {weight_indices[w] + 1: - output['triangular']['center'][w] for w in weights}
                contents[0] = radius_m_const_p_y if side == "left" else y_m_const
                hlines.append(create_line(d + 1, contents))
                
                hlines += [create_line(d + 1, {0: - weights[w][0], weight_indices[w] + 1: 1}) for w in weights]  # lower bounds
                hlines += [create_line(d + 1, {0: weights[w][1], weight_indices[w] + 1: -1}) for w in weights]  # upper bounds

            # deno_left / deno_right
            elif integral == "denominator":
                d = len(weights) + 1
                m = 2 * d + 2 if not relu else 2 * d + 3
                hlines = []
                # line from relu
                if relu:
                    relu_body, relu_v = relu[0], relu[1]
                    relu_coef = -1 if relu_v == 0 else 1
                    contents = {weight_indices[w] + 2: relu_coef * relu_body[w] for w in weights}
                    contents[0] = relu_coef * relu_body["constant"]
                    contents[1] = 0
                    hlines.append(create_line(d + 1, contents))
                # line from triangular distribution
                contents = {weight_indices[w] + 2: output['triangular']['center'][w] for w in weights}
                contents[0] = c if side == "left" else radius_p_const
                contents[1] = -1
                
                hlines.append(create_line(d + 1, contents))
                contents = {weight_indices[w] + 2: - output['triangular']['center'][w] for w in weights}
                contents[0] = radius_m_const if side == "left" else - c
                contents[1] = 1
                hlines.append(create_line(d + 1, contents))
                
                hlines += [create_line(d + 1, {0: - weights[w][0], weight_indices[w] + 2: 1}) for w in weights]  # lower bounds
                hlines += [create_line(d + 1, {0: weights[w][1], weight_indices[w] + 2: -1}) for w in weights]  # upper bounds
                
                hlines += [create_line(d + 1, {0: - output['range'][0], 1: 1})]
                hlines += [create_line(d + 1, {0: output['range'][1], 1: -1})]
                
            assert len(hlines) == m
            # hlines = [create_line(2, {0: m, 1: d + 1})] + hlines
            # hlines = [" ".join(str(entry) for entry in hline) for hline in hlines]
            # hlines = [" ".join(str(int(entry)) for entry in hline) for hline in hlines]
            hlines = [f"{m} {d + 1}"] + [" ".join(str(round(entry * 10**PRECISION)) for entry in hline) for hline in hlines]
            latte_dir = os.path.join(folder, f"{note}{integral}_{side}_{problem_idx}")
            if not os.path.isdir(latte_dir):
                os.mkdir(latte_dir)
            f_hrep = open(os.path.join(latte_dir, POLYTOPE_TEMPLATE), 'w')
            f_hrep.writelines([hline + "\n" for hline in hlines])
            f_hrep.close()


    """
    Polynomial
    """
    # basic components
    c = output['triangular']['center']['constant']
    y = output['query']
    r = output['triangular']['radius']

    r_m_c_over_r_sqr = (r - c) / r ** 2
    r_p_c_over_r_sqr = (r + c) / r ** 2
    r_p_y_m_c_over_r_sqr = (r + y - c) / r ** 2
    r_m_y_p_c_over_r_sqr = (r - y + c) / r ** 2
    one_over_r_sqr = 1 / r ** 2

    for integral in ["numerator", "denominator"]:
        for side in ["left", "right"]:
            monomials = []
            if integral == "denominator":
                d = len(weights) + 1
                # constant monomial
                coefficient = Fraction(r_m_c_over_r_sqr) if side == "left" else Fraction(r_p_c_over_r_sqr)
                exp_vec = create_line(d, {})
                monomials.append((coefficient, exp_vec))
                
                # output monomial
                coefficient = Fraction(one_over_r_sqr) if side == "left" else Fraction(-one_over_r_sqr)
                exp_vec = create_line(d, {0: 1})
                monomials.append((coefficient, exp_vec))
                
            elif integral == "numerator":
                d = len(weights)
                # constant monomial
                coefficient = Fraction(r_p_y_m_c_over_r_sqr) if side == "left" else Fraction(r_m_y_p_c_over_r_sqr)
                exp_vec = create_line(d, {})
                monomials.append((coefficient, exp_vec))

            # weight monomials
            for w in weights:
                coefficient = Fraction(- output['triangular']['center'][w] * one_over_r_sqr) if side == "left" else Fraction(output['triangular']['center'][w] * one_over_r_sqr)
                exp_vec = create_line(d, {weight_indices[w] + 1: 1}) if integral == "denominator" else create_line(d, {weight_indices[w]: 1})
                monomials.append((coefficient, exp_vec))

            monomials = [create_monomial(coefficient, exp_vec) for coefficient, exp_vec in monomials]
            polynomial = "[" + ",".join(monomials) + "]"

            latte_dir = os.path.join(folder, f"{note}{integral}_{side}_{problem_idx}")
            if not os.path.isdir(latte_dir):
                os.mkdir(latte_dir)
            f_poly = open(os.path.join(latte_dir, POLYNOMIAL_TEMPLATE), 'w')
            f_poly.write(polynomial)
            f_poly.close()


    """
    bounds
    """
    import csv
    weights["output"] = output["range"]
    with open(f'{FOLDER}/bounds_{problem_idx}.csv','w') as f:  # write to wmi folder
        fieldnames = ['variable', 'diff']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k, v in weights.items():
            w.writerow({'variable': k, 'diff': abs(v[1] - v[0])})


# latte input files
POLYTOPE_TEMPLATE = "polytope.hrep.latte"
POLYNOMIAL_TEMPLATE = "polynomial.latte"


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='input directory')
    args = parser.parse_args()
    FOLDER = args.dir
    likelihood_folder = os.path.join(FOLDER, "likelihood")
    if not os.path.isdir(likelihood_folder):
        os.mkdir(likelihood_folder)

    # parameters
    PRECISION = 16
    
    """
    read parameters
    """
    # indices
    LAYER = 0
    VARIABLE = 1
    DISTRIBUTION = 2

    for INPUT_FILE in os.listdir(FOLDER):

        if "input" not in INPUT_FILE:
            continue

        problem_idx = get_idx(INPUT_FILE)
        weights = dict()
        output = dict()
        f = open(os.path.join(FOLDER, INPUT_FILE), "r")
        for line in f.readlines():
            line = line.strip('\n')
            params = line.split(" ")
            layer = params[LAYER]
            var = params[VARIABLE]
            dis = params[DISTRIBUTION]
            if layer == "weight":
                assert dis == "uniform"
                weights[var] = get_bounds(params[3], params[4])
            elif layer == "output":
                if dis == "query":  # for likelihood query
                    output[dis] = float(params[3])
                elif dis == "range":
                    output[dis] = get_bounds(params[3], params[4])
                elif dis == "triangular":
                    y = dict()
                    i = 4
                    while i < len(params):
                        y[params[i]] = float(params[i + 1])
                        i += 2
                    output[dis] = {"radius": float(params[3]), "center": y}


        write_latte_likelihood(output, weights, y, problem_idx, relu=None, folder=likelihood_folder)