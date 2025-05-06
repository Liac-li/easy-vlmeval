from typing import Optional, List, Dict, Any


import re
import json
import sympy as sp
from sympy import simplify, Eq, sympify, Pow, pi
from sympy.parsing.latex import parse_latex
import sys
import math
# import pdb
import os
import argparse

from utils.logger import setup_logger

logger = setup_logger(__name__)

class AutoScoringJudge:
    def __init__(self):
        # Map of special symbols to their replacements
        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "厘米":"",
            # "∶": ":",
            "，": ",",
            "$": "",
            "（":"(",
            "）":")",
            "\\infty":"oo",
            "\\colon ":":",
            # "\\approx": "=",
            # "\\simeq": "=",
            # "\\sim": "=",
            # "^\\prime": "'",
            # "^{\\prime}": "'",
            "＋":"+",
            "\\, ": "",
            "\\,":"",
            "^\\circ": "",
            "^{\\circ}": "",
            # "%": "",
        }
        self.pi = parse_latex("\\pi")
        self.precision = 1e-8  # Default precision for comparison

    def trans_greater_sign_to_interval(self, expr:str):
        expr_tmp = expr.split("<")
        return "(" + expr_tmp[0] + ", "+ expr_tmp[-1] + ")"

    def split_by_comma(self, expr: str):
        # Splits expressions by commas outside of brackets
        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char in ["(", "["]:
                in_bracket_num += 1
            elif char in [")", "]"]:
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())   
        
        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list):
        # Translates plus-minus signs into separate expressions
        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)
        
        return new_expr_list
    
    def judge(self, expression1, expression2, precision=1e-8):
        # Judge if two expressions are equal (expression1 is considered as the Ground Truth)
        # Default precision is a list for supporting multiple expressions
        precision = precision if isinstance(precision, list) else [precision]

        try:
            expression1, expression2 = self.preprocess(expression1, expression2)
        except:
            return False
        if expression1 == expression2:
            # print("Exactly equal")
            return True
        
        
        # Remove Chinese characters from the string, as answers like "yes" or "no" in Chinese have been considered
        expression1 = expression1 if re.fullmatch(r"[\u4e00-\u9fff]+", expression1) else re.sub(r'[\u4e00-\u9fff]+', '', expression1)
        expression2 = expression2 if re.fullmatch(r'[\u4e00-\u9fff]+', expression2) else re.sub(r'[\u4e00-\u9fff]+', '', expression2)
        # Check if two < or > in expression
        if self.is_two_greater_sign(expression1):
            expression1 = self.trans_greater_sign_to_interval(expression1)
        
        if self.is_two_greater_sign(expression2):
            expression2 = self.trans_greater_sign_to_interval(expression2)

        expression1 = self.split_by_comma(expression1)
        expression2 = self.split_by_comma(expression2)

        temp_list1 = self.trans_plus_minus_sign(expression1)
        temp_list2 = self.trans_plus_minus_sign(expression2)

        # Set up a list for allowed errors
        if len(precision) <= 1:
            precision = precision * len(temp_list1)
        
        if len(temp_list1) != len(temp_list2):
            return False

        # Check if elements in both lists can be paired and are equal
        idx = -1
        while len(temp_list1) != 0:
            idx = (idx + 1) % len(temp_list1)

            item1 = temp_list1[idx]
            self.precision = precision[idx]

            for item2 in temp_list2:
                if self.is_equal(item1, item2):
                    temp_list1.remove(item1)
                    temp_list2.remove(item2)
                    precision.remove(self.precision)
                    break
            else:
                # If no match was found, return False
                return False

        # If all elements are matched, return True
        return True
    
    def is_interval(self, expr):
        # Checks if an expression is an interval
        return expr.startswith(("(", "[")) and expr.endswith((")", "]"))
    
    def is_two_greater_sign(self, expr):
        match = re.findall(r'<', expr)
        return len(match) == 2

    def sympy_sub_pi(self, expression_sympy):
        # Replaces the symbol for pi in sympy expressions with its numerical value
        return expression_sympy.subs(self.pi, math.pi)
    
    def is_equal(self, expression1, expression2):
        # Default first expression is ground truth. Check if expressions are equal in different aspects
        if expression1 == expression2 and expression1 != "" and expression2 != "":
            # print("Equivalent natively")
            return True
        


        # First check if both are intervals
        if self.is_interval(expression1) and self.is_interval(expression2):
            try:
                if self.interval_equal(expression1, expression2):
                    # print("Interval equivalent")
                    return True
            except:
                return False

        # Then check for numerical equality
        try:
            if self.numerical_equal(expression1, expression2):
                # print("Numerically equivalent")
                return True
        except:
            pass
        # Then check if expressions are mathematically equal
        try:
            if self.expression_equal(expression1, expression2) and not ("=" in expression1 and "=" in expression2):
                # print("Expression equivalent")
                return True
        except:
            pass
            
        # Lastly, check for equation equality
        try:
            if self.equation_equal(expression1, expression2):
                # print("Equation equivalent")
                return True
        except:
            pass
            
        return False

    def numerical_equal(self, expression1: str, expression2: str, include_percentage: bool = True):
        # Check if two numerical values are equal within an allowed error range
        # Includes possible percentage cases
        reference = float(expression1)
        prediction = float(expression2)
        
        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]
        
        for item in gt_result:
            if abs(item - prediction) <= self.precision * 1.01:
                return True
        return False
    

    def expression_equal(self, exp1, exp2):
        # Check if two expressions are mathematically equivalent
        # Extract expression and use sympy for equivalence checking
        def extract_expression(expression):
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()
        
        exp1 = extract_expression(exp1)
        exp2 = extract_expression(exp2)

        expr1_sym = sympify(parse_latex(exp1))
        expr2_sym = sympify(parse_latex(exp2))
        if expr1_sym == expr2_sym:
            return True
        else:
            expr1_sym = self.sympy_sub_pi(expr1_sym)
            expr2_sym = self.sympy_sub_pi(expr2_sym)

            if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)):
                return False
            elif not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
                try:
                    if not (self.can_compute_power(expr1_sym) and self.can_compute_power(expr2_sym)):
                        print(f"These two numbers cannot be calculated by the current computer for: \"{str(expr1_sym)}\" and \"{str(expr2_sym)}\"")
                        return False
                    if abs(expr1_sym.evalf() - expr2_sym.evalf()) <= self.precision * 1.01:
                        return True
                    else:
                        return False
                except:
                    return False
            else:
                try:
                    simplified_expr = simplify(expr1_sym - expr2_sym)

                    num_value = simplified_expr.evalf()
                    
                    return abs(num_value) < 1e-3
                except:
                    return False

    def equation_equal(self, expression1, expression2):
        # Check if two equations are mathematically equivalent
        # Simplify equations and use sympy for equivalence checking
        def simplify_equation(latex_eq):
            lhs, rhs = latex_eq.split('=')

            lhs_expr = parse_latex(lhs)
            rhs_expr = parse_latex(rhs)

            equation = Eq(lhs_expr, rhs_expr)

            simplified_eq = simplify(equation.lhs - equation.rhs)

            return simplified_eq

        expr1_sym = simplify_equation(expression1)
        expr2_sym = simplify_equation(expression2)

        division_result_1 = simplify(expr1_sym / expr2_sym)
        division_result_2 = simplify(expr2_sym / expr1_sym)

        if (division_result_1.is_Integer and division_result_1 != 0) or (division_result_2.is_Integer and division_result_2 != 0):
            return True
        else:
            return False

    def interval_equal(self, expression1, expression2):
        # Check if two intervals are mathematically equivalent
        def compare_two_interval(inter1, inter2):
            if inter1[0] != inter2[0] or inter1[-1] != inter2[-1]:
                return False
            
            inter1 = inter1.strip('[]()')
            inter2 = inter2.strip('[]()')

            items_1 = inter1.split(',')
            items_2 = inter2.split(',')

            for item_1, item_2 in zip(items_1, items_2):
                if not self.expression_equal(item_1, item_2):
                    return False
            return True
            
        interval1 = expression1
        interval2 = expression2

        if interval1 == interval2:
            return True
        else:
            inter_list1 = interval1.split("\\cup")
            inter_list2 = interval2.split("\\cup")
            
            if len(inter_list1) != len(inter_list2):
                return False
            else:
                for inter1, inter2 in zip(inter_list1, inter_list2):
                    if not compare_two_interval(inter1, inter2):
                        return False
                return True

    def preprocess(self, expression1, expression2):
        # Preprocess expressions to extract and replace special symbols
        def extract_boxed_content(latex_str):
            boxed_matches = re.finditer(r'\\boxed{', latex_str)
            results = ""

            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1

                while stack > 0 and end_index < len(latex_str):
                    if latex_str[end_index] == '{':
                        stack += 1
                    elif latex_str[end_index] == '}':
                        stack -= 1
                    end_index += 1

                if stack == 0:
                    content = latex_str[start_index:end_index - 1]
                    results += content + ","
                else:
                    raise ValueError("Mismatched braces in LaTeX string.")


            if results == "":
                last_line_ans = latex_str.strip().split("\n")[-1]
                dollar_pattern = r"\$(.*?)\$"
                answers = re.findall(dollar_pattern, last_line_ans)

                if answers:
                    for ans in answers:
                        results += ans + ","
                else:
                    results = latex_str
                
            return results
        
        def sepcial_symbol_replace(expression):

            expression = expression.replace("\\text{cm}^2", '').replace("\\text{cm}", "").replace("\\,cm", '').replace("\\text{ cm}", '').replace("cm", '').replace("\\text{分米}^2", '').replace("cm^{2}", '').replace("60 \\text{ cm}^2",'').replace("\\ \\text{m}", "").replace("\\text{米}","").strip()

            expression = re.sub(r"(.+)m$", r"\1", expression)

            if "\\in " in expression:
                expression = expression.split("\\in ")[1]
            
            for signal in self.special_signal_map:
                expression = expression.replace(signal, self.special_signal_map[signal])
            
            expression = re.sub(r'(\\sin|\\cos|\\tan)(\d+)', r'\1((\2/180)\\pi)', expression)

            # pdb.set_trace()
            expression = expression.strip("\n,.:;^_=+`!@#%^&*~，。")

            pattern = r'\\(?:mathrm|mathbf)\{~?([^}]*)\}'
            expression = re.sub(pattern, r'\1', expression)

            return expression
        
        exp1, exp2 = extract_boxed_content(expression1), extract_boxed_content(expression2)

        exp1, exp2 = sepcial_symbol_replace(exp1), sepcial_symbol_replace(exp2)

        return exp1, exp2
    
    def can_compute_power(self, expr):
        # Checks if a power expression can be computed
        if isinstance(expr, Pow):
            base, exp = expr.as_base_exp()
            if base.is_number and exp.is_number:
                MAX_EXP = 1000  # Adjust based on computing environment
                if abs(exp.evalf()) > MAX_EXP:
                    return False
                else:
                    return True
            else:
                return False
        else:
            return True  # Not a power expression, can compute

class MMMathEvaluator:
    def __init__(self, results_file_path: str=None, results: List[Dict[str, Any]]=None, precision: float = 1e-2):
        """
        初始化 MMmath 评估器
        Args:
            results_file_path: 结果文件路径
            precision: 数值比较的精度，默认为0.01
        """
        if results_file_path is None and results is None:
            raise ValueError("results_file_path or results must be provided")

        if results is not None:
            self.results = results
        else:
            with open(results_file_path, "rt", encoding="utf-8") as f:
                self.results = json.load(f)

        self.precision = precision
        self.scorer = AutoScoringJudge()

    def evaluate(self, ground_solution: str = "solution", model_solution: str = "response"):
        """
        评估模型结果
        Args:
            ground_solution: 标准答案的字段名
            model_solution: 模型预测的字段名
        Returns:
            dict: 包含评估结果和详细统计信息的字典
        """
        # 初始化统计数据
        difficult = dict()
        year = dict()
        knowledge_stats = {
            "Properties of Shapes": {"total": 0, "right": 0},
            "Transformations of Shapes": {"total": 0, "right": 0},
            "Functions": {"total": 0, "right": 0}
        }
        
        count_num = 0
        true_count = 0

        # 遍历评估每个样本
        for data in self.results:
            count_num += 1
            origin_data = data['origin_data']
            
            # 统计知识点
            for knowledge_type in knowledge_stats.keys():
                if knowledge_type in origin_data.get('knowledge', []):
                    knowledge_stats[knowledge_type]['total'] += 1

            # 统计难度
            if origin_data.get('difficult') not in difficult:
                difficult[origin_data['difficult']] = {"right": 0, 'total': 1}
            else:
                difficult[origin_data['difficult']]['total'] += 1
            
            # 统计年级
            if origin_data.get('year') not in year:
                year[origin_data['year']] = {"right": 0, 'total': 1}
            else:
                year[origin_data['year']]['total'] += 1

            # 获取答案和预测
            exp1 = origin_data.get(ground_solution, '')
            exp2 = data.get(model_solution, '')

            # 评估答案
            is_correct = False
            if "\\boxed{" in exp1 and "\\boxed{" in exp2:
                try:
                    is_correct = self.scorer.judge(exp1, exp2, self.precision)
                except:
                    is_correct = False

            if is_correct:
                true_count += 1
                year[origin_data['year']]['right'] += 1
                difficult[origin_data['difficult']]['right'] += 1
                for knowledge_type in knowledge_stats.keys():
                    if knowledge_type in data.get('knowledge', []):
                        knowledge_stats[knowledge_type]['right'] += 1

            # 将结果添加到数据中
            data['prediction_correct'] = is_correct

        # 计算总体准确率
        total_acc = true_count / (count_num + 0.0001)

        # 计算各年级的准确率
        grade_acc = {}
        for grade in ['七年级', '八年级', '九年级']:
            up = year.get(f'{grade}上学期', {'right': 0, 'total': 0})
            down = year.get(f'{grade}下学期', {'right': 0, 'total': 0})
            total = up['total'] + down['total']
            if total > 0:
                grade_acc[grade] = (up['right'] + down['right']) / total
            else:
                grade_acc[grade] = 0.0

        # 计算知识点准确率
        knowledge_acc = {k: v['right']/v['total'] if v['total'] > 0 else 0 
                        for k, v in knowledge_stats.items()}

        # 计算难度级别准确率
        difficulty_acc = {k: v['right']/v['total'] if v['total'] > 0 else 0 
                         for k, v in difficult.items()}

        # 整理评估结果
        evaluation_results = {
            "overall": {
                "accuracy": total_acc,
                "correct_count": true_count,
                "total_count": count_num
            },
            "by_grade": grade_acc,
            "by_difficulty": difficulty_acc,
            "by_knowledge": knowledge_acc
        }

        # 格式化输出结果
        logger.info("Final Evaluation Results:")
        logger.info(json.dumps(evaluation_results, indent=2, ensure_ascii=False))

        return self.results


if __name__ == "__main__":

    evaluator = MMMathEvaluator(
        results_file_path="/home/nfs05/yancy/easy-vllm/easy-vlmeval/output/checkpoint-667/mmmath_train_vllm_results_20250429_1736%.json",
    )
    results = evaluator.evaluate()
    # print(results[0])
