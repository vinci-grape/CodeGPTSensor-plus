import sys
# sys.path.append("../")
# sys.path.append("../parser")

from keyword import iskeyword
from tree_sitter import Language, Parser
import re
from io import StringIO
import tokenize
import json
import random

def remove_comments_and_docstrings(source, lang):
    if lang in ["python"]:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += " " * (start_col - last_col)
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split("\n"):
            if x.strip() != "":
                temp.append(x)
        return "\n".join(temp)
    elif lang in ["ruby"]:
        return source
    else:

        def replacer(match):
            s = match.group(0)
            if s.startswith("/"):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE,
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split("\n"):
            if x.strip() != "":
                temp.append(x)
        return "\n".join(temp)


def tree_to_token_index(root_node):
    if (
        len(root_node.children) == 0 or root_node.type == "string"
    ) and root_node.type != "comment":
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def tree_to_variable_index(root_node, index_to_code):
    if (
        len(root_node.children) == 0 or root_node.type == "string"
    ) and root_node.type != "comment":
        index = (root_node.start_point, root_node.end_point)
        _, code = index_to_code[index]
        if root_node.type != code:
            return [(root_node.start_point, root_node.end_point)]
        else:
            return []
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_variable_index(child, index_to_code)
        return code_tokens


def index_to_code_token(index, code):
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1] : end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1] :]
        for i in range(start_point[0] + 1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][: end_point[1]]
    return s


def DFG_python(root_node, index_to_code, states):
    assignment = ["assignment", "augmented_assignment", "for_in_clause"]
    if_statement = ["if_statement"]
    for_statement = ["for_statement"]
    while_statement = ["while_statement"]
    do_first_statement = ["for_in_clause"]
    def_statement = ["default_parameter"]
    states = states.copy()
    if (
        len(root_node.children) == 0 or root_node.type == "string"
    ) and root_node.type != "comment":
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, idx, "comesFrom", [code], states[code].copy())], states
        else:
            if root_node.type == "identifier":
                states[code] = [idx]
            return [(code, idx, "comesFrom", [], [])], states
    elif root_node.type in def_statement:
        name = root_node.child_by_field_name("name")
        value = root_node.child_by_field_name("value")
        DFG = []
        if value is None:
            indexs = tree_to_variable_index(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                DFG.append((code, idx, "comesFrom", [], []))
                states[code] = [idx]
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = DFG_python(value, index_to_code, states)
            DFG += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, "comesFrom", [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in assignment:
        if root_node.type == "for_in_clause":
            right_nodes = [root_node.children[-1]]
            left_nodes = [root_node.child_by_field_name("left")]
        else:
            if root_node.child_by_field_name("right") is None:
                return [], states
            left_nodes = [
                x
                for x in root_node.child_by_field_name("left").children
                if x.type != ","
            ]
            right_nodes = [
                x
                for x in root_node.child_by_field_name("right").children
                if x.type != ","
            ]
            if len(right_nodes) != len(left_nodes):
                left_nodes = [root_node.child_by_field_name("left")]
                right_nodes = [root_node.child_by_field_name("right")]
            if len(left_nodes) == 0:
                left_nodes = [root_node.child_by_field_name("left")]
            if len(right_nodes) == 0:
                right_nodes = [root_node.child_by_field_name("right")]
        DFG = []
        for node in right_nodes:
            temp, states = DFG_python(node, index_to_code, states)
            DFG += temp

        for left_node, right_node in zip(left_nodes, right_nodes):
            left_tokens_index = tree_to_variable_index(left_node, index_to_code)
            right_tokens_index = tree_to_variable_index(right_node, index_to_code)
            temp = []
            for token1_index in left_tokens_index:
                idx1, code1 = index_to_code[token1_index]
                temp.append(
                    (
                        code1,
                        idx1,
                        "computedFrom",
                        [index_to_code[x][1] for x in right_tokens_index],
                        [index_to_code[x][0] for x in right_tokens_index],
                    )
                )
                states[code1] = [idx1]
            DFG += temp
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in if_statement:
        DFG = []
        current_states = states.copy()
        others_states = []
        tag = False
        if "else" in root_node.type:
            tag = True
        for child in root_node.children:
            if "else" in child.type:
                tag = True
            if child.type not in ["elif_clause", "else_clause"]:
                temp, current_states = DFG_python(child, index_to_code, current_states)
                DFG += temp
            else:
                temp, new_states = DFG_python(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(DFG, key=lambda x: x[1]), new_states
    elif root_node.type in for_statement:
        DFG = []
        for i in range(2):
            right_nodes = [
                x
                for x in root_node.child_by_field_name("right").children
                if x.type != ","
            ]
            left_nodes = [
                x
                for x in root_node.child_by_field_name("left").children
                if x.type != ","
            ]
            if len(right_nodes) != len(left_nodes):
                left_nodes = [root_node.child_by_field_name("left")]
                right_nodes = [root_node.child_by_field_name("right")]
            if len(left_nodes) == 0:
                left_nodes = [root_node.child_by_field_name("left")]
            if len(right_nodes) == 0:
                right_nodes = [root_node.child_by_field_name("right")]
            for node in right_nodes:
                temp, states = DFG_python(node, index_to_code, states)
                DFG += temp
            for left_node, right_node in zip(left_nodes, right_nodes):
                left_tokens_index = tree_to_variable_index(left_node, index_to_code)
                right_tokens_index = tree_to_variable_index(right_node, index_to_code)
                temp = []
                for token1_index in left_tokens_index:
                    idx1, code1 = index_to_code[token1_index]
                    temp.append(
                        (
                            code1,
                            idx1,
                            "computedFrom",
                            [index_to_code[x][1] for x in right_tokens_index],
                            [index_to_code[x][0] for x in right_tokens_index],
                        )
                    )
                    states[code1] = [idx1]
                DFG += temp
            if root_node.children[-1].type == "block":
                temp, states = DFG_python(root_node.children[-1], index_to_code, states)
                DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0] + x[3])
                )
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1] + x[4]))
                )
        DFG = [
            (x[0], x[1], x[2], y[0], y[1])
            for x, y in sorted(dic.items(), key=lambda t: t[0][1])
        ]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in while_statement:
        DFG = []
        for i in range(2):
            for child in root_node.children:
                temp, states = DFG_python(child, index_to_code, states)
                DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0] + x[3])
                )
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1] + x[4]))
                )
        DFG = [
            (x[0], x[1], x[2], y[0], y[1])
            for x, y in sorted(dic.items(), key=lambda t: t[0][1])
        ]
        return sorted(DFG, key=lambda x: x[1]), states
    else:
        DFG = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = DFG_python(child, index_to_code, states)
                DFG += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = DFG_python(child, index_to_code, states)
                DFG += temp

        return sorted(DFG, key=lambda x: x[1]), states


def DFG_java(root_node, index_to_code, states):
    assignment = ["assignment_expression"]
    def_statement = ["variable_declarator"]
    increment_statement = ["update_expression"]
    if_statement = ["if_statement", "else"]
    for_statement = ["for_statement"]
    enhanced_for_statement = ["enhanced_for_statement"]
    while_statement = ["while_statement"]
    do_first_statement = []
    states = states.copy()
    if (
        len(root_node.children) == 0 or root_node.type == "string"
    ) and root_node.type != "comment":
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, idx, "comesFrom", [code], states[code].copy())], states
        else:
            if root_node.type == "identifier":
                states[code] = [idx]
            return [(code, idx, "comesFrom", [], [])], states
    elif root_node.type in def_statement:
        name = root_node.child_by_field_name("name")
        value = root_node.child_by_field_name("value")
        DFG = []
        if value is None:
            indexs = tree_to_variable_index(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                DFG.append((code, idx, "comesFrom", [], []))
                states[code] = [idx]
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = DFG_java(value, index_to_code, states)
            DFG += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, "comesFrom", [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name("left")
        right_nodes = root_node.child_by_field_name("right")
        DFG = []
        temp, states = DFG_java(right_nodes, index_to_code, states)
        DFG += temp
        name_indexs = tree_to_variable_index(left_nodes, index_to_code)
        value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, "computedFrom", [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in increment_statement:
        DFG = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, "computedFrom", [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in if_statement:
        DFG = []
        current_states = states.copy()
        others_states = []
        flag = False
        tag = False
        if "else" in root_node.type:
            tag = True
        for child in root_node.children:
            if "else" in child.type:
                tag = True
            if child.type not in if_statement and flag is False:
                temp, current_states = DFG_java(child, index_to_code, current_states)
                DFG += temp
            else:
                flag = True
                temp, new_states = DFG_java(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(DFG, key=lambda x: x[1]), new_states
    elif root_node.type in for_statement:
        DFG = []
        for child in root_node.children:
            temp, states = DFG_java(child, index_to_code, states)
            DFG += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp
            elif child.type == "local_variable_declaration":
                flag = True
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0] + x[3])
                )
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1] + x[4]))
                )
        DFG = [
            (x[0], x[1], x[2], y[0], y[1])
            for x, y in sorted(dic.items(), key=lambda t: t[0][1])
        ]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in enhanced_for_statement:
        name = root_node.child_by_field_name("name")
        value = root_node.child_by_field_name("value")
        body = root_node.child_by_field_name("body")
        DFG = []
        for i in range(2):
            temp, states = DFG_java(value, index_to_code, states)
            DFG += temp
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, "computedFrom", [code2], [idx2]))
                states[code1] = [idx1]
            temp, states = DFG_java(body, index_to_code, states)
            DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0] + x[3])
                )
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1] + x[4]))
                )
        DFG = [
            (x[0], x[1], x[2], y[0], y[1])
            for x, y in sorted(dic.items(), key=lambda t: t[0][1])
        ]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in while_statement:
        DFG = []
        for i in range(2):
            for child in root_node.children:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0] + x[3])
                )
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1] + x[4]))
                )
        DFG = [
            (x[0], x[1], x[2], y[0], y[1])
            for x, y in sorted(dic.items(), key=lambda t: t[0][1])
        ]
        return sorted(DFG, key=lambda x: x[1]), states
    else:
        DFG = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp

        return sorted(DFG, key=lambda x: x[1]), states

dfg_function = {
    # "c": DFG_c,
    "python": DFG_python,
    "java": DFG_java,
    # "ruby": DFG_ruby,
    # "go": DFG_go,
    # "php": DFG_php,
    # "javascript": DFG_javascript,
    # "csharp": DFG_csharp,
}

parsers = {}
for lang in dfg_function:
    LANGUAGE = Language("/data2/xiaodanxu/zzz_attack/MOAA/parser/my-languages.so", lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser

python_keywords = [
    "import",
    "",
    "[",
    "]",
    ":",
    ",",
    ".",
    "(",
    ")",
    "{",
    "}",
    "not",
    "is",
    "=",
    "+=",
    "-=",
    "<",
    ">",
    "+",
    "-",
    "*",
    "/",
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
]
java_keywords = [
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "throws",
    "transient",
    "try",
    "void",
    "volatile",
    "while",
]
java_special_ids = [
    "main",
    "args",
    "Math",
    "System",
    "Random",
    "Byte",
    "Short",
    "Integer",
    "Long",
    "Float",
    "Double",
    "Character",
    "Boolean",
    "Data",
    "ParseException",
    "SimpleDateFormat",
    "Calendar",
    "Object",
    "String",
    "StringBuffer",
    "StringBuilder",
    "DateFormat",
    "Collection",
    "List",
    "Map",
    "Set",
    "Queue",
    "ArrayList",
    "HashSet",
    "HashMap",
]
c_keywords = [
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Alignas",
    "_Alignof",
    "_Atomic",
    "_Bool",
    "_Complex",
    "_Generic",
    "_Imaginary",
    "_Noreturn",
    "_Static_assert",
    "_Thread_local",
    "__func__",
]

c_macros = [
    "NULL",
    "_IOFBF",
    "_IOLBF",
    "BUFSIZ",
    "EOF",
    "FOPEN_MAX",
    "TMP_MAX",  # <stdio.h> macro
    "FILENAME_MAX",
    "L_tmpnam",
    "SEEK_CUR",
    "SEEK_END",
    "SEEK_SET",
    "NULL",
    "EXIT_FAILURE",
    "EXIT_SUCCESS",
    "RAND_MAX",
    "MB_CUR_MAX",
]  # <stdlib.h> macro
c_special_ids = [
    "main",  # main function
    "stdio",
    "cstdio",
    "stdio.h",  # <stdio.h> & <cstdio>
    "size_t",
    "FILE",
    "fpos_t",
    "stdin",
    "stdout",
    "stderr",  # <stdio.h> types & streams
    "remove",
    "rename",
    "tmpfile",
    "tmpnam",
    "fclose",
    "fflush",  # <stdio.h> functions
    "fopen",
    "freopen",
    "setbuf",
    "setvbuf",
    "fprintf",
    "fscanf",
    "printf",
    "scanf",
    "snprintf",
    "sprintf",
    "sscanf",
    "vprintf",
    "vscanf",
    "vsnprintf",
    "vsprintf",
    "vsscanf",
    "fgetc",
    "fgets",
    "fputc",
    "getc",
    "getchar",
    "putc",
    "putchar",
    "puts",
    "ungetc",
    "fread",
    "fwrite",
    "fgetpos",
    "fseek",
    "fsetpos",
    "ftell",
    "rewind",
    "clearerr",
    "feof",
    "ferror",
    "perror",
    "getline" "stdlib",
    "cstdlib",
    "stdlib.h",  # <stdlib.h> & <cstdlib>
    "size_t",
    "div_t",
    "ldiv_t",
    "lldiv_t",  # <stdlib.h> types
    "atof",
    "atoi",
    "atol",
    "atoll",
    "strtod",
    "strtof",
    "strtold",  # <stdlib.h> functions
    "strtol",
    "strtoll",
    "strtoul",
    "strtoull",
    "rand",
    "srand",
    "aligned_alloc",
    "calloc",
    "malloc",
    "realloc",
    "free",
    "abort",
    "atexit",
    "exit",
    "at_quick_exit",
    "_Exit",
    "getenv",
    "quick_exit",
    "system",
    "bsearch",
    "qsort",
    "abs",
    "labs",
    "llabs",
    "div",
    "ldiv",
    "lldiv",
    "mblen",
    "mbtowc",
    "wctomb",
    "mbstowcs",
    "wcstombs",
    "string",
    "cstring",
    "string.h",  # <string.h> & <cstring>
    "memcpy",
    "memmove",
    "memchr",
    "memcmp",
    "memset",
    "strcat",  # <string.h> functions
    "strncat",
    "strchr",
    "strrchr",
    "strcmp",
    "strncmp",
    "strcoll",
    "strcpy",
    "strncpy",
    "strerror",
    "strlen",
    "strspn",
    "strcspn",
    "strpbrk",
    "strstr",
    "strtok",
    "strxfrm",
    "memccpy",
    "mempcpy",
    "strcat_s",
    "strcpy_s",
    "strdup",  # <string.h> extension functions
    "strerror_r",
    "strlcat",
    "strlcpy",
    "strsignal",
    "strtok_r",
    "iostream",
    "istream",
    "ostream",
    "fstream",
    "sstream",  # <iostream> family
    "iomanip",
    "iosfwd",
    "ios",
    "wios",
    "streamoff",
    "streampos",
    "wstreampos",  # <iostream> types
    "streamsize",
    "cout",
    "cerr",
    "clog",
    "cin",
    "boolalpha",
    "noboolalpha",
    "skipws",
    "noskipws",
    "showbase",  # <iostream> manipulators
    "noshowbase",
    "showpoint",
    "noshowpoint",
    "showpos",
    "noshowpos",
    "unitbuf",
    "nounitbuf",
    "uppercase",
    "nouppercase",
    "left",
    "right",
    "internal",
    "dec",
    "oct",
    "hex",
    "fixed",
    "scientific",
    "hexfloat",
    "defaultfloat",
    "width",
    "fill",
    "precision",
    "endl",
    "ends",
    "flush",
    "ws",
    "showpoint",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",  # <math.h> functions
    "cosh",
    "tanh",
    "exp",
    "sqrt",
    "log",
    "log10",
    "pow",
    "powf",
    "ceil",
    "floor",
    "abs",
    "fabs",
    "cabs",
    "frexp",
    "ldexp",
    "modf",
    "fmod",
    "hypot",
    "ldexp",
    "poly",
    "matherr",
]

go_keywords = [
    "break",
    "default",
    "func",
    "interface",
    "select",
    "case",
    "defer",
    "go",
    "map",
    "struct",
    "chan",
    "else",
    "goto",
    "package",
    "switch",
    "const",
    "fallthrough",
    "if",
    "range",
    "type",
    "continue",
    "for",
    "import",
    "return",
    "var",
]

go_predeclared_types = [
    "bool",
    "byte",
    "complex64",
    "complex128",
    "error",
    "float32",
    "float64",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "rune",
    "string",
    "uint",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uintptr",
]

go_predeclared_constants = ["true", "false", "iota", "nil"]

go_zero_value_identifiers = ["nil"]

go_predeclared_functions = [
    "append",
    "cap",
    "close",
    "complex",
    "copy",
    "delete",
    "imag",
    "len",
    "make",
    "new",
    "panic",
    "print",
    "println",
    "real",
    "recover",
]

js_keywords = [
    "abstract",
    "arguments",
    "await",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "debugger",
    "default",
    "delete",
    "do",
    "double",
    "else",
    "enum",
    "eval",
    "export",
    "extends",
    "false",
    "final",
    "finally",
    "float",
    "for",
    "function",
    "goto",
    "if",
    "implements",
    "import",
    "in",
    "instanceof",
    "int",
    "interface",
    "let",
    "long",
    "native",
    "new",
    "null",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "true",
    "try",
    "typeof",
    "var",
    "void",
    "volatile",
    "while",
    "with",
    "yield",
]


js_fundamental_objects = [
    "Object",
    "Function",
    "Boolean",
    "Symbol",
    "Error",
    "EvalError",
    "InternalError",
    "RangeError",
    "ReferenceError",
    "SyntaxError",
    "TypeError",
    "URIError",
]

js_numbers_dates = ["Number", "Math", "Date"]

js_text_processing = ["String", "RegExp"]

js_indexed_collections = [
    "Array",
    "Int8Array",
    "Uint8Array",
    "Uint8ClampedArray",
    "Int16Array",
    "Uint16Array",
    "Int32Array",
    "Uint32Array",
    "Float32Array",
    "Float64Array",
]

js_keyed_collections = ["Map", "Set", "WeakMap", "WeakSet"]

js_structured_data = ["JSON"]

js_control_abstraction_objects = [
    "Promise",
    "Generator",
    "GeneratorFunction",
    "AsyncFunction",
]

js_reflection = ["Reflect", "Proxy"]

js_global_functions = [
    "eval",
    "isFinite",
    "isNaN",
    "parseFloat",
    "parseInt",
    "decodeURI",
    "decodeURIComponent",
    "encodeURI",
    "encodeURIComponent",
]

php_keywords = [
    "__halt_compiler",
    "abstract",
    "and",
    "array",
    "as",
    "break",
    "callable",
    "case",
    "catch",
    "class",
    "clone",
    "const",
    "continue",
    "declare",
    "default",
    "die",
    "do",
    "echo",
    "else",
    "elseif",
    "empty",
    "enddeclare",
    "endfor",
    "endforeach",
    "endif",
    "endswitch",
    "endwhile",
    "eval",
    "exit",
    "extends",
    "final",
    "finally",
    "fn",
    "for",
    "foreach",
    "function",
    "global",
    "goto",
    "if",
    "implements",
    "include",
    "include_once",
    "instanceof",
    "insteadof",
    "interface",
    "isset",
    "list",
    "namespace",
    "new",
    "or",
    "print",
    "private",
    "protected",
    "public",
    "require",
    "require_once",
    "return",
    "static",
    "switch",
    "throw",
    "trait",
    "try",
    "unset",
    "use",
    "var",
    "while",
    "xor",
    "yield",
]

php_string_functions = [
    "addslashes",
    "chop",
    "explode",
    "htmlspecialchars",
    "strlen",
    "strpos",
    "str_replace",
    "str_split",
    "strtolower",
    "strtoupper",
    "trim",
]


php_array_functions = [
    "array_combine",
    "array_count_values",
    "array_diff",
    "array_filter",
    "array_map",
    "array_merge",
    "array_pop",
    "array_push",
    "array_slice",
    "array_unique",
    "sort",
]

php_file_system_functions = [
    "fclose",
    "feof",
    "fgets",
    "fopen",
    "fread",
    "fwrite",
    "glob",
    "is_dir",
    "is_file",
    "mkdir",
    "rename",
    "rmdir",
    "unlink",
]

php_date_time_functions = [
    "date",
    "date_create",
    "date_diff",
    "date_format",
    "getdate",
    "strtotime",
]

php_miscellaneous_functions = [
    "die",
    "empty",
    "eval",
    "exit",
    "isset",
    "print_r",
    "var_dump",
]

php_spl_classes = [
    "ArrayObject",
    "DirectoryIterator",
    "Exception",
    "FileNotFoundException",
    "GlobIterator",
    "InvalidArgumentException",
    "IteratorAggregate",
    "PDOException",
    "RuntimeException",
    "SplDoublyLinkedList",
    "SplFileInfo",
    "SplFileObject",
    "SplFixedArray",
    "SplHeap",
    "SplMinHeap",
    "SplMaxHeap",
    "SplObjectStorage",
    "SplPriorityQueue",
    "SplQueue",
    "SplStack",
    "SplTempFileObject",
]


ruby_keywords = [
    "__ENCODING__",
    "__LINE__",
    "__FILE__",
    "BEGIN",
    "END",
    "alias",
    "and",
    "begin",
    "break",
    "case",
    "class",
    "def",
    "defined?",
    "do",
    "else",
    "elsif",
    "end",
    "ensure",
    "false",
    "for",
    "if",
    "in",
    "module",
    "next",
    "nil",
    "not",
    "or",
    "redo",
    "rescue",
    "retry",
    "return",
    "self",
    "super",
    "then",
    "true",
    "undef",
    "unless",
    "until",
    "when",
    "while",
    "yield",
]

ruby_basic_classes = [
    "Array",
    "Hash",
    "String",
    "Symbol",
    "Numeric",
    "Integer",
    "Float",
    "Range",
    "Regexp",
    "Proc",
    "Lambda",
    "Thread",
    "Exception",
    "StandardError",
    "SystemExit",
    "IO",
    "File",
]


ruby_enumerable_methods = [
    "each",
    "map",
    "select",
    "reject",
    "grep",
    "find",
    "find_all",
    "reduce",
    "any?",
    "all?",
    "none?",
    "count",
    "sort",
    "sort_by",
]


ruby_numeric_classes = [
    "Numeric",
    "Integer",
    "Fixnum",
    "Bignum",
    "Float",
    "Rational",
    "Complex",
]

ruby_math_methods = ["sqrt", "log", "sin", "cos", "tan"]


ruby_io_classes = ["IO", "File", "Dir", "StringIO"]

ruby_io_methods = ["puts", "gets", "print", "printf", "read", "write", "open", "close"]


ruby_networking_classes = ["Socket", "TCPSocket", "TCPServer", "UDPSocket"]

csharp_keywords = [
    "bool",
    "byte",
    "sbyte",
    "char",
    "decimal",
    "double",
    "float",
    "int",
    "uint",
    "long",
    "ulong",
    "object",
    "dynamic",
    "short",
    "ushort",
    "string",
    "void",
    "as",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "default",
    "delegate",
    "do",
    "else",
    "enum",
    "event",
    "explicit",
    "extern",
    "false",
    "finally",
    "fixed",
    "for",
    "foreach",
    "goto",
    "if",
    "implicit",
    "in",
    "interface",
    "internal",
    "is",
    "lock",
    "namespace",
    "new",
    "null",
    "operator",
    "out",
    "override",
    "params",
    "private",
    "protected",
    "public",
    "readonly",
    "ref",
    "return",
    "sealed",
    "sizeof",
    "stackalloc",
    "static",
    "struct",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typeof",
    "unchecked",
    "unsafe",
    "using",
    "virtual",
    "volatile",
    "while",
    "abstract",
    "async",
    "await",
    "checked",
    "partial",
    "yield",
]

system_classes = ["Console", "Math", "DateTime", "Guid", "Random", "Convert"]

collections_generic = [
    "List<T>",
    "Dictionary<TKey, TValue>",
    "HashSet<T>",
    "Queue<T>",
    "Stack<T>",
]

system_io_classes = [
    "File",
    "FileInfo",
    "Directory",
    "DirectoryInfo",
    "FileStream",
    "StreamReader",
    "StreamWriter",
]

system_text_classes = ["StringBuilder", "Encoding"]

system_net_classes = ["WebClient", "HttpWebRequest", "HttpWebResponse", "Socket"]

system_threading_classes = ["Thread", "ThreadPool", "Mutex", "Semaphore"]

system_threading_tasks = ["Task", "Task<T>"]

system_linq_classes = ["Enumerable", "Queryable"]


special_char = [
    "[",
    "]",
    ":",
    ",",
    ".",
    "(",
    ")",
    "{",
    "}",
    "not",
    "is",
    "=",
    "+=",
    "-=",
    "<",
    ">",
    "+",
    "-",
    "*",
    "/",
    "|",
]


def is_valid_variable_python(name: str) -> bool:
    return name.isidentifier() and not iskeyword(name)


def is_valid_variable_java(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in java_keywords:
        return False
    elif name in java_special_ids:
        return False
    return True


def is_valid_variable_c(name: str) -> bool:

    if not name.isidentifier():
        return False
    elif name in c_keywords:
        return False
    elif name in c_macros:
        return False
    elif name in c_special_ids:
        return False
    return True


def is_valid_variable_go(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in go_keywords:
        return False
    elif name in go_predeclared_types:
        return False
    elif name in go_predeclared_constants:
        return False
    elif name in go_zero_value_identifiers:
        return False
    elif name in go_predeclared_functions:
        return False
    return True


def is_valid_variable_js(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in js_keywords:
        return False
    elif name in js_fundamental_objects:
        return False
    elif name in js_numbers_dates:
        return False
    elif name in js_text_processing:
        return False
    elif name in js_indexed_collections:
        return False
    elif name in js_keyed_collections:
        return False
    elif name in js_structured_data:
        return False
    elif name in js_control_abstraction_objects:
        return False
    elif name in js_reflection:
        return False
    elif name in js_global_functions:
        return False
    return True


def is_valid_variable_php(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in php_keywords:
        return False
    elif name in php_string_functions:
        return False
    elif name in php_array_functions:
        return False
    elif name in php_file_system_functions:
        return False
    elif name in php_date_time_functions:
        return False
    elif name in php_miscellaneous_functions:
        return False
    elif name in php_spl_classes:
        return False
    return True


def is_valid_variable_ruby(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in ruby_keywords:
        return False
    elif name in ruby_basic_classes:
        return False
    elif name in ruby_enumerable_methods:
        return False
    elif name in ruby_numeric_classes:
        return False
    elif name in ruby_math_methods:
        return False
    elif name in ruby_io_classes:
        return False
    elif name in ruby_io_methods:
        return False
    elif name in ruby_networking_classes:
        return False
    return True


def is_valid_variable_csharp(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in csharp_keywords:
        return False
    elif name in system_classes:
        return False
    elif name in collections_generic:
        return False
    elif name in system_io_classes:
        return False
    elif name in system_text_classes:
        return False
    elif name in system_net_classes:
        return False
    elif name in system_threading_classes:
        return False
    elif name in system_threading_tasks:
        return False
    elif name in system_linq_classes:
        return False
    return True


def is_valid_variable_name(name: str, lang: str) -> bool:
    # check if matches language keywords
    if lang == "python":
        return is_valid_variable_python(name)
    elif lang == "c":
        return is_valid_variable_c(name)
    elif lang == "java":
        return is_valid_variable_java(name)
    elif lang == "go":
        return is_valid_variable_go(name)
    elif lang == "javascript":
        return is_valid_variable_js(name)
    elif lang == "php":
        return is_valid_variable_php(name)
    elif lang == "ruby":
        return is_valid_variable_ruby(name)
    elif lang == "csharp":
        return is_valid_variable_csharp(name)
    else:
        return False


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def get_identifiers(code, lang):

    dfg, index_table, code_tokens = extract_dataflow(code, lang)
    ret = []
    for d in dfg:
        if is_valid_variable_name(d[0], lang):
            ret.append(d[0])
    ret = unique(ret)
    ret = [[i] for i in ret]
    return ret, code_tokens


def extract_dataflow(code, lang):
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    tree = parser[0].parse(bytes(code, "utf8"))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split("\n")
    # print(code)
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    index_to_code = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code)

    index_table = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_table[idx] = index

    DFG, _ = parser[1](root_node, index_to_code, {})

    DFG = sorted(DFG, key=lambda x: x[1])
    return DFG, index_table, code_tokens



def get_example_batch(code, chromesome, lang):
    # print('get_example_batch!', flush=True)
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    replace_pos = {}
    for tgt_word in chromesome.keys():
        diff = len(chromesome[tgt_word]) - len(tgt_word)
        for index, code_token in enumerate(code_tokens):
            if code_token == tgt_word:
                try:
                    replace_pos[tokens_index[index][0][0]].append((tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1]))
                except:
                    replace_pos[tokens_index[index][0][0]] = [(tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1])]
    for line in replace_pos.keys():
        diff = 0
        for index, pos in enumerate(replace_pos[line]):
            code[line] = code[line][:pos[3]+diff] + pos[1] + code[line][pos[4]+diff:]
            diff += pos[2]
    
    # print('done get_example_batch!', flush=True)

    return "\n".join(code)


def my_change_code_style(source_code: str, lang: str, orig_label: int):
    all_source_codes = [source_code]
    code_style_dic = {}
    code_style = []
    extra_words = []
    with open(f'/data2/xiaodanxu/zzz_attack/style/code/style/{lang}_code_style.json', 'r') as f:
        code_style_dic = json.load(f)
    if orig_label == 0:
        code_style = code_style_dic['gpt']
    else:
        code_style = code_style_dic['human']
    
    # while <-> for
    if lang in ['java']:
        source_code_list = source_code.split('\n')
        new_source_code_list = []
        if random.random() < code_style[4][0]: # code_style[4][0] 为 while 循环的数量占比p，以p的概率把for改成while
            # for -> while
            i = 0
            while i < len(source_code_list):
                line = source_code_list[i]
                if line.find('for') != -1:
                    condition = line.split('for')[1].strip()
                    # print('line', line, flush=True)
                    # print('condition', condition, flush=True)
                    # print('condition[0]', condition[0], flush=True)
                    # 检查是否是标准for循环格式: for(init; condition; increment)  有(;;)，长度至少为4
                    if len(condition) > 3 and condition[0] == '(' and len(condition.split(';')) == 3 and line.strip()[-1] == '{':
                        count_blank = 0
                        for char in line:
                            if char == ' ':
                                count_blank += 1
                            else:
                                break
                        prefix_blank = ' ' * count_blank

                        # 提取for循环的三个部分
                        init = condition.split(';')[0].split('(')[-1].strip()
                        test = condition.split(';')[1].strip()
                        increment = condition.split(';')[2].split(')')[0].strip()

                        if len(init) > 0:
                            new_source_code_list.append(prefix_blank + init + ';')
                        
                        new_source_code_list.append(prefix_blank + 'while (' + test +') {')
                        
                        start = 1
                        j = i + 1
                        while j < len(source_code_list):
                            new_source_code_list.append(source_code_list[j])
                            start += source_code_list[j].count('{')
                            start -= source_code_list[j].count('}')
                            j += 1
                            if start == 0:
                                break
                        
                        count_blank = 0
                        for char in new_source_code_list[-2]:
                            if char == ' ':
                                count_blank += 1
                            else:
                                break

                        if len(increment) > 0:
                            new_source_code_list[-1] = ' ' * count_blank+ increment + ';'
                            new_source_code_list.append(prefix_blank + '}')
                        i = j
                    else:
                        new_source_code_list.append(line)
                        i += 1
                else:
                    new_source_code_list.append(line)
                    i += 1
        else:
            # while -> for
            i = 0
            while i < len(source_code_list):
                line = source_code_list[i]
                if line.find('while') != -1 and line.strip()[-1] == '{':
                    count_blank = 0
                    for char in line:
                        if char == ' ':
                            count_blank += 1
                        else:
                            break
                    prefix_blank = ' ' * count_blank
                    try:
                        new_source_code_list.append(prefix_blank +'for (;' + line.split('while')[1].split('(')[1].split(')')[0].strip() + ';) {')
                    except:
                        new_source_code_list.append(line)
                else:
                    new_source_code_list.append(line)
                i += 1
        
        new_source_code = '\n'.join(new_source_code_list)
        all_source_codes.append(new_source_code)
    
    # 在上一个mutation的基础上进行
    source_code = all_source_codes[-1]
    # System.out.println("xxx"); <-> a = "xxx"; System.out.println(a);
    if random.random() < code_style[0][1]:
        # print('code_style[0], change print!', flush=True)
        all_variable_name = ['msg', 'message', 'tmp', 'temp', 't', 'text', 'val', 'data', 'result', 'output', 's', 'out', 'o', 'info', 'debug_text', 'debug_msg', 'debug_info','content', 'display_text', 'display_msg', 'display_info']
        used_identifiers, _ = get_identifiers(source_code, lang)
        # print('used_identifiers', used_identifiers, flush=True)
        used_identifiers = [item[0] for item in used_identifiers]
        all_variable_name = list(set(all_variable_name) - (set(all_variable_name) & set(used_identifiers)))
        random.shuffle(all_variable_name)
        if lang == 'java':
            all_java_print = ['System.out.println', 'System.out.print']
            for java_print in all_java_print:
                begin = 0
                end = 0
                while source_code.find(java_print+'("', end) != -1:
                    begin = source_code.find(java_print+'("', end)
                    end = source_code.find('")', begin)
                    if end == -1:
                        break
                    pre_code = source_code[:begin]
                    string = source_code[begin + len(java_print)+1:end+1]
                    after_code = source_code[end + 2:]

                    source_code = pre_code + 'String ' + all_variable_name[0] + ' = ' + string + '; ' + \
                                  java_print + '(' + all_variable_name[0] + ')' + after_code
                    extra_words.append(all_variable_name[0])
                    if len(all_variable_name) > 1:
                        all_variable_name = all_variable_name[1:]
                    else:
                        all_variable_name[0] += '_'
        elif lang == 'python':
            source_code_list = source_code.split('\n')
            new_source_code_list = []
            for line in source_code_list:
                temp_line = line.strip()
                is_skip = False
                # if len(temp_line) <= 5:
                if len(temp_line) < 7: # print("  至少7个字符 
                    is_skip = True
                for c in ['%', 'sep=', 'format', 'file=', 'end=', 'flush=', '"""']: # 排除print('%s' % xxx)等情况
                    if c in temp_line:
                        is_skip = True

                if is_skip:
                    new_source_code_list.append(line)
                    continue

                if temp_line[:5] == 'print':
                    blank = line[:line.find('print')]
                    
                    def help_solver(content):
                        tmp = content[:]
                        tmp = tmp.replace(' ', '')
                        if '"' in tmp or "'" in tmp:
                            if not ('",' in tmp or ',"' in tmp or "'," in tmp or ",'" in tmp): # 逗号在字符串内部
                                return content
                        items = []
                        for item in content.split(','):
                            if item:
                                item = item.strip()
                                if len(item) >=2 and (item[0] == '"' and item[-1] == '"') or (item[0] == "'" and item[-1] == "'"):
                                    items.append(item)
                                elif len(item) > 0:
                                    items.append('str(%s)' % item)
                        content = ' + '.join(items)
                        return content

                    if temp_line[5] == '(' and temp_line[6] in ['"', "'"]:
                        left_bracket_index = temp_line.find('(')
                        right_bracket_index = temp_line.rfind(')')
                        content = temp_line[left_bracket_index+1:right_bracket_index]
                        if content.find(',') != -1:
                            content = help_solver(content)

                        new_source_code_list.append(blank + '%s = %s' % (all_variable_name[0], content))
                        new_source_code_list.append(blank + 'print(%s)' % all_variable_name[0])
                        extra_words.append(all_variable_name[0])
                        if len(all_variable_name) > 1:
                            all_variable_name = all_variable_name[1:]
                        else:
                            all_variable_name[0] += '_'
                    elif temp_line[5] == ' ' and temp_line[6] in ['"', "'"]:
                        content = temp_line[6:]
                        if content.find(',') != -1:
                            content = help_solver(content)
                        new_source_code_list.append(blank + '%s = %s' % (all_variable_name[0], content))
                        new_source_code_list.append(blank + 'print %s' % all_variable_name[0])
                        extra_words.append(all_variable_name[0])
                        if len(all_variable_name) > 1:
                            all_variable_name = all_variable_name[1:]
                        else:
                            all_variable_name[0] += '_'
                    else:
                        new_source_code_list.append(line)
                        continue
                else:
                    new_source_code_list.append(line)
            source_code = '\n'.join(new_source_code_list)
        all_source_codes.append(source_code)
    
    source_code = all_source_codes[-1]
    if random.random() < code_style[2][0]:
        if lang in ['c', 'java']:
            all_unary_operator = ['++', '--']
            for unary_operator in all_unary_operator:
                index = 0
                while source_code.find(unary_operator, index) != -1:
                    # print('index', index, flush=True)
                    # print('source_code', source_code, flush=True)
                    index = source_code.find(unary_operator, index)
                    
                    if index == -1:
                        break
                    
                    # 11.24: 修复了会导致程序卡住的问题... 如果有连续三个以上的'+'或'-'，跳过
                    if index+2 < len(source_code) and source_code[index+2] == unary_operator[0]:
                        while source_code[index] == unary_operator[0]:
                            index += 1
                        continue
                    
                    # 如果找到了 i++ 或 i-- 的形式，替换
                    pre_code = source_code[:index]
                    after_code = source_code[index + len(unary_operator):]
                    temp_var = source_code[:index].strip().split('\n')[-1].split(';')[-1].split('(')[-1].split(' ')[-1]
                    
                    if not is_valid_variable_name(temp_var, lang): # 11.24: 排除a[i++]提取出a[i的情况，以及其他不合法变量名的情况
                        index += len(unary_operator)
                        continue

                    source_code = pre_code + '=' + temp_var + unary_operator[0] + '1' + after_code
        all_source_codes.append(source_code)

    source_code = all_source_codes[-1]
    if random.random() < code_style[1][0]:
        # print('code_style[1], change +=, -=, ...', flush=True)
        all_combined_assignments = []
        if lang == 'python':
            all_combined_assignments = ['+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=', '//=', '**=']
        elif lang == 'java':
            all_combined_assignments = ['+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=', '>>>=']
        
        for combined_assignments in all_combined_assignments:
            # print('*' * 100, flush=True)
            # print('source_code', source_code, flush=True)
            # print('combined_assignments', combined_assignments, flush=True)
            index = 0
            while source_code.find(combined_assignments, index) != -1:
                # print('source_code', source_code, flush=True)
                # print('index', index, flush=True)
                index = source_code.find(combined_assignments, index)
                if index == -1:
                    break
                pre_code = source_code[:index]
                after_code = source_code[index + len(combined_assignments):]
                temp_var = source_code[:index].strip().split('\n')[-1].split(';')[-1].split('(')[-1].split(' ')[-1]
                source_code = pre_code + '=' + temp_var + combined_assignments[:-1] + after_code

        all_source_codes.append(source_code)
    
    return all_source_codes[-1], extra_words, len(all_source_codes) > 0
