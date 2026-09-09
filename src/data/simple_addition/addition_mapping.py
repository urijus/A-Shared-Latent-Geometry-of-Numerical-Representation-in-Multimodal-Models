ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}

OPERATOR_TO_WORD = {
    "+": "plus",
    "=": "equals",
    "*": "times",
    "-": "minus"
}


def number_to_word(value):
    if not 0 <= value <= 99:
        raise ValueError(f"Expected value in [0, 99], got {value}")

    if value < 20:
        return ONES[value]

    tens = value // 10 * 10
    ones = value % 10

    if ones == 0:
        return TENS[tens]

    return f"{TENS[tens]} {ONES[ones]}"


NUMBER_TO_WORD = {
    value: number_to_word(value)
    for value in range(100)
}


def tokens_to_words(a, b, operator):
    return [
        NUMBER_TO_WORD[a],
        OPERATOR_TO_WORD[operator],
        NUMBER_TO_WORD[b],
        OPERATOR_TO_WORD["="]
    ]


def addition_expr_to_words(a, b, prompt="Output ONLY a number."):
    return f"{prompt} {' '.join(tokens_to_words(a, b, '+'))}"

def subtraction_expr_to_words(a, b, prompt="Output ONLY a number."):
    return f"{prompt} {' '.join(tokens_to_words(a, b, '-'))}"

def multiplication_expr_to_words(a, b, prompt="Output ONLY a number."):
    return f"{prompt} {' '.join(tokens_to_words(a, b, '*'))}"


if __name__=="__main__":
    print(addition_expr_to_words(25, 32))